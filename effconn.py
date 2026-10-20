# effconn.py
"""
Self-contained effective-connectivity estimators.

Provides:
    compute_granger(data, lag=5)              -> (n_ch, n_ch) matrix
    compute_transfer_entropy(data, bins=4, lag=1) -> (n_ch, n_ch) matrix
    compute_effective_connectivity(data, method='granger', **kwargs)

Conventions
-----------
* `data` is (n_samples, n_channels).
* Returned matrix `M` is directed: `M[i, j]` measures the influence **from j → i**.
* `compute_granger` returns ``-log10(p)`` (values ≳ 1.3 = significant at p<0.05).
* `compute_transfer_entropy` returns natural-log TE (nats).
"""
import numpy as np
from scipy.stats import f as _f_dist


# ---------------------------------------------------------------------------
# Granger causality (pairwise, based on OLS/VAR residuals)
# ---------------------------------------------------------------------------
def compute_granger(data, lag=5):
    """
    Pairwise Granger causality with an F-test.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
    lag  : int, VAR model order

    Returns
    -------
    matrix : ndarray, shape (n_channels, n_channels)
        `matrix[i, j]` = -log10(p) of the F-test that j's past improves the
        prediction of i beyond i's own past.  Zero means no test / failed fit.
    """
    data = np.asarray(data, dtype=float)
    T, n = data.shape
    if n < 2 or T <= lag + 2 or lag < 1:
        return np.zeros((n, n))

    Y = data[lag:]                           # (T-lag, n) targets
    T_eff = Y.shape[0]

    matrix = np.zeros((n, n))

    for i in range(n):
        y_i = Y[:, i]

        # Reduced design matrix: [1, i(t-1), ..., i(t-lag)]
        X_red = np.ones((T_eff, 1 + lag))
        for l in range(1, lag + 1):
            X_red[:, l] = data[lag - l:-l, i]

        try:
            beta_red, *_ = np.linalg.lstsq(X_red, y_i, rcond=None)
            rss_red = float(np.sum((y_i - X_red @ beta_red) ** 2))
        except np.linalg.LinAlgError:
            continue

        df1 = lag
        df2 = T_eff - 2 * lag - 1
        if df2 <= 0 or rss_red <= 0:
            continue

        for j in range(n):
            if j == i:
                continue

            # Full design matrix: [1, i(t-1..lag), j(t-1..lag)]
            X_full = np.ones((T_eff, 1 + 2 * lag))
            for l in range(1, lag + 1):
                X_full[:, l] = data[lag - l:-l, i]
                X_full[:, lag + l] = data[lag - l:-l, j]

            try:
                beta_full, *_ = np.linalg.lstsq(X_full, y_i, rcond=None)
                rss_full = float(np.sum((y_i - X_full @ beta_full) ** 2))
            except np.linalg.LinAlgError:
                continue

            if rss_full <= 1e-15:
                continue

            f_stat = ((rss_red - rss_full) / df1) / (rss_full / df2)
            p_val = 1.0 - _f_dist.cdf(max(f_stat, 0.0), df1, df2)
            p_val = max(p_val, 1e-10)

            matrix[i, j] = -np.log10(p_val)

    return matrix


# ---------------------------------------------------------------------------
# Transfer entropy (pairwise, binned estimator)
# ---------------------------------------------------------------------------
def _quantile_bin(x, bins):
    """Discretize `x` into ~`bins` quantile bins. Robust to ties & constants."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=int)
    std = x.std()
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(x, dtype=int)
    q = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    return np.digitize(x, q)


def _pairwise_te(x_disc, y_disc, bins, lag):
    """
    TE(X → Y) with a binned plug-in estimator:
        TE = Σ p(y_f, y_p, x_p) · log[ p(y_f,y_p,x_p)·p(y_p)
                                     / (p(y_f,y_p)·p(y_p,x_p)) ]
    """
    n = len(x_disc)
    if n <= lag + 1:
        return 0.0

    x_past = x_disc[:-lag]
    y_past = y_disc[:-lag]
    y_future = y_disc[lag:]
    m = len(x_past)

    # Flatten 3D histogram in one bincount call
    flat = y_future * bins * bins + y_past * bins + x_past
    hist = np.bincount(flat, minlength=bins ** 3).reshape(bins, bins, bins)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    joint = hist / total

    p_yp_xp = joint.sum(axis=0)          # (bins, bins) - p(y_p, x_p)
    p_yf_yp = joint.sum(axis=2)          # (bins, bins) - p(y_f, y_p)
    p_yp = p_yp_xp.sum(axis=1)           # (bins,)      - p(y_p)

    # Vectorized computation of the inner sum over nonzero entries
    yf_idx, yp_idx, xp_idx = np.nonzero(joint)
    pj = joint[yf_idx, yp_idx, xp_idx]
    num = pj * p_yp[yp_idx]
    den = p_yf_yp[yf_idx, yp_idx] * p_yp_xp[yp_idx, xp_idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0, num / den, 1.0)
        te = float(np.sum(pj * np.log(ratio)))

    return max(te, 0.0)


def compute_transfer_entropy(data, bins=4, lag=1):
    """
    Pairwise transfer entropy.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
    bins : int, number of quantization bins per channel
    lag  : int, embedding lag (samples)

    Returns
    -------
    matrix : ndarray, shape (n_channels, n_channels)
        `matrix[i, j]` = TE from j → i (nats).
    """
    data = np.asarray(data, dtype=float)
    T, n = data.shape
    if n < 2 or T <= lag + 2:
        return np.zeros((n, n))

    disc = np.column_stack([_quantile_bin(data[:, k], bins) for k in range(n)])
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            matrix[i, j] = _pairwise_te(disc[:, j], disc[:, i], bins, lag)

    return matrix


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def compute_effective_connectivity(data, method="granger", **kwargs):
    """
    Unified dispatcher.

    method:
        'granger' / 'gc'          -> compute_granger(data, **kwargs)
        'transfer_entropy' / 'te' -> compute_transfer_entropy(data, **kwargs)
    """
    method = method.lower()
    if method in ("granger", "gc"):
        return compute_granger(data, **kwargs)
    if method in ("transfer_entropy", "te"):
        return compute_transfer_entropy(data, **kwargs)
    raise ValueError(f"Unknown effective-connectivity method: {method!r}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_samples = 3000

    # x0 drives x1 with a 1-sample delay; x2 is independent noise
    x0 = rng.normal(size=n_samples)
    x1 = np.zeros(n_samples)
    x2 = rng.normal(size=n_samples)
    for t in range(1, n_samples):
        x1[t] = 0.7 * x1[t - 1] + 0.4 * x0[t - 1] + 0.1 * rng.normal()

    data = np.column_stack([x0, x1, x2])

    print("Granger (-log10 p):")
    print(np.round(compute_granger(data, lag=2), 3))

    print("\nTransfer entropy (nats):")
    print(np.round(compute_transfer_entropy(data, bins=4, lag=1), 4))
    print("\nExpected: strong edge 0 -> 1 in both matrices.")
