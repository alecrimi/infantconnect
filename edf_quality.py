"""EDF integrity and signal-quality check with a pop-up report window.

Usage from the main window:

    from edf_quality import show_edf_quality_dialog
    ...
    self.edf_quality_button.clicked.connect(lambda: show_edf_quality_dialog(self))
"""
import html
import re
import warnings
from pathlib import Path

import mne
import numpy as np
from scipy.signal import welch

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QMessageBox,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout,
)

# ---------------------------------------------------------------- thresholds
MONTAGE_NAMES = "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2".split()
MIN_SFREQ_HZ = 100.0          # below this: warn
MIN_DURATION_S = 10.0         # shorter recordings: warn
VIEWER_MAX_S = 60.0           # the viewer crops to this length for analysis
FLAT_WINDOW_S = 1.0           # window used to detect flat segments
FLAT_PTP_UV = 1e-4            # peak-to-peak below this = flat window
FLAT_WARN_FRAC = 0.05         # fraction of flat windows -> warn
FLAT_BAD_FRAC = 0.50          # fraction of flat windows -> bad
FLAT_RUN_WARN_S = 2.0         # longest run of identical samples -> warn
LOW_STD_UV = 0.5              # very low variance -> warn
HIGH_AMP_UV = 500.0           # 99th pct |x - median| above this -> warn
CLIP_FRAC = 0.005             # fraction of samples stuck at min/max -> warn
LINE_RATIO_WARN = 8.0         # line-noise power / neighbouring power -> warn
CORR_DUP = 0.995              # |r| above this between channels -> warn

SEV_ORDER = {"CRITICAL": 0, "WARN": 1, "INFO": 2}


# --------------------------------------------------------------- helpers
def _longest_run(mask):
    """Length (samples) of the longest run of True in a boolean array."""
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max())


def _match_1020(name):
    """Map an EDF channel label to a 10-20 name (same rules as the viewer)."""
    lookup = {n.lower(): n for n in MONTAGE_NAMES}
    lookup.update({"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"})
    clean = re.sub(r"^EEG[\s_-]*", "", name, flags=re.IGNORECASE)
    clean = re.sub(r"[-_\s]*(REF|LE|AV|A1|A2)$", "", clean, flags=re.IGNORECASE)
    return lookup.get(clean.strip().lower())


def _num(raw_bytes, cast):
    return cast(raw_bytes.decode("ascii", "ignore").strip())


def _check_header(path, add):
    """Validate the raw EDF header and file size without MNE."""
    info = {"labels": [], "nsamp": [], "rec_dur": None, "n_records": None,
            "ns": 0, "discontinuous": False, "edf_plus": False}
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(256)
        if len(head) < 256:
            add("CRITICAL", f"File is only {size} bytes: shorter than the 256-byte EDF header.")
            return info
        if head[:8] != b"0       ":
            add("CRITICAL", "Version field is not '0' - this is not a standard EDF file "
                            "(BDF or corrupted header).")
            return info
        try:
            hdr_bytes = _num(head[184:192], int)
            n_rec = _num(head[236:244], int)
            rec_dur = _num(head[244:252], float)
            ns = _num(head[252:256], int)
        except ValueError:
            add("CRITICAL", "Header numeric fields (header size / records / duration / "
                            "signal count) cannot be parsed.")
            return info
        reserved = head[192:236].decode("ascii", "ignore")
        info["edf_plus"] = reserved.startswith("EDF+")
        info["discontinuous"] = reserved.startswith("EDF+D")
        info.update(rec_dur=rec_dur, n_records=n_rec, ns=ns)

        if ns <= 0:
            add("CRITICAL", f"Header declares {ns} signals.")
            return info
        sig = f.read(256 * ns)
        if len(sig) < 256 * ns:
            add("CRITICAL", "File ends inside the per-signal header (truncated).")
            return info

    if hdr_bytes != 256 + 256 * ns:
        add("WARN", f"Header size field ({hdr_bytes}) does not match 256 + 256*{ns} "
                    f"= {256 + 256 * ns}.")
    if rec_dur <= 0:
        add("CRITICAL", f"Data record duration is {rec_dur} s (must be > 0).")

    def field(offset_per_sig, width, i):
        start = ns * offset_per_sig + i * width
        return sig[start:start + width]

    labels = [field(0, 16, i).decode("ascii", "ignore").strip() for i in range(ns)]
    info["labels"] = labels
    try:
        nsamp = [_num(field(216, 8, i), int) for i in range(ns)]
    except ValueError:
        add("CRITICAL", "Samples-per-record fields cannot be parsed.")
        return info
    info["nsamp"] = nsamp

    # per-signal physical / digital range sanity (gain would be 0 or inf)
    for i, lab in enumerate(labels):
        if lab.startswith("EDF Annotations"):
            continue
        try:
            pmin = _num(field(104, 8, i), float)
            pmax = _num(field(112, 8, i), float)
            dmin = _num(field(120, 8, i), float)
            dmax = _num(field(128, 8, i), float)
        except ValueError:
            add("WARN", f"Channel '{lab}': physical/digital min/max fields unreadable.")
            continue
        if pmax <= pmin or dmax <= dmin:
            add("CRITICAL", f"Channel '{lab}': invalid calibration range "
                            f"(physical {pmin}..{pmax}, digital {dmin}..{dmax}).")

    # file-size consistency
    rec_bytes = sum(nsamp) * 2
    if n_rec == -1:
        n_complete = (size - hdr_bytes) // rec_bytes if rec_bytes > 0 else 0
        add("WARN", f"Header records count is unknown (-1); file holds {n_complete} "
                    "complete data records (recording may not have been finalised).")
    elif rec_bytes > 0:
        expected = hdr_bytes + n_rec * rec_bytes
        if size < expected:
            n_complete = max(0, (size - hdr_bytes) // rec_bytes)
            add("CRITICAL", f"File is truncated: header declares {n_rec} records "
                            f"({expected} bytes) but only {size} bytes exist "
                            f"(~{n_complete} complete records, "
                            f"{n_complete * rec_dur:.1f} s).")
        elif size > expected:
            add("WARN", f"File has {size - expected} unexpected trailing bytes after the "
                        "last declared data record.")

    if info["discontinuous"]:
        add("WARN", "EDF+D (discontinuous) recording: time gaps may exist between records.")
    return info


def _fmt(x, nd=2):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


# ---------------------------------------------------------------- analysis
def analyze_edf(path):
    """Return a dict describing the integrity / quality of an EDF file."""
    path = Path(path)
    issues = []

    def add(sev, msg):
        issues.append((sev, msg))

    report = {
        "file": path.name, "path": str(path),
        "size_mb": path.stat().st_size / 1e6,
        "issues": issues, "channels": [], "summary": {}, "read_error": None,
    }

    header = _check_header(path, add)
    report["header"] = header

    # ---- sampling-rate consistency from the header
    if header["rec_dur"] and header["nsamp"]:
        rates = {}
        for lab, n in zip(header["labels"], header["nsamp"]):
            if lab.startswith("EDF Annotations"):
                continue
            rates.setdefault(round(n / header["rec_dur"], 4), []).append(lab)
        report["header_rates"] = rates
        if len(rates) > 1:
            txt = "; ".join(f"{r:g} Hz: {len(v)} ch" for r, v in rates.items())
            add("WARN", f"Mixed sampling rates in file ({txt}). MNE resamples slower "
                        "channels to the highest rate, which can distort them.")

    # ---- read with MNE
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
        for w in wlist:
            add("WARN", "Reader warning: " + str(w.message).splitlines()[0])
    except Exception as exc:
        report["read_error"] = f"{type(exc).__name__}: {exc}"
        add("CRITICAL", f"MNE could not read the file: {type(exc).__name__}: {exc}")
        return report

    sfreq = float(raw.info["sfreq"])
    nyq = sfreq / 2.0
    data = raw.get_data() * 1e6          # volts -> microvolts
    n_ch, n_t = data.shape
    duration = n_t / sfreq if sfreq > 0 else 0.0
    ch_names = list(raw.ch_names)

    meas = raw.info.get("meas_date")
    report["summary"] = {
        "Sampling rate": f"{sfreq:g} Hz (Nyquist {nyq:g} Hz)",
        "Channels": str(n_ch),
        "Samples per channel": str(n_t),
        "Duration": f"{duration:.2f} s ({duration / 60:.2f} min)",
        "File size": f"{report['size_mb']:.2f} MB",
        "Recording date": str(meas) if meas is not None else "not stored",
        "Header high-pass": f"{raw.info['highpass']:g} Hz",
        "Header low-pass": f"{raw.info['lowpass']:g} Hz",
        "Annotations": str(len(raw.annotations)),
        "EDF flavour": ("EDF+D" if header["discontinuous"]
                        else "EDF+C" if header["edf_plus"] else "EDF"),
    }

    # ---- recording-level checks
    if sfreq < MIN_SFREQ_HZ:
        add("WARN", f"Low sampling rate ({sfreq:g} Hz < {MIN_SFREQ_HZ:g} Hz): "
                    "limited bandwidth for EEG analysis.")
    if raw.info["lowpass"] > nyq + 1e-6:
        add("WARN", f"Header low-pass ({raw.info['lowpass']:g} Hz) exceeds the Nyquist "
                    f"frequency ({nyq:g} Hz).")
    if raw.info["highpass"] > 0 and raw.info["lowpass"] > 0 \
            and raw.info["highpass"] >= raw.info["lowpass"]:
        add("WARN", "Header high-pass is not below low-pass - filter metadata looks invalid.")
    if duration < MIN_DURATION_S:
        add("WARN", f"Very short recording ({duration:.1f} s): connectivity estimates "
                    "will be unreliable.")
    if duration > VIEWER_MAX_S:
        add("INFO", f"Recording is longer than {VIEWER_MAX_S:g} s; the viewer only analyses "
                    f"the first {VIEWER_MAX_S:g} s.")
    if meas is None:
        add("INFO", "No recording date/time stored in the header.")
    if len(set(ch_names)) != len(ch_names):
        add("WARN", "Duplicate channel names in file.")
    if n_ch == 0:
        add("CRITICAL", "No data channels found.")
        return report

    # ---- 10-20 montage match (needed by LORETA / effective connectivity)
    matched, unmatched = {}, []
    for ch in ch_names:
        tgt = _match_1020(ch)
        if tgt and tgt not in matched.values():
            matched[ch] = tgt
        else:
            unmatched.append(ch)
    report["summary"]["10-20 channels matched"] = f"{len(matched)} of {n_ch}"
    if len(matched) < 4:
        add("CRITICAL", f"Only {len(matched)} channel(s) match the 10-20 montage; the viewer "
                        "needs at least 4 for LORETA / effective connectivity.")
    if unmatched:
        add("INFO", "Channels ignored by the viewer (no 10-20 match): "
                    + ", ".join(unmatched))

    # ---- per-channel analysis
    win = max(int(round(sfreq * FLAT_WINDOW_S)), 2)
    n_win = n_t // win
    clean = np.zeros_like(data)
    for i, ch in enumerate(ch_names):
        x = data[i]
        finite = np.isfinite(x)
        n_nan = int(np.isnan(x).sum())
        n_inf = int(np.isinf(x).sum())
        xf = x[finite]
        notes = []
        status = "OK"

        def bump(level):
            nonlocal status
            order = {"OK": 0, "WARN": 1, "BAD": 2}
            if order[level] > order[status]:
                status = level

        rec = {"name": ch, "n_nan": n_nan, "n_inf": n_inf, "mean": None, "std": None,
               "min": None, "max": None, "flat_frac": None, "flat_run_s": None,
               "clip_frac": None, "line_ratio": None}

        if n_nan or n_inf:
            notes.append(f"{n_nan} NaN, {n_inf} Inf ({100 * (1 - finite.mean()):.2f}%)")
            bump("BAD")
        if xf.size < 2:
            notes.append("no valid samples")
            bump("BAD")
            rec.update(status=status, notes=notes)
            report["channels"].append(rec)
            continue

        rec.update(mean=float(xf.mean()), std=float(xf.std()),
                   min=float(xf.min()), max=float(xf.max()))
        xc = np.where(finite, x - xf.mean(), 0.0)
        clean[i] = xc

        # flat windows
        if n_win >= 1:
            w = np.where(finite, x, 0.0)[: n_win * win].reshape(n_win, win)
            flat_frac = float((np.ptp(w, axis=1) < FLAT_PTP_UV).mean())
        else:
            flat_frac = float(rec["std"] < 1e-6)
        run_s = _longest_run(np.diff(xf) == 0) / sfreq
        rec["flat_frac"], rec["flat_run_s"] = flat_frac, run_s
        if flat_frac >= FLAT_BAD_FRAC or rec["std"] < 1e-6:
            notes.append(f"flat/dead signal ({100 * flat_frac:.0f}% flat windows)")
            bump("BAD")
        elif flat_frac >= FLAT_WARN_FRAC:
            notes.append(f"{100 * flat_frac:.0f}% of 1 s windows are flat")
            bump("WARN")
        if run_s >= FLAT_RUN_WARN_S and rec["std"] >= 1e-6:
            notes.append(f"constant for {run_s:.1f} s in a row")
            bump("WARN")

        # variance / amplitude
        if 1e-6 <= rec["std"] < LOW_STD_UV:
            notes.append(f"very low variance (SD {rec['std']:.3f} uV)")
            bump("WARN")
        p99 = float(np.percentile(np.abs(xf - np.median(xf)), 99))
        if p99 > HIGH_AMP_UV:
            notes.append(f"very large amplitude (99th pct {p99:.0f} uV)")
            bump("WARN")

        # clipping / saturation
        if rec["std"] >= 1e-6:
            clip = max(float((xf == xf.max()).mean()), float((xf == xf.min()).mean()))
            rec["clip_frac"] = clip
            if clip > CLIP_FRAC and (xf == xf.max()).sum() + (xf == xf.min()).sum() >= 10:
                notes.append(f"possible clipping ({100 * clip:.2f}% at min/max)")
                bump("WARN")

        # line noise
        if n_t >= 64 and rec["std"] >= 1e-6:
            nperseg = min(n_t, int(4 * sfreq))
            f, psd = welch(xc, fs=sfreq, nperseg=nperseg)
            ratios = []
            for mains in (50.0, 60.0):
                if mains + 1.0 >= nyq:
                    continue
                line = psd[(f >= mains - 1) & (f <= mains + 1)]
                refs = []
                for lo, hi in ((mains - 8, mains - 3), (mains + 3, mains + 8)):
                    if hi < nyq:
                        band = psd[(f >= lo) & (f <= hi)]
                        if band.size:
                            refs.append(band.mean())
                if line.size and refs and np.mean(refs) > 0:
                    ratios.append((float(line.mean() / np.mean(refs)), mains))
            if ratios:
                ratio, mains = max(ratios)
                rec["line_ratio"] = ratio
                if ratio > LINE_RATIO_WARN:
                    notes.append(f"strong {mains:g} Hz mains interference (x{ratio:.0f})")
                    bump("WARN")

        rec.update(status=status, notes=notes)
        report["channels"].append(rec)

    # ---- inter-channel checks
    good = [i for i, r in enumerate(report["channels"])
            if r.get("std") and r["std"] >= 1e-6]
    if len(good) > 1:
        step = max(1, n_t // 20000)
        c = np.corrcoef(clean[good][:, ::step])
        dup = []
        for a in range(len(good)):
            for b in range(a + 1, len(good)):
                if abs(c[a, b]) > CORR_DUP:
                    dup.append((ch_names[good[a]], ch_names[good[b]], c[a, b]))
        for a, b, r in dup:
            add("WARN", f"Channels {a} and {b} are nearly identical (r = {r:.4f}): "
                        "possible shorted electrodes or duplicated channel.")
            for rec in report["channels"]:
                if rec["name"] in (a, b) and rec["status"] == "OK":
                    rec["status"] = "WARN"
                    rec["notes"].append("near-duplicate of another channel")

    # ---- roll-up of channel issues
    bad = [r["name"] for r in report["channels"] if r["status"] == "BAD"]
    warn = [r["name"] for r in report["channels"] if r["status"] == "WARN"]
    if bad:
        add("CRITICAL", f"{len(bad)} bad channel(s): {', '.join(bad)}")
    if warn:
        add("WARN", f"{len(warn)} channel(s) with warnings: {', '.join(warn)}")
    total_bad = sum(r["n_nan"] + r["n_inf"] for r in report["channels"])
    if total_bad == 0:
        add("INFO", "No missing (NaN) or infinite values found.")
    else:
        add("CRITICAL", f"{total_bad} missing/invalid samples in total.")

    issues.sort(key=lambda t: SEV_ORDER[t[0]])
    return report


def verdict(report):
    sevs = {s for s, _ in report["issues"]}
    if "CRITICAL" in sevs:
        return "FAIL", "#C62828"
    if "WARN" in sevs:
        return "PASS WITH WARNINGS", "#E69500"
    return "PASS", "#2E7D32"


def report_to_text(report):
    v, _ = verdict(report)
    lines = [f"EDF quality report: {report['file']}", f"Path: {report['path']}",
             f"Verdict: {v}", ""]
    for k, val in report["summary"].items():
        lines.append(f"{k}: {val}")
    lines += ["", "Findings:"]
    lines += [f"  [{s}] {m}" for s, m in report["issues"]]
    if report["channels"]:
        lines += ["", "Channels:"]
        for r in report["channels"]:
            lines.append(
                f"  {r['name']:<10} {r['status']:<5} mean={_fmt(r['mean'])} "
                f"sd={_fmt(r['std'])} min={_fmt(r['min'])} max={_fmt(r['max'])} uV "
                f"NaN={r['n_nan']} Inf={r['n_inf']} "
                f"flat={_fmt(None if r['flat_frac'] is None else 100 * r['flat_frac'], 0)}% "
                + ("; ".join(r["notes"]) if r["notes"] else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------- dialog
class EDFQualityDialog(QDialog):
    COLUMNS = ["Channel", "Status", "Mean (µV)", "SD (µV)", "Min (µV)", "Max (µV)",
               "NaN", "Inf", "Flat %", "Longest const. (s)", "Clip %",
               "Line-noise ×", "Notes"]
    COLORS = {"OK": "#C8E6C9", "WARN": "#FFE0A3", "BAD": "#F4B6B6"}

    def __init__(self, report, parent=None):
        super().__init__(parent)
        self.report = report
        self.setWindowTitle(f"EDF Quality Check - {report['file']}")
        self.resize(1100, 640)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs, stretch=1)

        summary = QTextEdit()
        summary.setReadOnly(True)
        summary.setHtml(self._summary_html())
        tabs.addTab(summary, "Summary")

        if report["channels"]:
            tabs.addTab(self._channel_table(), f"Channels ({len(report['channels'])})")

        row = QHBoxLayout()
        layout.addLayout(row)
        row.addStretch(1)
        save_btn = QPushButton("Save report…")
        save_btn.clicked.connect(self._save)
        row.addWidget(save_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)

    def _summary_html(self):
        r = self.report
        v, color = verdict(r)
        e = html.escape
        parts = [
            f"<h2>{e(r['file'])}</h2>",
            f"<h3 style='color:{color}'>Verdict: {v}</h3>",
        ]
        if r["summary"]:
            parts.append("<table cellpadding='3'>")
            for k, val in r["summary"].items():
                parts.append(f"<tr><td><b>{e(k)}</b></td><td>{e(val)}</td></tr>")
            parts.append("</table>")
        sev_color = {"CRITICAL": "#C62828", "WARN": "#E69500", "INFO": "#1565C0"}
        parts.append("<h3>Findings</h3><ul>")
        for sev, msg in r["issues"]:
            parts.append(f"<li><b style='color:{sev_color[sev]}'>{sev}</b> - {e(msg)}</li>")
        parts.append("</ul>")
        return "".join(parts)

    def _channel_table(self):
        rows = self.report["channels"]
        t = QTableWidget(len(rows), len(self.COLUMNS))
        t.setHorizontalHeaderLabels(self.COLUMNS)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setAlternatingRowColors(False)
        for i, r in enumerate(rows):
            flat = None if r["flat_frac"] is None else 100 * r["flat_frac"]
            clip = None if r["clip_frac"] is None else 100 * r["clip_frac"]
            vals = [r["name"], r["status"], _fmt(r["mean"]), _fmt(r["std"]),
                    _fmt(r["min"]), _fmt(r["max"]), str(r["n_nan"]), str(r["n_inf"]),
                    _fmt(flat, 1), _fmt(r["flat_run_s"], 2), _fmt(clip, 2),
                    _fmt(r["line_ratio"], 1), "; ".join(r["notes"])]
            for j, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setBackground(QColor(self.COLORS[r["status"]]))
                item.setForeground(QColor("#111111"))
                if j == 1:
                    f = QFont()
                    f.setBold(True)
                    item.setFont(f)
                t.setItem(i, j, item)
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(
            len(self.COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        return t

    def _save(self):
        default = str(Path.home() / f"{Path(self.report['file']).stem}_quality.txt")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Quality Report", default, "Text (*.txt);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog)
        if not path:
            return
        try:
            Path(path).write_text(report_to_text(self.report), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", f"{type(exc).__name__}: {exc}")


def show_edf_quality_dialog(parent=None):
    """Ask for an EDF file, analyse it and show the report in a pop-up."""
    path, _ = QFileDialog.getOpenFileName(
        parent, "Select EDF File to Check", str(Path.home()),
        "EDF Files (*.edf);;All Files (*)",
        options=QFileDialog.Option.DontUseNativeDialog)
    if not path:
        return
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        report = analyze_edf(path)
    except Exception as exc:
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(parent, "EDF Quality Check",
                             f"{type(exc).__name__}: {exc}")
        return
    QApplication.restoreOverrideCursor()
    EDFQualityDialog(report, parent).exec()
