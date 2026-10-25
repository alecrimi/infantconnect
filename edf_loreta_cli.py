"""Command-line LORETA-style forward/inverse modelling of an EDF recording.

Headless version of the viewer's "Forward Modeling (LORETA)" step (no Qt). For each
EDF file it writes, next to the input or into -o:

  <stem>_loreta_sources.csv       one row per volume source: head-frame position (mm),
                                  position aligned to the infant brain (viewer units),
                                  peak dSPM power and power normalised to 0..1
  <stem>_loreta_channel_corr.csv  channel x channel correlation (the viewer's connectome)
  <stem>_loreta_<hemi>.vtp        brain surface per hemisphere with point data
                                  'loreta_power' (open in ParaView / pyvista)
  <stem>_loreta_summary.json      parameters, template used, peak source, top sources
  <stem>_loreta_view.png          optional off-screen render (--screenshot)

The script must find the viewer's atlas.json and mesh files: put it in the viewer
folder or pass --root. Infant scalp templates are downloaded on first use into
--subjects-dir (same default as the viewer).

Examples:
  python edf_loreta_cli.py rec.edf --age 12 -o out/
  python edf_loreta_cli.py data/*.edf --age 6 --surface inflated --screenshot
  python edf_loreta_cli.py rec.edf --age 12 --no-surface      # CSV/JSON only
"""
import argparse
import csv
import json
import re
import sys
import traceback
import zipfile
from pathlib import Path

import mne
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
DEFAULT_SUBJECTS_DIR = Path.home() / "Documents" / "infant_viewer" / "mne_templates"

MONTAGE_NAMES = "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2".split()
INFANT_SCALP_TEMPLATES = {
    "0mo": 0, "3mo": 3, "6mo": 6, "9mo": 9,
    "12mo": 12, "18mo": 18, "2yr": 24,
}
SUBJECT_DIR_NAMES = {
    "0mo": "ANTS0-0Months3T", "3mo": "ANTS3-0Months3T",
    "6mo": "ANTS6-0Months3T", "9mo": "ANTS9-0Months3T",
    "12mo": "ANTS12-0Months3T", "18mo": "ANTS18-0Months3T",
    "2yr": "ANTS2-0Years3T",
}
TEMPLATE_URL = (
    "https://github.com/christian-oreilly/infant_template_paper/"
    "releases/download/v0.1-alpha/{subject}.zip"
)
SCALP_ZIP_CANDIDATES = (
    "bem/outer_skin.surf", "bem/outer_skin_large.surf", "bem/head.surf",
)
SKULL_ZIP_CANDIDATES = (
    "bem/inner_skull.surf", "bem/inner_skull_large.surf",
    "bem/outer_skull.surf", "bem/outer_skull_large.surf",
)
MONTAGE_FIF_CANDIDATES = ("montages/10-20-montage.fif",)
MONTAGE_SURF_CANDIDATES = ("surf/10-20-montage.surf",)
MONTAGE_ALL_CANDIDATES = MONTAGE_FIF_CANDIDATES + MONTAGE_SURF_CANDIDATES
SURFACE_TYPES = ("outer", "middle", "inner", "inflated")


# ------------------------------------------------------------ template download
def _find_zip_entry(zf, suffix):
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if name == suffix or name.endswith("/" + suffix):
            return name
    return None


def fetch_scalp_and_montage(template_name, subjects_dir, log=print):
    subject = SUBJECT_DIR_NAMES[template_name]
    subject_dir = Path(subjects_dir) / subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    has_scalp = any((subject_dir / s).is_file() for s in SCALP_ZIP_CANDIDATES)
    has_montage = any((subject_dir / s).is_file() for s in MONTAGE_ALL_CANDIDATES)
    if has_scalp and has_montage:
        return subject_dir

    import pooch                                   # only needed for the download
    zip_path = Path(subjects_dir) / f"{subject}.zip"
    if not zip_path.is_file():
        log(f"[{template_name}] downloading {subject}.zip ...")
        pooch.retrieve(url=TEMPLATE_URL.format(subject=subject), known_hash=None,
                       fname=zip_path.name, path=str(subjects_dir), progressbar=True)

    log(f"[{template_name}] extracting scalp, skulls, + montage ...")
    with zipfile.ZipFile(zip_path) as zf:
        for suffix in SCALP_ZIP_CANDIDATES + SKULL_ZIP_CANDIDATES + MONTAGE_ALL_CANDIDATES:
            entry = _find_zip_entry(zf, suffix)
            if entry is None:
                continue
            dest = subject_dir / suffix
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry) as src, open(dest, "wb") as dst:
                dst.write(src.read())

    for base in ["inner_skull", "outer_skull", "outer_skin"]:
        large = subject_dir / "bem" / f"{base}_large.surf"
        standard = subject_dir / "bem" / f"{base}.surf"
        if large.is_file() and not standard.is_file():
            large.rename(standard)

    if not any((subject_dir / s).is_file() for s in SCALP_ZIP_CANDIDATES):
        raise FileNotFoundError(f"No scalp surface in {zip_path.name}")
    if not any((subject_dir / s).is_file() for s in MONTAGE_ALL_CANDIDATES):
        raise FileNotFoundError(f"No 10-20 montage in {zip_path.name}")
    zip_path.unlink()
    return subject_dir


def load_mne_infant_scalp(template_name, subjects_dir=DEFAULT_SUBJECTS_DIR, log=print):
    """Return (scalp_vertices[N,3], {electrode: xyz}) in the template's scalp frame."""
    subject_dir = fetch_scalp_and_montage(template_name, subjects_dir, log)
    scalp_path = next((subject_dir / s for s in SCALP_ZIP_CANDIDATES
                       if (subject_dir / s).is_file()), None)
    if scalp_path is None:
        raise FileNotFoundError(f"No scalp surface in {subject_dir}")
    vertices, _ = mne.read_surface(str(scalp_path))
    vertices = np.asarray(vertices, dtype=float)

    fif_path = next((subject_dir / s for s in MONTAGE_FIF_CANDIDATES
                     if (subject_dir / s).is_file()), None)
    surf_path = next((subject_dir / s for s in MONTAGE_SURF_CANDIDATES
                      if (subject_dir / s).is_file()), None)
    names = MONTAGE_NAMES
    if fif_path is not None:
        montage = mne.channels.read_dig_fif(str(fif_path), verbose="error")
        montage.rename_channels(
            {f"EEG{i:03d}": name for i, name in enumerate(names, start=1)})
        pos = montage.get_positions()["ch_pos"]
        raw_positions = {n: np.asarray(pos[n], dtype=float) * 1000.0
                         for n in names if n in pos}
    elif surf_path is not None:
        verts, _ = mne.read_surface(str(surf_path))
        if len(verts) < len(names):
            raise ValueError(f"{surf_path} has {len(verts)} vertices")
        raw_positions = {n: np.asarray(verts[i], dtype=float) for i, n in enumerate(names)}
    else:
        raise FileNotFoundError(f"No montage in {subject_dir}")

    tree = cKDTree(vertices)                       # snap electrodes to the scalp surface
    electrodes = {}
    for name, p in raw_positions.items():
        _, idx = tree.query(p)
        electrodes[name] = vertices[idx].copy()
    return vertices, electrodes


# --------------------------------------------------------------- geometry helpers
def fit_electrodes(electrode_positions, ref_bounds, margin=(1.12, 1.10), clearance=0.03):
    """Same affine fit as the viewer's fit_template_to_brain, applied to electrodes.

    ref_bounds = (xmin, xmax, ymin, ymax, zmin, zmax) of the brain reference surface.
    """
    E = np.vstack(list(electrode_positions.values()))
    e_lo, e_hi = E.min(axis=0), E.max(axis=0)
    b = np.asarray(ref_bounds, dtype=float)
    b_lo, b_hi = b[[0, 2, 4]], b[[1, 3, 5]]
    ex = max(e_hi[0] - e_lo[0], 1e-6)
    ey = max(e_hi[1] - e_lo[1], 1e-6)
    sx = margin[0] * (b_hi[0] - b_lo[0]) / ex
    sy = margin[1] * (b_hi[1] - b_lo[1]) / ey
    scale = np.array([sx, sy, 0.5 * (sx + sy)])
    src = np.array([(e_lo[0] + e_hi[0]) / 2, (e_lo[1] + e_hi[1]) / 2, e_hi[2]])
    dst = np.array([(b_lo[0] + b_hi[0]) / 2, (b_lo[1] + b_hi[1]) / 2,
                    b_hi[2] + clearance * (b_hi[2] - b_lo[2])])
    return {n: (np.asarray(p, dtype=float) - src) * scale + dst
            for n, p in electrode_positions.items()}


def _match_1020(name):
    lookup = {n.lower(): n for n in MONTAGE_NAMES}
    lookup.update({"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"})
    clean = re.sub(r"^EEG[\s_-]*", "", name, flags=re.IGNORECASE)
    clean = re.sub(r"[-_\s]*(REF|LE|AV|A1|A2)$", "", clean, flags=re.IGNORECASE)
    return lookup.get(clean.strip().lower())


def prep_raw(path, max_duration=60.0, start=0.0):
    """Read EDF, crop and rename channels to 10-20 names (as the viewer does)."""
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    if raw.n_times < 2:
        raise ValueError("The EDF file contains too few time samples.")
    if start > 0 or (max_duration and raw.times[-1] - start > max_duration):
        tmax = start + max_duration if max_duration else None
        raw.crop(tmin=start, tmax=min(tmax, raw.times[-1]) if tmax else None)
    new_names = {}
    for ch in raw.ch_names:
        target = _match_1020(ch)
        if target and target not in new_names.values():
            new_names[ch] = target
    raw.rename_channels(new_names)
    return raw


# ------------------------------------------------------------------- core model
def compute_loreta(raw, electrode_positions, scalp_center, ref_bounds,
                   head_radius_m=0.07, source_step_m=0.007, lambda2=0.1, log=print):
    """Run the viewer's sphere-model forward + dSPM inverse. Returns a result dict.

    `raw` must already use 10-20 channel names. Positions are in the template's frame;
    `ref_bounds` are the bounds of the brain reference surface for alignment.
    """
    keep = [ch for ch in raw.ch_names if ch in electrode_positions]
    if len(keep) < 4:
        raise ValueError(f"Only {len(keep)} EDF channels match the 10-20 montage "
                         f"(need at least 4). EDF channels: {raw.ch_names}")
    raw = raw.copy().pick(keep)

    pts = np.array([np.asarray(electrode_positions[ch], dtype=float) for ch in keep])
    rel = pts - np.asarray(scalp_center, dtype=float)
    mean_r = float(np.linalg.norm(rel, axis=1).mean())
    if not np.isfinite(mean_r) or mean_r <= 0:
        raise ValueError("Electrode positions are degenerate (all at one point).")
    rel_m = rel * (head_radius_m / mean_r)
    montage = mne.channels.make_dig_montage(
        ch_pos={ch: rel_m[i] for i, ch in enumerate(keep)}, coord_frame="head")
    raw.set_montage(montage, match_case=False, on_missing="raise", verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj(verbose=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        log("  computing forward solution ...")
        bem = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=head_radius_m,
                                    verbose=False)
        src_radius = head_radius_m * 0.75
        g = np.arange(-src_radius, src_radius + 1e-9, source_step_m)
        xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
        rr = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        rr = rr[np.linalg.norm(rr, axis=1) <= src_radius]
        nn = np.zeros_like(rr)
        nn[:, 2] = 1.0
        src = mne.setup_volume_source_space(pos=dict(rr=rr, nn=nn), verbose=False)
        log(f"  source space: {src[0]['nuse']} sources")

        fwd = mne.make_forward_solution(raw.info, trans=None, src=src, bem=bem,
                                        meg=False, eeg=True, mindist=3.0, verbose=False)
        cov = mne.make_ad_hoc_cov(raw.info, verbose=False)
        cd = cov["data"].copy()
        cd = np.maximum(cd, 1e-10) if cd.ndim == 1 else cd + np.eye(cd.shape[0]) * 1e-10
        cov["data"] = cd
        inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=1.0,
                                                     depth=0.8, verbose=False)
        log("  applying inverse ...")
        method = "dSPM"
        try:
            stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=lambda2,
                                                     method="dSPM", verbose=False)
        except Exception as exc:
            log(f"  dSPM failed ({exc}); falling back to MNE with lambda2=10")
            method = "MNE"
            stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=10.0,
                                                     method="MNE", verbose=False)

        data = np.nan_to_num(stc.data, nan=0.0, posinf=0.0, neginf=0.0)
        power = data.max(axis=1)

        vert = stc.vertices
        vert = vert[0] if isinstance(vert, (list, tuple)) else vert
        head_mm = src[0]["rr"][vert] * 1000.0
        s_lo, s_hi = head_mm.min(axis=0), head_mm.max(axis=0)
        s_size = np.maximum(s_hi - s_lo, 1e-6)
        s_center = (s_lo + s_hi) / 2.0
        b = np.asarray(ref_bounds, dtype=float)
        b_lo, b_hi = b[[0, 2, 4]], b[[1, 3, 5]]
        brain_pts = (head_mm - s_center) * ((b_hi - b_lo) / s_size) * 0.95 \
            + (b_lo + b_hi) / 2.0

        max_p = float(power.max())
        if max_p == 0 or np.isnan(max_p):
            max_p = 1e-12

        ch = raw.get_data()
        sd = ch.std(axis=1, keepdims=True)
        sd[sd == 0] = 1e-12
        z = (ch - ch.mean(axis=1, keepdims=True)) / sd
        corr = np.nan_to_num(z @ z.T / max(ch.shape[1], 1), nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "channels": keep, "method": method, "n_sources": int(power.size),
        "head_mm": head_mm, "brain_pts": brain_pts, "power": power,
        "max_power": max_p, "power_norm": power / max_p, "channel_corr": corr,
        "sfreq": float(raw.info["sfreq"]), "duration_s": float(raw.times[-1]),
    }


def map_to_surface(brain_pts, power_norm, mesh_points):
    """Nearest-source power for each surface vertex (as in the viewer)."""
    _, idx = cKDTree(brain_pts).query(mesh_points)
    return np.nan_to_num(power_norm[idx], nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------- viewer assets
class ViewerAssets:
    """atlas.json + brain meshes + infant scalp templates, as the viewer uses them."""

    def __init__(self, root=ROOT, subjects_dir=DEFAULT_SUBJECTS_DIR, log=print):
        self.root = Path(root)
        self.subjects_dir = Path(subjects_dir)
        self.log = log
        manifest = self.root / "atlas.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"atlas.json not found in {self.root} "
                                    "(use --root to point at the viewer folder).")
        self.atlas = json.loads(manifest.read_text(encoding="utf-8"))
        self.ages = sorted(int(a) for a, e in self.atlas.items()
                           if any(e.get(h, {}).get("outer") for h in ("left", "right")))
        if not self.ages:
            raise RuntimeError("No usable outer surfaces found in atlas.json.")
        self._mesh_cache, self._tpl_cache = {}, {}

    def nearest_age(self, age):
        return min(self.ages, key=lambda a: abs(a - age))

    def _mesh(self, age, hemi, surface):
        key = (age, hemi, surface)
        if key not in self._mesh_cache:
            rel = self.atlas.get(str(age), {}).get(hemi, {}).get(surface)
            if not rel:
                self._mesh_cache[key] = None
            else:
                import pyvista as pv                 # only needed to read meshes
                path = self.root / rel
                if not path.is_file():
                    raise FileNotFoundError(f"Mesh not found: {path}")
                mesh = pv.read(str(path))
                if not isinstance(mesh, pv.PolyData):
                    mesh = mesh.extract_surface()
                if mesh.n_points == 0:
                    raise ValueError(f"Empty surface mesh: {path}")
                self._mesh_cache[key] = mesh
        return self._mesh_cache[key]

    def surface_meshes(self, age, surface):
        return {h: m for h in ("left", "right")
                if (m := self._mesh(age, h, surface)) is not None}

    def reference_bounds(self, age):
        parts = list(self.surface_meshes(age, "outer").values())
        if not parts:
            raise ValueError("No brain surface available to align the sources to.")
        lo = np.min([[m.bounds[0], m.bounds[2], m.bounds[4]] for m in parts], axis=0)
        hi = np.max([[m.bounds[1], m.bounds[3], m.bounds[5]] for m in parts], axis=0)
        return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])

    def template(self, age):
        """Nearest infant scalp template that loads: (name, vertices, electrodes)."""
        for name, _ in sorted(INFANT_SCALP_TEMPLATES.items(),
                              key=lambda kv: abs(kv[1] - age)):
            if name in self._tpl_cache:
                return (name,) + self._tpl_cache[name]
            try:
                res = load_mne_infant_scalp(name, self.subjects_dir, self.log)
                self._tpl_cache[name] = res
                return (name,) + res
            except Exception as exc:
                self.log(f"  template {name} unavailable: {exc}")
        raise ValueError("No overlay template available.")


# ------------------------------------------------------------------------ outputs
def _write_matrix(path, matrix, names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + list(names))
        for n, row in zip(names, matrix):
            w.writerow([n] + [repr(float(v)) for v in row])


def _write_sources(path, res):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x_head_mm", "y_head_mm", "z_head_mm", "x_brain", "y_brain",
                    "z_brain", "power", "power_norm"])
        for h, b, p, pn in zip(res["head_mm"], res["brain_pts"], res["power"],
                               res["power_norm"]):
            w.writerow([*(repr(float(v)) for v in h), *(repr(float(v)) for v in b),
                        repr(float(p)), repr(float(pn))])


def _screenshot(meshes, path, view="axial", log=print):
    try:
        import pyvista as pv
        pl = pv.Plotter(off_screen=True, window_size=(1200, 900))
        pl.set_background("#202833")
        for m in meshes:
            pl.add_mesh(m, scalars="loreta_power", cmap="hot", clim=(0, 1),
                        smooth_shading=True, show_scalar_bar=True,
                        scalar_bar_args={"title": "LORETA Power", "color": "white"})
        {"axial": pl.view_xy, "coronal": pl.view_xz, "sagittal": pl.view_yz}[view]()
        pl.screenshot(str(path))
        pl.close()
        return True
    except Exception as exc:
        log(f"  screenshot skipped ({type(exc).__name__}: {exc}); off-screen rendering "
            "may need OSMesa/xvfb on a headless machine.")
        return False


def run_file(path, out_dir, assets, *, age=12, surface="outer", max_duration=60.0,
             start=0.0, lambda2=0.1, head_radius_mm=70.0, source_step_mm=7.0,
             export_surface=True, screenshot=False, view="axial", overwrite=False,
             log=print):
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{path.stem}_loreta"
    summary_path = Path(f"{prefix}_summary.json")
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"{summary_path} exists (use --overwrite)")

    raw = prep_raw(path, max_duration, start)
    log(f"  {len(raw.ch_names)} channels, {raw.info['sfreq']:g} Hz, {raw.times[-1]:.1f} s")

    used_age = assets.nearest_age(age)
    tpl_name, scalp_vertices, electrodes = assets.template(used_age)
    log(f"  atlas age {used_age} mo, scalp template {tpl_name}")
    scalp_center = (scalp_vertices.min(axis=0) + scalp_vertices.max(axis=0)) / 2.0
    ref_bounds = assets.reference_bounds(used_age)

    res = compute_loreta(raw, electrodes, scalp_center, ref_bounds,
                         head_radius_m=head_radius_mm / 1000.0,
                         source_step_m=source_step_mm / 1000.0,
                         lambda2=lambda2, log=log)

    written = []
    p = Path(f"{prefix}_sources.csv")
    _write_sources(p, res)
    written.append(p)
    p = Path(f"{prefix}_channel_corr.csv")
    _write_matrix(p, res["channel_corr"], res["channels"])
    written.append(p)

    surface_files = {}
    meshes = []
    if export_surface:
        for hemi, mesh in assets.surface_meshes(used_age, surface).items():
            m = mesh.copy()
            vals = map_to_surface(res["brain_pts"], res["power_norm"], m.points)
            m["loreta_power"] = vals
            p = Path(f"{prefix}_{hemi}.vtp")
            m.save(str(p))
            written.append(p)
            surface_files[hemi] = p.name
            meshes.append(m)
        if screenshot and meshes:
            p = Path(f"{prefix}_view.png")
            if _screenshot(meshes, p, view, log):
                written.append(p)

    top = np.argsort(res["power"])[::-1][:10]
    summary = {
        "input": str(path), "atlas_age_months": used_age, "requested_age_months": age,
        "scalp_template": tpl_name, "surface": surface, "method": res["method"],
        "lambda2": lambda2, "head_radius_mm": head_radius_mm,
        "source_step_mm": source_step_mm, "n_sources": res["n_sources"],
        "channels": res["channels"], "sfreq": res["sfreq"],
        "analysed_duration_s": res["duration_s"], "max_power": res["max_power"],
        "peak_source": {"head_mm": res["head_mm"][top[0]].tolist(),
                        "brain": res["brain_pts"][top[0]].tolist(),
                        "power": float(res["power"][top[0]])},
        "top_sources": [{"head_mm": res["head_mm"][i].tolist(),
                         "brain": res["brain_pts"][i].tolist(),
                         "power": float(res["power"][i])} for i in top],
        "surface_files": surface_files,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(summary_path)
    return written


# -------------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(
        description="Run the viewer's LORETA forward modelling on EDF recordings "
                    "and save the results to files.")
    ap.add_argument("files", nargs="+", help="EDF file(s)")
    ap.add_argument("--age", type=float, default=12.0,
                    help="infant age in months; picks the nearest atlas / template (default 12)")
    ap.add_argument("--root", default=str(ROOT), help="viewer folder containing atlas.json")
    ap.add_argument("--subjects-dir", default=str(DEFAULT_SUBJECTS_DIR),
                    help="where infant scalp templates are stored / downloaded")
    ap.add_argument("-o", "--output-dir", help="output directory (default: next to input)")
    ap.add_argument("--surface", choices=SURFACE_TYPES, default="outer",
                    help="brain surface for the .vtp export (default outer)")
    ap.add_argument("--no-surface", action="store_true",
                    help="skip the .vtp surface export (CSV/JSON only; no pyvista needed "
                         "for the export, atlas meshes are still read for alignment)")
    ap.add_argument("--screenshot", action="store_true", help="also save an off-screen PNG")
    ap.add_argument("--view", choices=("axial", "coronal", "sagittal"), default="axial")
    ap.add_argument("--max-duration", type=float, default=60.0,
                    help="seconds analysed, 0 = whole file (default 60, as in the viewer)")
    ap.add_argument("--start", type=float, default=0.0, help="start time in s (default 0)")
    ap.add_argument("--lambda2", type=float, default=0.1, help="dSPM regularisation (0.1)")
    ap.add_argument("--head-radius-mm", type=float, default=70.0)
    ap.add_argument("--source-step-mm", type=float, default=7.0,
                    help="source grid spacing; smaller = finer but slower (default 7)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    log = (lambda m: None) if args.quiet else print
    try:
        assets = ViewerAssets(args.root, args.subjects_dir, log)
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    failures = 0
    for f in args.files:
        src = Path(f)
        try:
            if not src.is_file():
                raise FileNotFoundError(f"No such file: {src}")
            log(f"{src.name}:")
            out_dir = Path(args.output_dir) if args.output_dir else src.parent
            files = run_file(
                src, out_dir, assets, age=args.age, surface=args.surface,
                max_duration=args.max_duration, start=args.start, lambda2=args.lambda2,
                head_radius_mm=args.head_radius_mm, source_step_mm=args.source_step_mm,
                export_surface=not args.no_surface, screenshot=args.screenshot,
                view=args.view, overwrite=args.overwrite, log=log)
            for w in files:
                log(f"  saved {w}")
        except Exception as exc:
            failures += 1
            print(f"{src}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            if not isinstance(exc, (ValueError, FileNotFoundError, FileExistsError)):
                traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
