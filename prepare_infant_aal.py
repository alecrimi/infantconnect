#!/usr/bin/env python3
"""Project the UNC Infant-AAL label maps onto the cortical surfaces in atlas.json.

Infant AAL (Shi et al., 2011, PLoS ONE 6(4):e18746) is distributed as VOLUMETRIC
label maps (90 ROIs) for neonates, 1-year-olds and 2-year-olds. This script
automatically downloads the three NIfTI volumes from the UNC/AFNI atlas server
into ./UNC_parcellation when they are not already present.

There is no label map for ages above 2 years, so the 2-year labels are reused
for the 36-72 month surfaces. Those labels are NOT age-matched.

Each atlas age in atlas.json gets one integer label per vertex of each hemisphere,
sampled on the MIDDLE surface (all surface types of one atlas age/hemisphere are
assumed to share vertex order; the viewer checks the vertex count before use).

Output (default ./parcellation):
    infant_aal_<age:02d>_<left|right>.npy   int16, one label per vertex (0 = none)
    infant_aal_meta.json                    source volume per age, QC numbers, ROI names

Usage:
    python prepare_infant_aal.py
    python prepare_infant_aal.py --out parcellation_test --max-fallback-mm 15
    python prepare_infant_aal.py --shift 0 0 0 --flip x   # override alignment defaults

CHECK THE PRINTED "direct hit" PERCENTAGES. If the surfaces are not in the same
coordinate frame as the label volumes, they will be low and the labels are
meaningless; use --shift / --flip, or register the surfaces to the label volumes.
"""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# Standard AAL90 abbreviations: odd index = left, even index = right.
# Verify against Table 2 of the UNC release document before publishing.
_AAL_BASE = [
    "PreCG", "SFGdor", "ORBsup", "MFG", "ORBmid", "IFGoperc", "IFGtriang",
    "ORBinf", "ROL", "SMA", "OLF", "SFGmed", "ORBsupmed", "REC", "INS", "ACG",
    "DCG", "PCG", "HIP", "PHG", "AMYG", "CAL", "CUN", "LING", "SOG", "MOG",
    "IOG", "FFG", "PoCG", "SPG", "IPL", "SMG", "ANG", "PCUN", "PCL", "CAU",
    "PUT", "PAL", "THA", "HES", "STG", "TPOsup", "MTG", "TPOmid", "ITG",
]
AAL_NAMES = {}
for _k, _n in enumerate(_AAL_BASE):
    AAL_NAMES[2 * _k + 1] = f"{_n}-L"
    AAL_NAMES[2 * _k + 2] = f"{_n}-R"

SUBCORTICAL = range(71, 79)  # caudate, putamen, pallidum, thalamus: not cortex

VOLUME_FILES = {"neo": "infant-neo-aal", "1yr": "infant-1yr-aal", "2yr": "infant-2yr-aal"}
AAL_DOWNLOAD_BASE = (
    "https://afni.nimh.nih.gov/pub/dist/atlases/UNC_Infant_Atlases/V1.0"
)


def ensure_aal_volumes(aal_dir):
    """Download the three UNC Infant-AAL NIfTI volumes if they are missing."""
    aal_dir = Path(aal_dir)
    aal_dir.mkdir(parents=True, exist_ok=True)

    for stem in VOLUME_FILES.values():
        destination = aal_dir / f"{stem}.nii.gz"
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"using existing {destination}")
            continue

        url = f"{AAL_DOWNLOAD_BASE}/{destination.name}"
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"downloading {url}")
        request = Request(url, headers={"User-Agent": "prepare-infant-aal/1.0"})
        try:
            with urlopen(request, timeout=60) as response, temporary.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"Downloaded an empty file: {url}")
            temporary.replace(destination)
            print(f"saved {destination}")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def volume_for_age(age_months):
    """Neonate / 1-year / 2-year label map used for a given atlas age."""
    if age_months <= 6:
        return "neo"
    if age_months <= 18:
        return "1yr"
    return "2yr"


def allowed_labels(side):
    idx = np.arange(0, 91)
    ok = idx >= 1
    ok &= ~np.isin(idx, list(SUBCORTICAL))
    ok &= (idx % 2 == 1) if side == "left" else (idx % 2 == 0)
    return ok


def find_volume(aal_dir, stem):
    aal_dir = Path(aal_dir)
    for suffix in (".nii.gz", ".nii", ".hdr"):
        p = aal_dir / f"{stem}{suffix}"
        if p.is_file():
            return p
    hits = sorted(aal_dir.rglob(f"{stem}.hdr")) + sorted(aal_dir.rglob(f"{stem}.nii*"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No {stem}.hdr/.nii[.gz] under {aal_dir}")


def load_label_volume(path, origin=(90, 126, 72), flips=""):
    """Return (labels int16 array, voxel->mm affine).

    NIfTI files use their own affine. For Analyze .hdr/.img the affine is built from
    voxel sizes and the SPM-style 1-based origin (the 2013 UNC release sets the
    origin on the anterior commissure at [90 126 72]). Analyze orientation is
    ambiguous, hence --flip.
    """
    img = nib.load(str(path))
    data = np.rint(np.asarray(img.dataobj)).astype(np.int16)
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3D label volume, got {data.shape}")
    if str(path).endswith((".nii", ".nii.gz")):
        affine = np.array(img.affine, dtype=float)
    else:
        zooms = np.array(img.header.get_zooms()[:3], dtype=float)
        affine = np.diag([*zooms, 1.0])
        affine[:3, 3] = -(np.asarray(origin, dtype=float) - 1.0) * zooms
    sign = np.ones(4)
    for ax in flips.lower():
        sign["xyz".index(ax)] = -1.0
    return data, np.diag(sign) @ affine


def sample_labels(points_mm, volume, affine, side, max_fallback_mm=8.0):
    """Label per vertex. Vertices not landing on an admissible label (cortical,
    right hemisphere parity) take the nearest admissible voxel within max_fallback_mm,
    else 0. Returns (labels int16, stats dict)."""
    pts = np.asarray(points_mm, dtype=float)
    allowed = allowed_labels(side)
    inv = np.linalg.inv(affine)
    ijk = np.rint(nib.affines.apply_affine(inv, pts)).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(volume.shape)), axis=1)

    lab = np.zeros(len(pts), dtype=np.int16)
    lab[inside] = volume[tuple(ijk[inside].T)]
    valid = inside & (lab >= 0) & (lab <= 90)
    ok = valid & allowed[np.clip(lab, 0, 90)]
    stats = {"direct_hit": float(ok.mean()), "fallback_mean_mm": 0.0, "unlabeled": 0}

    bad = np.flatnonzero(~ok)
    if bad.size:
        vol_ok = (volume >= 0) & (volume <= 90) & allowed[np.clip(volume, 0, 90)]
        cand = np.argwhere(vol_ok)
        if cand.size == 0:
            lab[bad] = 0
        else:
            tree = cKDTree(nib.affines.apply_affine(affine, cand))
            d, j = tree.query(pts[bad])
            lab[bad] = volume[tuple(cand[j].T)]
            far = bad[d > max_fallback_mm]
            lab[far] = 0
            stats["fallback_mean_mm"] = float(d.mean())
    stats["unlabeled"] = int((lab == 0).sum())
    return lab, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aal-dir", default="UNC_parcellation",
                    help="folder for downloaded infant-{neo,1yr,2yr}-aal files (default: UNC_parcellation)")
    ap.add_argument("--root", default=".", help="repo root that atlas.json paths are relative to")
    ap.add_argument("--atlas-json", default="atlas.json")
    ap.add_argument("--out", default="parcellation")
    ap.add_argument("--surface", default="middle", help="surface type to sample (default middle)")
    ap.add_argument("--origin", type=float, nargs=3, default=(90, 126, 72),
                    help="1-based voxel origin for Analyze volumes (default: AC at 90 126 72)")
    ap.add_argument("--shift", type=float, nargs=3, default=(-128, -128, -128),
                    help="mm added to surface coordinates before sampling (default: -128 -128 -128)")
    ap.add_argument("--flip", default="", help="axes of the label volume to mirror, e.g. 'x' or 'xy'")
    ap.add_argument("--max-fallback-mm", type=float, default=8.0)
    args = ap.parse_args()

    root = Path(args.root)
    aal_dir = Path(args.aal_dir)
    if not aal_dir.is_absolute():
        aal_dir = root / aal_dir

    # Fetch the UNC volumes automatically before attempting the projection.
    ensure_aal_volumes(aal_dir)

    atlas = json.loads((root / args.atlas_json).read_text(encoding="utf-8"))
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    volumes = {}
    meta = {"sources": {}, "qc": {}, "aal_names": {str(k): v for k, v in AAL_NAMES.items()},
            "note": "Infant AAL labels are defined for 0, 1 and 2 years only; "
                    "ages above 24 months reuse the 2-year labels (not age-matched)."}

    for age in sorted(int(a) for a in atlas):
        src = volume_for_age(age)
        if src not in volumes:
            path = find_volume(aal_dir, VOLUME_FILES[src])
            volumes[src] = load_label_volume(path, args.origin, args.flip)
            print(f"loaded {path.name}: shape {volumes[src][0].shape}")
        vol, aff = volumes[src]
        meta["sources"][str(age)] = src
        for hemi in ("left", "right"):
            rel = atlas[str(age)].get(hemi, {}).get(args.surface)
            if not rel:
                continue
            mesh = pv.read(str(root / rel))
            pts = np.asarray(mesh.points, dtype=float) + np.asarray(args.shift)
            lab, st = sample_labels(pts, vol, aff, hemi, args.max_fallback_mm)
            np.save(out / f"infant_aal_{age:02d}_{hemi}.npy", lab)
            meta["qc"][f"{age}_{hemi}"] = {"n_vertices": int(len(lab)), **st,
                                          "n_labels": int(len(np.unique(lab[lab > 0])))}
            flag = "  <-- LOW: check alignment" if st["direct_hit"] < 0.6 else ""
            print(f"age {age:>2} {hemi:<5} {src}  direct hit {st['direct_hit']*100:5.1f}%  "
                  f"fallback {st['fallback_mean_mm']:.2f} mm  unlabeled {st['unlabeled']}{flag}")

    (out / "infant_aal_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote labels and infant_aal_meta.json to {out}")


if __name__ == "__main__":
    main()
