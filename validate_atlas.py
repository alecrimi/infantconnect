#!/usr/bin/env python3
"""
Audit an infant cortical-surface atlas and produce reproducible QC reports.

Expected layout:
    project/
      validate_atlas.py
      atlas.json
      01/SparseAtlas_01_lh.OuterSurf.vtk
      01/SparseAtlas_01_rh.OuterSurf.vtk
      ...

The manifest may use age keys such as "01" or "1". Hemisphere keys are
"left"/"right"; surface keys are "outer", "middle", "inner", "inflated",
and optionally "sphere". Paths in atlas.json should be relative to this
script/project directory.

Outputs:
    qc/atlas_qc.csv
    qc/atlas_qc.json
    qc/atlas_qc_summary.png  (if matplotlib is installed)

Install:
    pip install pyvista numpy
Optional plot:
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

DEFAULT_SURFACES = ("outer", "middle", "inner", "inflated", "sphere")
HEMISPHERES = ("left", "right")
HEMI_FILE_CODES = {"left": "lh", "right": "rh"}
SURFACE_FILE_CODES = {
    "outer": "OuterSurf",
    "middle": "MiddleSurf",
    "inner": "InnerSurf",
    "inflated": "InflateSurf",
    "sphere": "SphereSurf",
}


def json_safe(value: Any) -> Any:
    """Convert numpy/scalar values to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def normalize_age_key(age_key: Any) -> str:
    """Normalize an age key to an integer-month string, e.g. '01' -> '1'."""
    try:
        return str(int(str(age_key)))
    except (TypeError, ValueError):
        return str(age_key)


def find_manifest_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else (root / path).resolve()


def count_boundary_edges(mesh: pv.PolyData) -> int | None:
    """Count boundary edges; nonzero often indicates an open surface."""
    try:
        edges = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        return int(edges.n_cells)
    except Exception:
        return None


def mesh_qc(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "file_exists": path.is_file(),
        "points": None,
        "cells": None,
        "triangles": None,
        "surface_area": None,
        "volume": None,
        "bounds_xmin": None,
        "bounds_xmax": None,
        "bounds_ymin": None,
        "bounds_ymax": None,
        "bounds_zmin": None,
        "bounds_zmax": None,
        "boundary_edges": None,
        "connected_components": None,
        "finite_coordinates": None,
        "status": "ERROR",
        "message": "",
    }
    if not path.is_file():
        row["message"] = "File not found"
        return row

    try:
        mesh = pv.read(str(path))
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()
        row["points"] = int(mesh.n_points)
        row["cells"] = int(mesh.n_cells)

        if mesh.n_points == 0 or mesh.n_cells == 0:
            row["message"] = "Empty mesh (zero points or cells)"
            return row

        points = np.asarray(mesh.points)
        finite = bool(np.isfinite(points).all())
        row["finite_coordinates"] = finite
        if not finite:
            row["message"] = "Non-finite vertex coordinates"
            return row

        bounds = mesh.bounds
        for key, value in zip(
            ("bounds_xmin", "bounds_xmax", "bounds_ymin",
             "bounds_ymax", "bounds_zmin", "bounds_zmax"),
            bounds,
        ):
            row[key] = float(value)

        # Triangulate before area and triangle-count checks.
        tri = mesh.triangulate()
        row["triangles"] = int(tri.n_cells)
        sizes = tri.compute_cell_sizes(
            length=False, area=True, volume=False
        )
        areas = np.asarray(sizes.cell_data.get("Area", []), dtype=float)
        row["surface_area"] = float(np.nansum(areas)) if areas.size else None

        row["boundary_edges"] = count_boundary_edges(mesh)

        try:
            connected = mesh.connectivity()
            labels = np.asarray(connected.cell_data.get("RegionId", []))
            row["connected_components"] = (
                int(np.unique(labels).size) if labels.size else None
            )
        except Exception:
            row["connected_components"] = None

        # Volume is informative only for closed, consistently oriented meshes.
        try:
            row["volume"] = float(mesh.volume)
        except Exception:
            row["volume"] = None

        messages = []
        if row["triangles"] == 0:
            messages.append("No polygonal faces after triangulation")
        if row["boundary_edges"] not in (None, 0):
            messages.append("Open boundary edges detected")
        if row["connected_components"] not in (None, 1):
            messages.append("Multiple connected components detected")
        if row["surface_area"] is None or row["surface_area"] <= 0:
            messages.append("Surface area is zero or unavailable")

        row["status"] = "WARN" if messages else "PASS"
        row["message"] = "; ".join(messages) if messages else "Basic checks passed"
        return row

    except Exception as exc:
        row["message"] = f"{type(exc).__name__}: {exc}"
        return row


def discover_surface_files(root: Path) -> dict[tuple[str, str, str], str]:
    """Discover VTK files based on the known filename convention."""
    found: dict[tuple[str, str, str], str] = {}
    for age_dir in root.iterdir():
        if not age_dir.is_dir() or not age_dir.name.isdigit():
            continue
        age = normalize_age_key(age_dir.name)
        for path in age_dir.glob("*.vtk"):
            name = path.name
            for hemi, hemi_code in HEMI_FILE_CODES.items():
                for surface, surface_code in SURFACE_FILE_CODES.items():
                    if name.endswith(f"_{hemi_code}.{surface_code}.vtk"):
                        found[(age, hemi, surface)] = str(path.relative_to(root))
    return found


def load_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError("atlas.json must contain a JSON object keyed by age.")
    return manifest


def build_manifest_rows(
    root: Path,
    manifest: dict[str, Any],
    discovered: dict[tuple[str, str, str], str],
    surfaces: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = set()

    for age_key, age_entry in manifest.items():
        age = normalize_age_key(age_key)
        if not isinstance(age_entry, dict):
            continue
        for hemi in HEMISPHERES:
            hemi_entry = age_entry.get(hemi, {})
            if not isinstance(hemi_entry, dict):
                hemi_entry = {}
            for surface in surfaces:
                rel = hemi_entry.get(surface)
                discovered_rel = discovered.get((age, hemi, surface))
                # Use the manifest path if present; otherwise report discovered
                # file separately and flag the missing manifest entry.
                rel_path = str(rel) if rel else (discovered_rel or "")
                path = find_manifest_path(root, rel_path) if rel_path else root / "__missing__"
                qc = mesh_qc(path)
                if not rel:
                    qc["status"] = "WARN" if qc["file_exists"] else "ERROR"
                    prefix = "Missing manifest entry"
                    if discovered_rel:
                        qc["message"] = f"{prefix}; matching file discovered: {discovered_rel}"
                    else:
                        qc["message"] = prefix
                row = {
                    "age_months": int(age) if age.isdigit() else age,
                    "age_key_in_manifest": str(age_key),
                    "hemisphere": hemi,
                    "surface": surface,
                    "manifest_path": str(rel) if rel else "",
                    "discovered_path": discovered_rel or "",
                    "resolved_path": str(path) if rel_path else "",
                    **qc,
                }
                rows.append(json_safe(row))
                seen.add((age, hemi, surface))

    # Include files found on disk but not represented by a manifest age.
    manifest_ages = {normalize_age_key(k) for k in manifest.keys()}
    for (age, hemi, surface), rel in sorted(discovered.items()):
        if age not in manifest_ages:
            path = find_manifest_path(root, rel)
            qc = mesh_qc(path)
            rows.append(json_safe({
                "age_months": int(age),
                "age_key_in_manifest": "",
                "hemisphere": hemi,
                "surface": surface,
                "manifest_path": "",
                "discovered_path": rel,
                "resolved_path": str(path),
                **qc,
                "status": "WARN" if qc["status"] == "PASS" else qc["status"],
                "message": "File exists on disk but age is absent from atlas.json; " + qc["message"],
            }))
    return rows


def write_summary_plot(rows: list[dict[str, Any]], output_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ages = sorted({
        int(r["age_months"])
        for r in rows
        if isinstance(r.get("age_months"), (int, float))
    })
    surfaces = list(DEFAULT_SURFACES)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

    # Plot point counts per age and surface, aggregating hemispheres by mean.
    for surface in surfaces:
        xs, ys = [], []
        for age in ages:
            vals = [
                r["points"] for r in rows
                if r.get("surface") == surface
                and r.get("age_months") == age
                and isinstance(r.get("points"), (int, float))
            ]
            if vals:
                xs.append(age)
                ys.append(float(np.mean(vals)))
        if xs:
            axes[0].plot(xs, ys, marker="o", label=surface)
    axes[0].set_title("Mean vertices per hemisphere by age and surface")
    axes[0].set_xlabel("Age (months)")
    axes[0].set_ylabel("Number of vertices")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    statuses = Counter(r.get("status", "ERROR") for r in rows)
    axes[1].bar(list(statuses.keys()), list(statuses.values()))
    axes[1].set_title("QC status counts (PASS is not a full anatomical validation)")
    axes[1].set_ylabel("Number of manifest entries")
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project directory containing atlas.json and age folders.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest path (default: ROOT/atlas.json).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: ROOT/qc).")
    parser.add_argument("--include-sphere", action="store_true",
                        help="Include sphere surfaces in the audit (default: yes if in manifest).")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = (args.manifest or (root / "atlas.json")).resolve()
    out_dir = (args.out or (root / "qc")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        manifest = load_manifest(root, manifest_path)
    except Exception as exc:
        print(f"ERROR: cannot read manifest: {exc}", file=sys.stderr)
        return 2

    discovered = discover_surface_files(root / "data")
    surfaces = DEFAULT_SURFACES
    rows = build_manifest_rows(root, manifest, discovered, surfaces)

    csv_path = out_dir / "atlas_qc.csv"
    json_path = out_dir / "atlas_qc.json"
    png_path = out_dir / "atlas_qc_summary.png"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(r.get("status", "ERROR") for r in rows)
    summary = {
        "root": str(root),
        "manifest": str(manifest_path),
        "entries_audited": len(rows),
        "status_counts": dict(counts),
        "discovered_files": len(discovered),
        "manifest_age_keys": list(manifest.keys()),
        "normalized_manifest_ages_months": sorted({
            normalize_age_key(k) for k in manifest.keys()
        }),
        "limitations": [
            "These are basic geometric/file-integrity checks, not anatomical validation.",
            "A closed surface or nonzero volume is not guaranteed to be anatomically correct.",
            "Coordinates and units are reported as stored; their meaning must be established from atlas documentation.",
            "Vertex counts across ages do not establish vertex-wise correspondence.",
            "Sphere meshes are included in the audit but should not be treated as anatomical surfaces.",
        ],
        "entries": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)

    plot_written = write_summary_plot(rows, png_path)

    print(f"Audited {len(rows)} manifest entries; discovered {len(discovered)} VTK files.")
    print("Status counts:", dict(counts))
    print(f"CSV report:  {csv_path}")
    print(f"JSON report: {json_path}")
    if plot_written:
        print(f"QC plot:     {png_path}")
    else:
        print("QC plot skipped (install matplotlib to enable it).")
    print("\nInterpretation:")
    print("- ERROR: missing file, unreadable mesh, empty mesh, or invalid coordinates.")
    print("- WARN: mesh loaded but a basic structural check flagged a potential issue.")
    print("- PASS: basic checks passed; this is not a certification of anatomical validity.")
    return 0 if counts.get("ERROR", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
