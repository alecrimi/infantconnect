#!/usr/bin/env python3
"""InfantConnect asset, manifest, and geometric quality-control audit.

The script deliberately keeps three populations separate:

1. Physical assets: recognised resource files found recursively below
   --asset-root. This can include meshes, CSV/TSV tables, JSON metadata, and
   common neuroimaging/image files.
2. Manifest entries: the expected age x hemisphere x surface combinations in
   atlas.json (11 x 2 x 5 = 110 when the manifest has 11 ages).
3. Geometry: extent statistics for (a) every successfully read mesh and
   (b) cortical surfaces only (OuterSurf, MiddleSurf, InnerSurf, InflateSurf;
   SphereSurf is excluded from the cortical subset).

IMPORTANT:
- To reproduce a historical total such as 889, --asset-root must point to the
  same common resource directory used by the historical inventory. The script
  reports a count by extension so the asset population can be checked.
- The default asset extension allow-list is intentionally explicit. Extend it
  with --extra-extensions or replace it with --asset-extensions if the original
  inventory used other file types.
- Geometric extent ratios use the symmetric definition:
      max(extent / median_extent, median_extent / extent)
  so both unusually large and unusually small extents are detected.

Example:
    python validate_infantconnect_atlas.py \
      --root /home/alex/Documents/infant_viewer/src/infant_brain_viewer \
      --asset-root /home/alex/Documents/infant_viewer

Dependencies:
    pip install numpy pyvista
Optional plot:
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HEMISPHERES = ("left", "right")
HEMI_FILE_CODES = {"left": "lh", "right": "rh"}

SURFACE_FILE_CODES = {
    "outer": "OuterSurf",
    "middle": "MiddleSurf",
    "inner": "InnerSurf",
    "inflated": "InflateSurf",
    "sphere": "SphereSurf",
}

# Keep all five manifest surfaces so that 11 ages x 2 hemispheres x 5 surfaces
# produces 110 expected entries.
MANIFEST_SURFACES = tuple(SURFACE_FILE_CODES.keys())

# PyVista-readable mesh formats. The cortical atlas files are usually VTK.
MESH_EXTENSIONS = {
    ".vtk", ".vtp", ".vtu", ".vts", ".vtr", ".vtm",
    ".stl", ".ply", ".obj", ".off",
}

# Physical resource files included in the general asset inventory by default.
# Add formats from the original inventory with --extra-extensions, for example
# --extra-extensions .mat .fib .trk .tck .mif.
DEFAULT_ASSET_EXTENSIONS = {
    *MESH_EXTENSIONS,
    ".csv", ".tsv", ".json", ".xml", ".txt",
    ".nii", ".nii.gz", ".gii", ".npy", ".npz", ".mat",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    ".fib", ".trk", ".tck", ".mif",
}

# Directories that are not part of a resource inventory.
EXCLUDED_DIR_NAMES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "build", "dist", "qc", ".idea", ".vscode",
}

SURFACE_FILENAME_RE = re.compile(
    r"^SparseAtlas_(?P<age>\d+)_(?P<hemi>lh|rh)\."
    r"(?P<surface>OuterSurf|MiddleSurf|InnerSurf|InflateSurf|SphereSurf)\.vtk$",
    re.IGNORECASE,
)

GEOMETRIC_CORTICAL_SUFFIXES = (
    ".OuterSurf.vtk",
    ".MiddleSurf.vtk",
    ".InnerSurf.vtk",
    ".InflateSurf.vtk",
)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    """Recursively convert NumPy and Path values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
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
    """Normalise '01' and '1' to the same age key ('1')."""
    try:
        return str(int(str(age_key)))
    except (TypeError, ValueError):
        return str(age_key)


def file_extension(path: Path) -> str:
    """Return a lower-case extension, handling common compound suffixes."""
    name = path.name.lower()
    for suffix in (".nii.gz", ".tar.gz"):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def is_relative_to(path: Path, parent: Path) -> bool:
    """Compatibility helper for Path.is_relative_to()."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_manifest_path(
    root: Path,
    asset_root: Path,
    raw_path: str,
) -> Path:
    """Resolve a manifest path against project root, then asset root."""
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    candidates = [
        (root / candidate).resolve(),
        (asset_root / candidate).resolve(),
    ]
    for resolved in candidates:
        if resolved.is_file():
            return resolved
    # Return the project-root candidate so a missing reference is explicit.
    return candidates[0]


def extension_counts(paths: Iterable[Path]) -> dict[str, int]:
    counts = Counter(file_extension(p) or "[no extension]" for p in paths)
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Mesh QC
# ---------------------------------------------------------------------------

def count_boundary_edges(mesh: pv.PolyData) -> int | None:
    """Count open boundary edges; nonzero is a warning, not always an error."""
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


def _as_polydata(mesh: Any) -> pv.PolyData:
    """Convert a PyVista dataset or multiblock to PolyData for surface QC."""
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if isinstance(mesh, pv.PolyData):
        return mesh
    if hasattr(mesh, "extract_surface"):
        return mesh.extract_surface()
    raise TypeError(f"Unsupported PyVista object: {type(mesh).__name__}")


def mesh_qc(path: Path) -> dict[str, Any]:
    """Load a mesh and collect basic integrity and geometry measures."""
    row: dict[str, Any] = {
        "file_exists": path.is_file(),
        "file_size_bytes": path.stat().st_size if path.is_file() else 0,
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
        "extent_x": None,
        "extent_y": None,
        "extent_z": None,
        "boundary_edges": None,
        "connected_components": None,
        "finite_coordinates": None,
        "status": "ERROR",
        "message": "",
    }

    if not path.is_file():
        row["message"] = "File not found"
        return row
    if row["file_size_bytes"] == 0:
        row["message"] = "File is empty"
        return row

    try:
        loaded = pv.read(str(path))
        mesh = _as_polydata(loaded)

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
        bound_keys = (
            "bounds_xmin", "bounds_xmax",
            "bounds_ymin", "bounds_ymax",
            "bounds_zmin", "bounds_zmax",
        )
        for key, value in zip(bound_keys, bounds):
            row[key] = float(value)

        row["extent_x"] = float(bounds[1] - bounds[0])
        row["extent_y"] = float(bounds[3] - bounds[2])
        row["extent_z"] = float(bounds[5] - bounds[4])

        tri = mesh.triangulate()
        row["triangles"] = int(tri.n_cells)
        sizes = tri.compute_cell_sizes(length=False, area=True, volume=False)
        areas = np.asarray(sizes.cell_data.get("Area", []), dtype=float)
        if areas.size and np.isfinite(areas).any():
            row["surface_area"] = float(np.nansum(areas))

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
            volume = float(mesh.volume)
            row["volume"] = volume if math.isfinite(volume) else None
        except Exception:
            row["volume"] = None

        warnings: list[str] = []
        if row["triangles"] == 0:
            warnings.append("No polygonal faces after triangulation")
        if row["boundary_edges"] not in (None, 0):
            warnings.append("Open boundary edges detected")
        if row["connected_components"] not in (None, 1):
            warnings.append("Multiple connected components detected")
        if row["surface_area"] is None or row["surface_area"] <= 0:
            warnings.append("Surface area is zero or unavailable")
        if min(row["extent_x"], row["extent_y"], row["extent_z"]) <= 0:
            warnings.append("One or more spatial extents are zero")

        row["status"] = "WARN" if warnings else "PASS"
        row["message"] = "; ".join(warnings) if warnings else "Basic mesh checks passed"
        return row

    except Exception as exc:
        row["message"] = f"{type(exc).__name__}: {exc}"
        return row


# ---------------------------------------------------------------------------
# Non-mesh asset QC and inventory
# ---------------------------------------------------------------------------

def tabular_file_qc(path: Path) -> dict[str, Any]:
    """Check that a CSV/TSV can be read and contains at least one record."""
    row: dict[str, Any] = {
        "file_exists": path.is_file(),
        "file_size_bytes": path.stat().st_size if path.is_file() else 0,
        "table_rows": None,
        "table_columns": None,
        "status": "ERROR",
        "message": "",
    }
    if not path.is_file():
        row["message"] = "File not found"
        return row
    if row["file_size_bytes"] == 0:
        row["message"] = "File is empty"
        return row

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            first_row = next(reader, None)
            if first_row is None:
                row["message"] = "Table contains no rows"
                return row
            row_count = 1
            row_count += sum(1 for _ in reader)
        row["table_rows"] = int(row_count)
        row["table_columns"] = int(len(first_row))
        if len(first_row) == 0:
            row["status"] = "WARN"
            row["message"] = "First row is empty"
        else:
            row["status"] = "PASS"
            row["message"] = "Table is readable"
        return row
    except Exception as exc:
        row["message"] = f"{type(exc).__name__}: {exc}"
        return row


def json_file_qc(path: Path) -> dict[str, Any]:
    """Validate JSON syntax for JSON resources."""
    row: dict[str, Any] = {
        "file_exists": path.is_file(),
        "file_size_bytes": path.stat().st_size if path.is_file() else 0,
        "status": "ERROR",
        "message": "",
    }
    if not path.is_file():
        row["message"] = "File not found"
        return row
    if row["file_size_bytes"] == 0:
        row["message"] = "File is empty"
        return row
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        row["status"] = "PASS"
        row["message"] = "Valid JSON"
    except Exception as exc:
        row["message"] = f"{type(exc).__name__}: {exc}"
    return row


def generic_file_qc(path: Path) -> dict[str, Any]:
    """Basic existence and non-empty check for other resource formats."""
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    if not exists:
        status, message = "ERROR", "File not found"
    elif size == 0:
        status, message = "ERROR", "File is empty"
    else:
        status, message = "PASS", "File exists and is non-empty; format not parsed"
    return {
        "file_exists": exists,
        "file_size_bytes": size,
        "status": status,
        "message": message,
    }


def audit_asset_file(path: Path, asset_root: Path) -> dict[str, Any]:
    """Audit one physical asset; detailed parsing depends on its file type."""
    ext = file_extension(path)
    base: dict[str, Any] = {
        "asset_path": str(path.relative_to(asset_root)),
        "asset_type": "other",
        "file_extension": ext or "[no extension]",
        "file_exists": path.is_file(),
        "file_size_bytes": path.stat().st_size if path.is_file() else 0,
        "status": "ERROR",
        "message": "",
    }

    if ext in MESH_EXTENSIONS:
        base["asset_type"] = "mesh"
        details = mesh_qc(path)
    elif ext in {".csv", ".tsv"}:
        base["asset_type"] = "table"
        details = tabular_file_qc(path)
    elif ext == ".json":
        base["asset_type"] = "json"
        details = json_file_qc(path)
    else:
        details = generic_file_qc(path)

    return {**base, **details}


def discover_asset_files(
    asset_root: Path,
    allowed_extensions: set[str],
    all_files: bool,
    excluded_paths: list[Path],
) -> list[Path]:
    """Recursively discover resource files below asset_root."""
    discovered: list[Path] = []
    for path in asset_root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(asset_root).parts
        if any(part in EXCLUDED_DIR_NAMES for part in rel_parts[:-1]):
            continue
        if any(part.startswith(".") for part in rel_parts[:-1]):
            continue
        resolved = path.resolve()
        if any(resolved == excluded or is_relative_to(resolved, excluded) for excluded in excluded_paths):
            continue
        if all_files or file_extension(path) in allowed_extensions:
            discovered.append(path)
    return sorted(discovered)


# ---------------------------------------------------------------------------
# Atlas surface discovery and manifest audit
# ---------------------------------------------------------------------------

def discover_surface_files(asset_root: Path) -> dict[tuple[str, str, str], list[Path]]:
    """Find atlas surface meshes recursively, retaining duplicate matches."""
    found: dict[tuple[str, str, str], list[Path]] = {}
    for path in asset_root.rglob("*.vtk"):
        rel_parts = path.relative_to(asset_root).parts
        if any(part in EXCLUDED_DIR_NAMES for part in rel_parts[:-1]):
            continue
        match = SURFACE_FILENAME_RE.match(path.name)
        if not match:
            continue

        age = normalize_age_key(match.group("age"))
        hemi = "left" if match.group("hemi").lower() == "lh" else "right"
        surface_code = match.group("surface").lower()
        surface = {
            "outersurf": "outer",
            "middlesurf": "middle",
            "innersurf": "inner",
            "inflatesurf": "inflated",
            "spheresurf": "sphere",
        }[surface_code]
        found.setdefault((age, hemi, surface), []).append(path.resolve())

    for key in found:
        found[key] = sorted(found[key])
    return found


def build_manifest_rows(
    root: Path,
    asset_root: Path,
    manifest: dict[str, Any],
    discovered: dict[tuple[str, str, str], list[Path]],
) -> list[dict[str, Any]]:
    """Audit every expected age/hemisphere/surface manifest slot."""
    rows: list[dict[str, Any]] = []
    manifest_keys: set[tuple[str, str, str]] = set()

    for age_key, age_entry in manifest.items():
        age = normalize_age_key(age_key)
        if not isinstance(age_entry, dict):
            rows.append({
                "age_months": int(age) if age.isdigit() else age,
                "age_key_in_manifest": str(age_key),
                "hemisphere": "",
                "surface": "",
                "manifest_path": "",
                "discovered_path": "",
                "resolved_path": "",
                "file_exists": False,
                "status": "ERROR",
                "message": "Age entry is not a JSON object",
            })
            continue

        for hemi in HEMISPHERES:
            hemi_entry = age_entry.get(hemi, {})
            if not isinstance(hemi_entry, dict):
                hemi_entry = {}

            for surface in MANIFEST_SURFACES:
                key = (age, hemi, surface)
                manifest_keys.add(key)
                raw_value = hemi_entry.get(surface)
                raw_path = str(raw_value).strip() if raw_value else ""
                matches = discovered.get(key, [])
                discovered_rel = str(matches[0].relative_to(asset_root)) if matches else ""

                if raw_path:
                    resolved = resolve_manifest_path(root, asset_root, raw_path)
                    if resolved.is_file():
                        qc = mesh_qc(resolved)
                        message = qc["message"]
                        status = qc["status"]
                        # A declared file can exist yet differ from the filename
                        # convention; preserve the successful path but report it.
                        if matches and resolved.resolve() not in matches:
                            status = "WARN" if status == "PASS" else status
                            message = (message + "; " if message else "") + (
                                "Manifest path differs from discovered filename-convention path"
                            )
                    elif len(matches) == 1:
                        resolved = matches[0]
                        qc = mesh_qc(resolved)
                        status = "WARN" if qc["status"] in {"PASS", "WARN"} else "ERROR"
                        message = (
                            f"Manifest path does not exist; matching file found at "
                            f"{discovered_rel}. " + qc["message"]
                        )
                    elif len(matches) > 1:
                        resolved = matches[0]
                        qc = mesh_qc(resolved)
                        status = "ERROR"
                        message = (
                            "Manifest path does not exist and multiple matching files "
                            f"were discovered ({len(matches)})."
                        )
                    else:
                        qc = mesh_qc(resolved)
                        status = "ERROR"
                        message = f"Manifest path does not exist: {raw_path}"
                else:
                    if len(matches) == 1:
                        resolved = matches[0]
                        qc = mesh_qc(resolved)
                        status = "WARN" if qc["status"] in {"PASS", "WARN"} else "ERROR"
                        message = "Missing manifest entry; matching physical file discovered. " + qc["message"]
                    elif len(matches) > 1:
                        resolved = matches[0]
                        qc = mesh_qc(resolved)
                        status = "ERROR"
                        message = f"Missing manifest entry; {len(matches)} matching physical files discovered."
                    else:
                        resolved = None
                        qc = {
                            "file_exists": False,
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
                            "extent_x": None,
                            "extent_y": None,
                            "extent_z": None,
                            "boundary_edges": None,
                            "connected_components": None,
                            "finite_coordinates": None,
                        }
                        status = "ERROR"
                        message = "Missing manifest entry and no matching physical file was discovered"

                row = {
                    "age_months": int(age) if age.isdigit() else age,
                    "age_key_in_manifest": str(age_key),
                    "hemisphere": hemi,
                    "surface": surface,
                    "manifest_path": raw_path,
                    "discovered_path": discovered_rel,
                    "resolved_path": str(resolved) if resolved else "",
                    **qc,
                    "status": status,
                    "message": message,
                }
                rows.append(json_safe(row))

    # Report discovered surface meshes whose age is not represented in atlas.json.
    manifest_ages = {normalize_age_key(k) for k in manifest.keys()}
    for (age, hemi, surface), matches in sorted(discovered.items()):
        if age in manifest_ages:
            continue
        for path in matches:
            qc = mesh_qc(path)
            rows.append(json_safe({
                "age_months": int(age) if age.isdigit() else age,
                "age_key_in_manifest": "",
                "hemisphere": hemi,
                "surface": surface,
                "manifest_path": "",
                "discovered_path": str(path.relative_to(asset_root)),
                "resolved_path": str(path),
                **qc,
                "status": "WARN" if qc["status"] in {"PASS", "WARN"} else "ERROR",
                "message": "Physical surface file belongs to an age absent from atlas.json. " + qc["message"],
            }))

    # Report duplicate filename-convention matches even if the manifest points
    # to one of them (otherwise dictionary overwrites can hide duplicates).
    for key, matches in discovered.items():
        if len(matches) <= 1:
            continue
        age, hemi, surface = key
        rows.append({
            "age_months": int(age) if age.isdigit() else age,
            "age_key_in_manifest": "",
            "hemisphere": hemi,
            "surface": surface,
            "manifest_path": "",
            "discovered_path": "; ".join(str(p.relative_to(asset_root)) for p in matches),
            "resolved_path": "",
            "file_exists": True,
            "status": "ERROR",
            "message": f"Duplicate physical files discovered for this age/hemisphere/surface: {len(matches)} files",
        })

    return rows


# ---------------------------------------------------------------------------
# Geometric summaries
# ---------------------------------------------------------------------------

def geometric_summary(
    mesh_rows: list[dict[str, Any]],
    cortical_only: bool,
) -> dict[str, Any]:
    """Compute medians and symmetric max extent ratio for a mesh population."""
    candidates = mesh_rows
    if cortical_only:
        candidates = [
            row for row in mesh_rows
            if any(str(row.get("asset_path", "")).endswith(suffix)
                   for suffix in GEOMETRIC_CORTICAL_SUFFIXES)
        ]

    valid = [
        row for row in candidates
        if row.get("status") in {"PASS", "WARN"}
        and all(row.get(k) is not None for k in ("extent_x", "extent_y", "extent_z"))
        and all(float(row[k]) > 0 for k in ("extent_x", "extent_y", "extent_z"))
    ]

    result: dict[str, Any] = {
        "population": "cortical_surfaces_excluding_sphere" if cortical_only else "all_discovered_meshes",
        "mesh_files_in_population": len(candidates),
        "meshes_with_valid_geometry": len(valid),
        "meshes_excluded_from_geometry": len(candidates) - len(valid),
        "median_extent_x": None,
        "median_extent_y": None,
        "median_extent_z": None,
        "maximum_symmetric_extent_ratio": None,
        "maximum_ratio_asset": None,
        "maximum_ratio_axis": None,
    }
    if not valid:
        return result

    extents = np.asarray([
        [float(row["extent_x"]), float(row["extent_y"]), float(row["extent_z"])]
        for row in valid
    ], dtype=float)
    medians = np.median(extents, axis=0)
    if np.any(medians <= 0):
        raise ValueError("Median extent is non-positive; cannot compute extent ratios")

    # Symmetric ratio: flags both larger-than-median and smaller-than-median
    # extents. This is not the one-sided extents / median calculation.
    ratio_matrix = np.maximum(extents / medians, medians / extents)
    flat_index = int(np.argmax(ratio_matrix))
    row_index, axis_index = np.unravel_index(flat_index, ratio_matrix.shape)
    axis_names = ("X", "Y", "Z")

    result.update({
        "median_extent_x": float(medians[0]),
        "median_extent_y": float(medians[1]),
        "median_extent_z": float(medians[2]),
        "maximum_symmetric_extent_ratio": float(ratio_matrix[row_index, axis_index]),
        "maximum_ratio_asset": valid[row_index].get("asset_path"),
        "maximum_ratio_axis": axis_names[axis_index],
    })
    return result


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

ASSET_CSV_FIELDS = [
    "asset_path", "asset_type", "file_extension", "file_exists", "file_size_bytes",
    "points", "cells", "triangles", "surface_area", "volume",
    "bounds_xmin", "bounds_xmax", "bounds_ymin", "bounds_ymax",
    "bounds_zmin", "bounds_zmax", "extent_x", "extent_y", "extent_z",
    "boundary_edges", "connected_components", "finite_coordinates",
    "table_rows", "table_columns", "status", "message",
]

MANIFEST_CSV_FIELDS = [
    "age_months", "age_key_in_manifest", "hemisphere", "surface",
    "manifest_path", "discovered_path", "resolved_path", "file_exists",
    "file_size_bytes", "points", "cells", "triangles", "surface_area", "volume",
    "bounds_xmin", "bounds_xmax", "bounds_ymin", "bounds_ymax",
    "bounds_zmin", "bounds_zmax", "extent_x", "extent_y", "extent_z",
    "boundary_edges", "connected_components", "finite_coordinates", "status", "message",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def write_summary_plot(rows: list[dict[str, Any]], output_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ages = sorted({
        int(row["age_months"])
        for row in rows
        if isinstance(row.get("age_months"), (int, float))
    })
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

    for surface in MANIFEST_SURFACES:
        xs: list[int] = []
        ys: list[float] = []
        for age in ages:
            values = [
                row.get("points") for row in rows
                if row.get("surface") == surface
                and row.get("age_months") == age
                and isinstance(row.get("points"), (int, float))
            ]
            if values:
                xs.append(age)
                ys.append(float(np.mean(values)))
        if xs:
            axes[0].plot(xs, ys, marker="o", label=surface)

    axes[0].set_title("Mean vertices per hemisphere by age and surface")
    axes[0].set_xlabel("Age (months)")
    axes[0].set_ylabel("Number of vertices")
    axes[0].grid(True, alpha=0.25)
    if axes[0].lines:
        axes[0].legend(ncol=3, fontsize=8)

    statuses = Counter(row.get("status", "ERROR") for row in rows)
    axes[1].bar(list(statuses.keys()), list(statuses.values()))
    axes[1].set_title("Manifest QC status counts (PASS is not anatomical validation)")
    axes[1].set_ylabel("Number of entries")
    axes[1].grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent,
        help="Project directory used as the base for relative atlas.json paths.",
    )
    parser.add_argument(
        "--asset-root", type=Path, default=None,
        help="Common directory containing the complete physical resource inventory (default: --root).",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Manifest path (default: ROOT/atlas.json).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory (default: ROOT/qc).",
    )
    parser.add_argument(
        "--asset-extensions", nargs="+", default=None,
        help="Replace the default asset extension allow-list, e.g. --asset-extensions .vtk .vtp .csv .json",
    )
    parser.add_argument(
        "--extra-extensions", nargs="*", default=[],
        help="Additional asset extensions to add to the default allow-list.",
    )
    parser.add_argument(
        "--all-files", action="store_true",
        help="Include every non-excluded file below --asset-root, regardless of extension.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    asset_root = (args.asset_root or root).expanduser().resolve()
    manifest_path = (args.manifest or (root / "atlas.json")).expanduser().resolve()
    out_dir = (args.out or (root / "qc")).expanduser().resolve()

    if not root.is_dir():
        print(f"ERROR: project root is not a directory: {root}", file=sys.stderr)
        return 2
    if not asset_root.is_dir():
        print(f"ERROR: asset root is not a directory: {asset_root}", file=sys.stderr)
        return 2
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("atlas.json must contain a JSON object keyed by age")
    except Exception as exc:
        print(f"ERROR: cannot read manifest: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.asset_extensions:
        allowed_extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                              for ext in args.asset_extensions}
    else:
        allowed_extensions = set(DEFAULT_ASSET_EXTENSIONS)
    for ext in args.extra_extensions:
        allowed_extensions.add(ext.lower() if ext.startswith(".") else f".{ext.lower()}")

    excluded_paths = [out_dir.resolve(), manifest_path.resolve()]
    # First inventory every non-excluded file. This also reports file types
    # outside the allow-list, which is useful when reconciling a historical
    # total such as 889 assets. Only selected resource types are deeply audited.
    all_candidate_files = discover_asset_files(
        asset_root=asset_root,
        allowed_extensions=allowed_extensions,
        all_files=True,
        excluded_paths=excluded_paths,
    )
    physical_files = (
        all_candidate_files
        if args.all_files
        else [p for p in all_candidate_files if file_extension(p) in allowed_extensions]
    )
    selected_set = set(physical_files)
    unselected_files = [p for p in all_candidate_files if p not in selected_set]

    print("\nDiscovering physical assets...")
    print(f"Asset root: {asset_root}")
    print(f"All non-excluded files under asset root: {len(all_candidate_files)}")
    print(f"Selected resource assets for QC:         {len(physical_files)}")
    print("Selected assets by extension:")
    for ext, count in extension_counts(physical_files).items():
        print(f"  {ext:12s} {count:6d}")
    if unselected_files:
        print("File types not selected by the current extension allow-list:")
        for ext, count in extension_counts(unselected_files).items():
            print(f"  {ext:12s} {count:6d}")

    print("\nAuditing physical assets...")
    asset_rows: list[dict[str, Any]] = []
    for index, path in enumerate(physical_files, start=1):
        asset_rows.append(audit_asset_file(path, asset_root))
        if index % 100 == 0 or index == len(physical_files):
            print(f"  Processed {index}/{len(physical_files)}")

    asset_status_counts = Counter(row.get("status", "ERROR") for row in asset_rows)
    asset_type_counts = Counter(row.get("asset_type", "other") for row in asset_rows)
    mesh_rows = [row for row in asset_rows if row.get("asset_type") == "mesh"]
    mesh_status_counts = Counter(row.get("status", "ERROR") for row in mesh_rows)

    print("\nDiscovering cortical surface meshes...")
    discovered_surfaces = discover_surface_files(asset_root)
    print(f"Unique age/hemisphere/surface keys discovered: {len(discovered_surfaces)}")
    print(f"Physical cortical-surface files discovered: {sum(len(v) for v in discovered_surfaces.values())}")

    print("\nAuditing manifest...")
    manifest_rows = build_manifest_rows(
        root=root,
        asset_root=asset_root,
        manifest=manifest,
        discovered=discovered_surfaces,
    )
    manifest_status_counts = Counter(row.get("status", "ERROR") for row in manifest_rows)

    geometry_all_meshes = geometric_summary(mesh_rows, cortical_only=False)
    geometry_cortical = geometric_summary(mesh_rows, cortical_only=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    asset_csv_path = out_dir / "atlas_assets_qc.csv"
    mesh_csv_path = out_dir / "atlas_mesh_qc.csv"
    manifest_csv_path = out_dir / "atlas_manifest_qc.csv"
    json_path = out_dir / "atlas_qc.json"
    plot_path = out_dir / "atlas_qc_summary.png"

    write_csv(asset_csv_path, asset_rows, ASSET_CSV_FIELDS)
    write_csv(mesh_csv_path, mesh_rows, ASSET_CSV_FIELDS)
    write_csv(manifest_csv_path, manifest_rows, MANIFEST_CSV_FIELDS)

    summary = {
        "project_root": str(root),
        "asset_root": str(asset_root),
        "manifest": str(manifest_path),
        "output_directory": str(out_dir),
        "asset_inventory": {
            "assets_discovered": len(asset_rows),
            "all_non_excluded_files_under_asset_root": len(all_candidate_files),
            "files_not_selected_by_extension_filter": len(unselected_files),
            "unselected_file_extension_counts": extension_counts(unselected_files),
            "asset_status_counts": dict(asset_status_counts),
            "asset_type_counts": dict(asset_type_counts),
            "asset_extension_counts": extension_counts(physical_files),
            "assets_with_errors": asset_status_counts.get("ERROR", 0),
            "assets_with_warnings": asset_status_counts.get("WARN", 0),
        },
        "mesh_inventory": {
            "mesh_assets_discovered": len(mesh_rows),
            "mesh_status_counts": dict(mesh_status_counts),
            "meshes_with_errors": mesh_status_counts.get("ERROR", 0),
            "meshes_with_warnings": mesh_status_counts.get("WARN", 0),
        },
        "manifest_audit": {
            "manifest_entries_audited": len(manifest_rows),
            "manifest_status_counts": dict(manifest_status_counts),
            "manifest_errors": manifest_status_counts.get("ERROR", 0),
            "manifest_warnings": manifest_status_counts.get("WARN", 0),
            "manifest_age_keys": list(manifest.keys()),
            "normalized_manifest_ages_months": sorted({normalize_age_key(k) for k in manifest.keys()}),
        },
        "geometry_all_meshes": geometry_all_meshes,
        "geometry_cortical_surfaces_excluding_sphere": geometry_cortical,
        "limitations": [
            "A PASS status indicates basic file/readability checks, not anatomical validation.",
            "Generic image/neuroimaging resources are checked for existence and non-zero size, not decoded.",
            "CSV/TSV and JSON files are syntax/readability checked; their scientific contents are not validated.",
            "Surface extents use the coordinates stored in each mesh; units and coordinate conventions must be established from atlas documentation.",
            "The symmetric maximum extent ratio is max(extent/median, median/extent) for each axis and mesh.",
            "The cortical geometry subset includes OuterSurf, MiddleSurf, InnerSurf and InflateSurf; SphereSurf is excluded.",
            "The 889 historical asset count is reproducible only when --asset-root and the extension allow-list describe the same population as the original inventory.",
        ],
        "asset_entries": asset_rows,
        "manifest_entries": manifest_rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)

    plot_written = write_summary_plot(manifest_rows, plot_path)

    print("\n" + "=" * 66)
    print("INFANTCONNECT ATLAS QUALITY CONTROL")
    print("=" * 66)

    print("\nPHYSICAL ASSET AUDIT")
    print("-" * 42)
    print(f"Assets discovered:             {len(asset_rows)}")
    print(f"Asset types:                   {dict(asset_type_counts)}")
    print(f"Assets with errors:            {asset_status_counts.get('ERROR', 0)}")
    print(f"Asset status counts:           {dict(asset_status_counts)}")

    print("\nMESH AUDIT")
    print("-" * 42)
    print(f"Mesh assets discovered:        {len(mesh_rows)}")
    print(f"Meshes with errors:            {mesh_status_counts.get('ERROR', 0)}")
    print(f"Mesh status counts:            {dict(mesh_status_counts)}")

    print("\nMANIFEST AUDIT")
    print("-" * 42)
    print(f"Manifest entries audited:      {len(manifest_rows)}")
    print(f"Manifest status counts:        {dict(manifest_status_counts)}")

    for title, geometry in (
        ("GEOMETRY: ALL MESHES", geometry_all_meshes),
        ("GEOMETRY: CORTICAL SURFACES (SPHERE EXCLUDED)", geometry_cortical),
    ):
        print(f"\n{title}")
        print("-" * 42)
        print(f"Meshes in population:          {geometry['mesh_files_in_population']}")
        print(f"Meshes with valid geometry:    {geometry['meshes_with_valid_geometry']}")
        print(f"Median surface extent X:       {geometry['median_extent_x']}")
        print(f"Median surface extent Y:       {geometry['median_extent_y']}")
        print(f"Median surface extent Z:       {geometry['median_extent_z']}")
        print(f"Maximum symmetric ratio:       {geometry['maximum_symmetric_extent_ratio']}")
        print(f"Maximum-ratio asset:           {geometry['maximum_ratio_asset']}")
        print(f"Maximum-ratio axis:            {geometry['maximum_ratio_axis']}")

    print("\nOUTPUT FILES")
    print("-" * 42)
    print(f"Asset CSV:                     {asset_csv_path}")
    print(f"Mesh CSV:                      {mesh_csv_path}")
    print(f"Manifest CSV:                  {manifest_csv_path}")
    print(f"JSON report:                   {json_path}")
    if plot_written:
        print(f"QC plot:                       {plot_path}")
    else:
        print("QC plot skipped (install matplotlib to enable it).")
    print("=" * 66)

    has_errors = (
        asset_status_counts.get("ERROR", 0) > 0
        or manifest_status_counts.get("ERROR", 0) > 0
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
