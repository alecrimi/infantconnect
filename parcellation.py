"""Infant-AAL parcellation overlay for the InfantConnect viewer.

Loads the per-vertex labels written by prepare_infant_aal.py and builds two
PyVista meshes that can be drawn on top of any cortical surface (and therefore on
top of the LORETA heat map and next to the tractography):

    boundary_mesh()  thin dark lines where neighbouring vertices have different labels
    filled_mesh()    translucent copy of the surface coloured by region

Labels are attached to vertices, so they only apply to a surface whose vertex
count equals the label file (all surface types of one atlas age/hemisphere).
"""
import colorsys
import json
from pathlib import Path

import numpy as np
import pyvista as pv



# Standard AAL90 abbreviations expanded to readable anatomical names.
# The Infant AAL metadata stores the abbreviated form plus hemisphere suffix.
AAL_FULL_NAMES = {
    "PreCG": "Precentral gyrus",
    "SFGdor": "Superior frontal gyrus, dorsolateral part",
    "ORBsup": "Superior frontal gyrus, orbital part",
    "MFG": "Middle frontal gyrus",
    "ORBmid": "Middle frontal gyrus, orbital part",
    "IFGoperc": "Inferior frontal gyrus, opercular part",
    "IFGtriang": "Inferior frontal gyrus, triangular part",
    "ORBinf": "Inferior frontal gyrus, orbital part",
    "ROL": "Rolandic operculum",
    "SMA": "Supplementary motor area",
    "OLF": "Olfactory cortex",
    "SFGmed": "Superior frontal gyrus, medial part",
    "ORBsupmed": "Superior frontal gyrus, medial orbital part",
    "REC": "Gyrus rectus",
    "INS": "Insula",
    "ACG": "Anterior cingulate and paracingulate gyri",
    "DCG": "Median cingulate and paracingulate gyri",
    "PCG": "Posterior cingulate gyrus",
    "HIP": "Hippocampus",
    "PHG": "Parahippocampal gyrus",
    "AMYG": "Amygdala",
    "CAL": "Calcarine fissure and surrounding cortex",
    "CUN": "Cuneus",
    "LING": "Lingual gyrus",
    "SOG": "Superior occipital gyrus",
    "MOG": "Middle occipital gyrus",
    "IOG": "Inferior occipital gyrus",
    "FFG": "Fusiform gyrus",
    "PoCG": "Postcentral gyrus",
    "SPG": "Superior parietal gyrus",
    "IPL": "Inferior parietal gyrus",
    "SMG": "Supramarginal gyrus",
    "ANG": "Angular gyrus",
    "PCUN": "Precuneus",
    "PCL": "Paracentral lobule",
    "CAU": "Caudate nucleus",
    "PUT": "Putamen",
    "PAL": "Pallidum",
    "THA": "Thalamus",
    "HES": "Heschl's gyrus",
    "STG": "Superior temporal gyrus",
    "TPOsup": "Temporal pole, superior temporal part",
    "MTG": "Middle temporal gyrus",
    "TPOmid": "Temporal pole, middle temporal part",
    "ITG": "Inferior temporal gyrus",
}

def aal_volume_for_age(age_months):
    """Which Infant-AAL label map (neo / 1yr / 2yr) serves a given age."""
    if age_months <= 6:
        return "neo"
    if age_months <= 18:
        return "1yr"
    return "2yr"


def label_palette(n=91):
    """RGB per label. Left/right homologues share a hue (left lighter, right darker)."""
    colors = np.full((n, 3), 128, dtype=np.uint8)
    golden = 0.61803398875
    for i in range(1, n):
        pair = (i + 1) // 2
        h = (pair * golden) % 1.0
        v = 0.95 if i % 2 == 1 else 0.72
        r, g, b = colorsys.hsv_to_rgb(h, 0.60, v)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


class ParcellationStore:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.palette = label_palette(91)
        self.last_error = ""
        self._labels = {}
        self._edges = {}
        self._meta = {}
        meta_path = self.dir / "infant_aal_meta.json"
        if meta_path.is_file():
            try:
                self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:  # unreadable meta is not fatal
                self.last_error = f"meta unreadable: {exc}"

    @property
    def available(self):
        return self.dir.is_dir() and any(self.dir.glob("infant_aal_*_*.npy"))

    def source_volume(self, age):
        return self._meta.get("sources", {}).get(str(age), aal_volume_for_age(age))

    def get_labels(self, age, hemi, n_points):
        key = (age, hemi)
        if key not in self._labels:
            path = self.dir / f"infant_aal_{int(age):02d}_{hemi}.npy"
            self._labels[key] = np.load(path) if path.is_file() else None
        labels = self._labels[key]
        if labels is None:
            self.last_error = f"no labels for {age} mo {hemi}"
            return None
        if len(labels) != n_points:
            self.last_error = (f"{age} mo {hemi}: {len(labels)} labels but the displayed "
                               f"surface has {n_points} vertices (labels not valid here)")
            return None
        return labels

    def get_region_info(self, label):
        """Return the AAL label, acronym, and readable anatomical name.

        ``label`` is the integer stored in the per-vertex label arrays. The
        acronym is read from infant_aal_meta.json so the metadata remains the
        source of truth for the label numbering.
        """
        label = int(label)
        if label <= 0:
            return None
        names = self._meta.get("aal_names", {})
        acronym = names.get(str(label), f"AAL-{label}")
        base_acronym, separator, hemi_suffix = acronym.rpartition("-")
        if not separator:
            base_acronym, hemi_suffix = acronym, ""
        full_name = AAL_FULL_NAMES.get(base_acronym, base_acronym)
        hemisphere = {"L": "Left", "R": "Right"}.get(hemi_suffix, "")
        return {
            "label": label,
            "acronym": acronym,
            "full_name": full_name,
            "hemisphere": hemisphere,
        }

    def _boundary_edges(self, age, hemi, mesh, labels):
        key = (age, hemi, mesh.n_cells)
        if key in self._edges:
            return self._edges[key]
        tri = mesh if mesh.is_all_triangles else mesh.triangulate()
        f = np.asarray(tri.faces).reshape(-1, 4)[:, 1:]
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        e = np.sort(e, axis=1)
        e = e[labels[e[:, 0]] != labels[e[:, 1]]]
        e = np.unique(e, axis=0) if len(e) else e
        self._edges[key] = e
        return e

    @staticmethod
    def _offset_points(mesh, fraction):
        """Vertices pushed slightly outward along the normals to avoid z-fighting."""
        pts = np.asarray(mesh.points, dtype=float)
        try:
            normals = mesh.compute_normals(
                cell_normals=False, point_normals=True, split_vertices=False,
                auto_orient_normals=False, inplace=False,
            ).point_data["Normals"]
            normals = np.asarray(normals, dtype=float)
        except Exception:
            return pts
        centre = pts.mean(axis=0)
        if np.mean(np.einsum("ij,ij->i", normals, pts - centre)) < 0:
            normals = -normals
        diag = float(np.linalg.norm(np.ptp(pts, axis=0)))
        return pts + normals * (fraction * diag)

    def boundary_mesh(self, mesh, age, hemi, offset_fraction=0.002):
        labels = self.get_labels(age, hemi, mesh.n_points)
        if labels is None:
            return None
        edges = self._boundary_edges(age, hemi, mesh, labels)
        if len(edges) == 0:
            return None
        pts = self._offset_points(mesh, offset_fraction)
        used, inverse = np.unique(edges.ravel(), return_inverse=True)
        seg = inverse.reshape(-1, 2)
        lines = np.column_stack((np.full(len(seg), 2), seg)).ravel()
        return pv.PolyData(pts[used], lines=lines)

    def filled_mesh(self, mesh, age, hemi, offset_fraction=0.0012):
        labels = self.get_labels(age, hemi, mesh.n_points)
        if labels is None:
            return None
        out = mesh.copy(deep=True)
        out.points = self._offset_points(mesh, offset_fraction)
        out.point_data["aal_label"] = labels.astype(np.int32)
        out.point_data["aal_rgb"] = self.palette[np.clip(labels, 0, 90)]
        return out
