from pathlib import Path
import json
import traceback
import sys
import zipfile
import re

import numpy as np
import pooch
import pyvista as pv
import mne
import mne.io.constants
import vtk
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QMessageBox, QCheckBox, QProgressDialog,
    QFileDialog, QPushButton, QRadioButton, QButtonGroup, QFrame, QSplitter,
)

from eeg_trace_panel import EEGTracePanel
from effconn import compute_effective_connectivity
from bundle_animation import BundleAnimationMixin
from edf_quality import show_edf_quality_dialog
from edf_preprocess import show_edf_preprocess_dialog
from parcellation import ParcellationStore

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "atlas.json"
TRACT_DIR = ROOT / "tract_atlases"

DISPLAY_MAX_AGE_MONTHS = 72
MNE_SUBJECTS_DIR = Path.home() / "Documents" / "infant_viewer" / "mne_templates"

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

SURFACE_TYPES = {
    "Outer surface": "outer",
    "Middle surface": "middle",
    "Inner surface": "inner",
    "Inflated surface": "inflated",
}




def load_surface(relative_path):
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Mesh not found:\n{path}")
    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Empty surface mesh:\n{path}")
    return mesh


def _find_zip_entry(zf, suffix):
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if name == suffix or name.endswith("/" + suffix):
            return name
    return None


def fetch_scalp_and_montage(template_name, subjects_dir):
    subject = SUBJECT_DIR_NAMES[template_name]
    subject_dir = Path(subjects_dir) / subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    has_scalp = any((subject_dir / s).is_file() for s in SCALP_ZIP_CANDIDATES)
    has_montage = any((subject_dir / s).is_file() for s in MONTAGE_ALL_CANDIDATES)
    has_inner = any((subject_dir / s).is_file() for s in SKULL_ZIP_CANDIDATES if "inner" in s)
    has_outer = any((subject_dir / s).is_file() for s in SKULL_ZIP_CANDIDATES if "outer" in s and "skin" not in s)

    if has_scalp and has_montage and has_inner and has_outer:
        return subject_dir

    zip_path = Path(subjects_dir) / f"{subject}.zip"
    if not zip_path.is_file():
        print(f"[{template_name}] downloading {subject}.zip ...")
        pooch.retrieve(
            url=TEMPLATE_URL.format(subject=subject),
            known_hash=None, fname=zip_path.name,
            path=str(subjects_dir), progressbar=True,
        )

    print(f"[{template_name}] extracting scalp, skulls, + montage ...")
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for suffix in SCALP_ZIP_CANDIDATES + SKULL_ZIP_CANDIDATES + MONTAGE_ALL_CANDIDATES:
            entry = _find_zip_entry(zf, suffix)
            if entry is None:
                continue
            dest = subject_dir / suffix
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted.append(suffix)

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
    print(f"[{template_name}] kept {extracted}; removed zip.")
    return subject_dir


def load_mne_infant_scalp(template_name):
    subject_dir = fetch_scalp_and_montage(template_name, MNE_SUBJECTS_DIR)

    scalp_path = next(
        (subject_dir / s for s in SCALP_ZIP_CANDIDATES
         if (subject_dir / s).is_file()), None,
    )
    if scalp_path is None:
        raise FileNotFoundError(f"No scalp surface in {subject_dir}")

    vertices, faces = mne.read_surface(str(scalp_path))
    faces_pv = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    scalp = pv.PolyData(vertices, faces_pv)

    names = "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2".split()

    fif_path = next(
        (subject_dir / s for s in MONTAGE_FIF_CANDIDATES
         if (subject_dir / s).is_file()), None,
    )
    surf_path = next(
        (subject_dir / s for s in MONTAGE_SURF_CANDIDATES
         if (subject_dir / s).is_file()), None,
    )

    if fif_path is not None:
        montage = mne.channels.read_dig_fif(str(fif_path), verbose="error")
        montage.rename_channels(
            {f"EEG{i:03d}": name for i, name in enumerate(names, start=1)}
        )
        positions = montage.get_positions()["ch_pos"]
        raw_positions = {
            name: np.asarray(positions[name], dtype=float) * 1000.0
            for name in names if name in positions
        }
    elif surf_path is not None:
        verts, _ = mne.read_surface(str(surf_path))
        if len(verts) < len(names):
            raise ValueError(f"{surf_path} has {len(verts)} vertices")
        raw_positions = {
            name: np.asarray(verts[i], dtype=float)
            for i, name in enumerate(names)
        }
    else:
        raise FileNotFoundError(f"No montage in {subject_dir}")

    electrode_positions = {}
    for name, p_mm in raw_positions.items():
        idx = scalp.find_closest_point(p_mm)
        electrode_positions[name] = np.asarray(scalp.points[idx], dtype=float)

    fiducials = {}
    fid_path = subject_dir / "surf" / "fiducials.surf"
    if fid_path.is_file():
        try:
            fid_verts, _ = mne.read_surface(str(fid_path))
            if len(fid_verts) >= 3:
                fiducials = {
                    'lpa': fid_verts[0] / 1000.0,
                    'nasion': fid_verts[1] / 1000.0,
                    'rpa': fid_verts[2] / 1000.0,
                }
        except Exception:
            pass

    if not fiducials:
        fiducials = {
            'nasion': scalp.points[np.argmax(scalp.points[:, 1])] / 1000.0,
            'lpa': scalp.points[np.argmin(scalp.points[:, 0])] / 1000.0,
            'rpa': scalp.points[np.argmax(scalp.points[:, 0])] / 1000.0,
        }

    return scalp, electrode_positions, montage if fif_path else None, fiducials


def fit_template_to_brain(scalp, electrode_positions, reference_mesh,
                          margin=(1.12, 1.10), clearance=0.03):
    scalp = scalp.copy()
    E = np.vstack(list(electrode_positions.values()))
    e_lo, e_hi = E.min(axis=0), E.max(axis=0)
    b = np.asarray(reference_mesh.bounds)
    b_lo, b_hi = b[[0, 2, 4]], b[[1, 3, 5]]

    ex = max(e_hi[0] - e_lo[0], 1e-6)
    ey = max(e_hi[1] - e_lo[1], 1e-6)

    sx = margin[0] * (b_hi[0] - b_lo[0]) / ex
    sy = margin[1] * (b_hi[1] - b_lo[1]) / ey
    scale = np.array([sx, sy, 0.5 * (sx + sy)])

    src = np.array([(e_lo[0] + e_hi[0]) / 2, (e_lo[1] + e_hi[1]) / 2, e_hi[2]])
    dst = np.array([
        (b_lo[0] + b_hi[0]) / 2,
        (b_lo[1] + b_hi[1]) / 2,
        b_hi[2] + clearance * (b_hi[2] - b_lo[2]),
    ])

    def f(p):
        return (np.asarray(p, dtype=float) - src) * scale + dst

    scalp.points = f(scalp.points)
    return scalp, {name: f(pos) for name, pos in electrode_positions.items()}


def fit_tracts_to_brain(tracts, reference_mesh, margin=0.95):
    tracts = tracts.copy()

    b = np.asarray(reference_mesh.bounds)
    b_lo, b_hi = b[[0, 2, 4]], b[[1, 3, 5]]
    b_size = b_hi - b_lo
    b_center = (b_lo + b_hi) / 2

    t = np.asarray(tracts.bounds)
    t_lo, t_hi = t[[0, 2, 4]], t[[1, 3, 5]]
    t_size = np.maximum(t_hi - t_lo, 1e-6)
    t_center = (t_lo + t_hi) / 2

    scale = (b_size / t_size) * margin

    tracts.points = (tracts.points - t_center) * scale + b_center
    return tracts

def add_direction_colors(tracts):
    """Per-point RGB from the local fiber direction (standard DTI convention).

    Red   = left-right (x), green = anterior-posterior (y), blue = superior-inferior (z).
    """
    n_pts = tracts.n_points
    pts = np.asarray(tracts.points, dtype=float)
    tang = np.zeros((n_pts, 3), dtype=float)

    for cid in range(tracts.n_cells):
        ids = np.asarray(tracts.get_cell(cid).point_ids, dtype=np.int64)
        n = len(ids)
        if n < 2:
            continue
        p = pts[ids]
        t = np.empty_like(p)
        t[1:-1] = p[2:] - p[:-2]
        t[0] = p[1] - p[0]
        t[-1] = p[-1] - p[-2]
        tang[ids] = t

    norm = np.linalg.norm(tang, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    rgb = np.abs(tang / norm)
    tracts["direction_rgb"] = (rgb * 255).astype(np.uint8)
    return tracts


class InfantBrainViewer(BundleAnimationMixin, QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Infant Brain 3D Visualizer")
        self.resize(1400, 900)

        self.status = self.statusBar()
        self.status.setStyleSheet(
            "QStatusBar { color: #E0E0E0; background: #202833; }"
        )
        self.status.showMessage("Ready.")

        if not MANIFEST.is_file():
            raise FileNotFoundError(f"Manifest not found:\n{MANIFEST}")

        with MANIFEST.open("r", encoding="utf-8") as f:
            self.atlas = json.load(f)

        self.available_ages = sorted(
            int(age) for age, entry in self.atlas.items()
            if any(entry.get(h, {}).get("outer") for h in ("left", "right"))
        )
        if not self.available_ages:
            raise RuntimeError("No usable outer surfaces found in atlas.json.")

        self.mesh_cache = {}
        self.wireframe_cache = {}
        self.scalp_templates = {}
        self.electrode_templates = {}
        self.template_errors = {}

        self.tract_cache = {}
        self.tract_load_errors = {}

        self.parcellation = ParcellationStore(ROOT / "parcellation")
        self._parc_note = ""

        self.loreta_data = None
        self.connectome_data = None
        self.loreta_file = None
        self.loreta_raw = None

        self.effconn_raw = None
        self.effconn_source_file = None
        self.effconn_matrix = None
        self.effconn_nodes = None
        self.effconn_method = None

        self._saved_camera = None
        self._roi_pick_info = {}
        self._roi_picker = vtk.vtkPointPicker()
        self._roi_picker.SetTolerance(0.015)
        self._roi_picker.PickFromListOn()
        self._selected_roi_actor = None
        self._selected_roi = None

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_step)
        # Each entry is a bundle of fibers seeded at a random EEG sensor
        # (see bundle_animation.BundleAnimationMixin).
        self.anim_pulses = []

        self._build_ui()
        self._connect_signals()
        self.update_view()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        age_row = QHBoxLayout()
        layout.addLayout(age_row)
        age_row.addWidget(QLabel("Age (months):"))

        self.age_slider = QSlider(Qt.Orientation.Horizontal)
        self.age_slider.setRange(
            min(self.available_ages),
            max(max(self.available_ages), DISPLAY_MAX_AGE_MONTHS),
        )
        self.age_slider.setSingleStep(1)
        self.age_slider.setPageStep(6)
        self.age_slider.setTracking(False)
        self.age_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.age_slider.setTickInterval(6)
        initial_age = min(self.available_ages, key=lambda a: abs(a - 12))
        self.age_slider.setValue(initial_age)
        age_row.addWidget(self.age_slider, stretch=1)

        self.age_label = QLabel(f"{initial_age} months")
        self.age_label.setMinimumWidth(90)
        self.age_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.age_label.setStyleSheet(
            "font-weight: bold; color: #FFD700; padding: 0 6px;"
        )
        age_row.addWidget(self.age_label)

        controls = QHBoxLayout()
        layout.addLayout(controls)
        controls.addWidget(QLabel("Hemisphere:"))

        self.hemisphere_combo = QComboBox()
        self.hemisphere_combo.addItems(["Both", "Left", "Right"])
        controls.addWidget(self.hemisphere_combo)

        controls.addWidget(QLabel("Surface:"))
        self.surface_combo = QComboBox()
        self.surface_combo.addItems(SURFACE_TYPES.keys())
        controls.addWidget(self.surface_combo)

        controls.addWidget(QLabel("Rendering:"))
        self.render_combo = QComboBox()
        self.render_combo.addItems(["Smooth", "Flat", "Wireframe"])
        controls.addWidget(self.render_combo)

        self.axes_checkbox = QCheckBox("Show 3D axes")
        self.axes_checkbox.setChecked(False)
        controls.addWidget(self.axes_checkbox)

        self.electrodes_checkbox = QCheckBox("Show scalp + electrodes")
        self.electrodes_checkbox.setChecked(True)
        controls.addWidget(self.electrodes_checkbox)

        self.tracts_checkbox = QCheckBox("Show tractography")
        self.tracts_checkbox.setChecked(False)
        controls.addWidget(self.tracts_checkbox)

        self.animate_checkbox = QCheckBox("Animate glow")
        self.animate_checkbox.setChecked(False)
        self.animate_checkbox.setEnabled(False)
        controls.addWidget(self.animate_checkbox)

        self.loreta_checkbox = QPushButton("Static LORETA")
        controls.addWidget(self.loreta_checkbox)

        self.connectome_checkbox = QCheckBox("Show Connectome")
        self.connectome_checkbox.setChecked(False)
        self.connectome_checkbox.setEnabled(False)
        controls.addWidget(self.connectome_checkbox)

        self.dynamic_loreta_button = QPushButton("Dynamic LORETA")
        self.dynamic_loreta_button.setToolTip(
            "Reconstruct time-varying sLORETA source estimates from an EDF and save them to HDF5 (no visualization)"
        )
        controls.addWidget(self.dynamic_loreta_button)

        qc_row = QHBoxLayout()
        layout.addLayout(qc_row)

        self.edf_quality_button = QPushButton("Check EDF quality…")
        self.edf_quality_button.setToolTip(
            "Load an EDF recording and check its integrity and signal quality"
        )
        qc_row.addWidget(self.edf_quality_button)

        self.edf_preprocess_button = QPushButton("Preprocess EDF…")
        self.edf_preprocess_button.setToolTip(
            "Filter, resample and re-reference an EDF recording and save a new EDF"
        )
        qc_row.addWidget(self.edf_preprocess_button)

        qc_row.addStretch(1)

        parc_row = QHBoxLayout()
        layout.addLayout(parc_row)
        parc_row.addWidget(QLabel("Parcellation:"))

        self.parc_group = QButtonGroup(self)
        self.parc_off = QRadioButton("Off")
        self.parc_edges = QRadioButton("Infant AAL boundaries")
        self.parc_fill = QRadioButton("Infant AAL filled")
        for i, rb in enumerate((self.parc_off, self.parc_edges, self.parc_fill)):
            self.parc_group.addButton(rb, i)
            parc_row.addWidget(rb)
        self.parc_off.setChecked(True)
        if not self.parcellation.available:
            for rb in (self.parc_edges, self.parc_fill):
                rb.setEnabled(False)
                rb.setToolTip("Run prepare_infant_aal.py first (no parcellation/ folder found)")
        parc_row.addStretch(1)

        eff_row = QHBoxLayout()
        layout.addLayout(eff_row)

        eff_row.addWidget(QLabel("Eff.Connectivity:"))

        self.effconn_combo = QComboBox()
        self.effconn_combo.addItems(["Granger causality", "Transfer entropy"])
        eff_row.addWidget(self.effconn_combo)

        self.effconn_checkbox = QCheckBox("Compute && show")
        self.effconn_checkbox.setChecked(False)
        eff_row.addWidget(self.effconn_checkbox)

        self.effconn_change_file_btn = QPushButton("Change file…")
        self.effconn_change_file_btn.setToolTip(
            "Load a different EDF file for effective-connectivity analysis"
        )
        eff_row.addWidget(self.effconn_change_file_btn)

        self.effconn_status_label = QLabel("")
        self.effconn_status_label.setStyleSheet("color: #AAAAAA;")
        eff_row.addWidget(self.effconn_status_label)

        eff_row.addStretch(1)
        
        self.about_button = QPushButton("About")
        self.about_button.setToolTip("Show software information")
        eff_row.addWidget(self.about_button)

        self.plotter = QtInteractor(central)
        self.plotter.set_background("#202833")

        # EEG time series (left) | 3D view (right). The trace panel stays
        # hidden until an EDF has been processed with Static LORETA.
        self.eeg_panel = EEGTracePanel()
        self.eeg_panel.setVisible(False)
        self.view_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.view_splitter.addWidget(self.eeg_panel)
        self.view_splitter.addWidget(self.plotter.interactor)
        self.view_splitter.setStretchFactor(0, 1)
        self.view_splitter.setStretchFactor(1, 2)
        self.view_splitter.setCollapsible(1, False)
        layout.addWidget(self.view_splitter, stretch=1)

        view_row = QHBoxLayout()
        layout.addLayout(view_row)

        self.axial_button = QPushButton("Axial view")
        self.coronal_button = QPushButton("Coronal view")
        self.sagittal_button = QPushButton("Sagittal view")

        view_row.addWidget(self.axial_button)
        view_row.addWidget(self.coronal_button)
        view_row.addWidget(self.sagittal_button)
        view_row.addStretch(1)
        self.screenshot_button = QPushButton("Save as screenshot")
        view_row.addWidget(self.screenshot_button)


        # Dedicated ROI information panel. Keep this separate from the status
        # bar, which is reserved for the application's view/configuration summary.
        self.roi_panel = QFrame()
        self.roi_panel.setObjectName("roiInfoPanel")
        self.roi_panel.setMinimumHeight(94)
        self.roi_panel.setMaximumHeight(118)
        self.roi_panel.setStyleSheet(
            "QFrame#roiInfoPanel { background: #303B49; "
            "border: 1px solid #526273; border-radius: 7px; }"
        )
        roi_panel_layout = QVBoxLayout(self.roi_panel)
        roi_panel_layout.setContentsMargins(16, 9, 16, 9)
        roi_panel_layout.setSpacing(4)

        self.roi_info_title = QLabel("INFANT AAL · REGION INFORMATION")
        self.roi_info_title.setStyleSheet(
            "QLabel { color: #9FB3C8; font-size: 10pt; font-weight: bold; "
            "background: transparent; border: none; }"
        )
        roi_panel_layout.addWidget(self.roi_info_title)

        self.roi_info_label = QLabel(
            "Select Infant AAL boundaries or filled mode, then click a cortical region."
        )
        self.roi_info_label.setWordWrap(True)
        self.roi_info_label.setMinimumHeight(42)
        self.roi_info_label.setStyleSheet(
            "QLabel { color: #F2F2F2; font-size: 15pt; font-weight: 600; "
            "background: transparent; border: none; padding: 1px 0; }"
        )
        roi_panel_layout.addWidget(self.roi_info_label)
        layout.addWidget(self.roi_panel)

    def _connect_signals(self):
        self.age_slider.valueChanged.connect(self.update_view)
        self.age_slider.sliderMoved.connect(self._update_age_label)
        self.hemisphere_combo.currentIndexChanged.connect(self.update_view)
        self.surface_combo.currentIndexChanged.connect(self.update_view)
        self.render_combo.currentIndexChanged.connect(self.update_view)
        self.axes_checkbox.toggled.connect(self.update_view)
        self.electrodes_checkbox.toggled.connect(self.update_view)
        self.tracts_checkbox.toggled.connect(self.update_view)
        self.animate_checkbox.toggled.connect(self.update_view)
        self.loreta_checkbox.clicked.connect(self.on_loreta_toggled)
        self.connectome_checkbox.toggled.connect(self.update_view)
        self.dynamic_loreta_button.clicked.connect(self.on_dynamic_loreta_clicked)
        self.screenshot_button.clicked.connect(self.save_screenshot)

        self.edf_quality_button.clicked.connect(
            lambda: show_edf_quality_dialog(self)
        )
        self.edf_preprocess_button.clicked.connect(
            lambda: show_edf_preprocess_dialog(self)
        )

        self.parc_group.idClicked.connect(self.update_view)
        self.effconn_checkbox.toggled.connect(self.on_effconn_toggled)
        self.effconn_combo.currentIndexChanged.connect(self.on_effconn_method_changed)
        self.effconn_change_file_btn.clicked.connect(self._change_effconn_file)
        
        self.about_button.clicked.connect(
            lambda: QMessageBox.about(
                self,
                "About",
                "Software made by Alessandro Crimi, "
                "AGH University of Krakow, Poland.\n\n"
                "License Apache v.2",
            )
        )
        
        self.axial_button.clicked.connect(lambda: self._set_view("axial"))
        self.coronal_button.clicked.connect(lambda: self._set_view("coronal"))
        self.sagittal_button.clicked.connect(lambda: self._set_view("sagittal"))

        # Restrict left-click picking to the cortical surface actors registered
        # in update_view; other overlays (scalp, tracts, electrodes) are ignored.
        self.plotter.enable_point_picking(
            callback=self._on_roi_picked,
            picker=self._roi_picker,
            use_picker=True,
            left_clicking=True,
            right_clicking=False,
            show_message=False,
            show_point=False,
            clear_on_no_selection=False,
        )

    @staticmethod
    def _vtk_actor_key(actor):
        if actor is None:
            return None
        try:
            return actor.GetAddressAsString("")
        except Exception:
            return str(actor)

    def _on_roi_picked(self, point, picker=None):
        """Resolve a picked cortical point to its per-vertex Infant-AAL label."""
        try:
            picker = picker or self._roi_picker
            actor = picker.GetActor()
            info = self._roi_pick_info.get(self._vtk_actor_key(actor))
            if info is None:
                return

            point_id = int(picker.GetPointId()) if hasattr(picker, "GetPointId") else -1
            mesh = info["mesh"]
            labels = info["labels"]
            if point_id < 0 or point_id >= len(labels):
                picked = point if point is not None else picker.GetPickPosition()
                picked = np.asarray(picked, dtype=float).reshape(1, 3)
                _, point_id = info["tree"].query(picked[0])
            point_id = int(point_id)
            label = int(labels[point_id])
            hemi = info["hemi"]

            if label <= 0:
                self.roi_info_label.setText(
                    f"{hemi.title()} hemisphere · Unlabelled cortical point "
                    "(outside Infant AAL coverage)"
                )
                self._remove_selected_roi_highlight()
                return

            region = self.parcellation.get_region_info(label)
            if region is None:
                return
            acronym = region["acronym"]
            full_name = region["full_name"]
            hemi_name = "Left" if hemi == "left" else "Right"
            message = (
                f"{acronym} — {full_name} ({hemi_name} hemisphere)  |  "
                f"Infant AAL ROI {label}"
            )
            self.roi_info_label.setText(message)
            self._highlight_selected_roi(hemi, mesh, labels, label)
            self._selected_roi = (hemi, label)
        except Exception as exc:
            print(f"ROI picking failed: {type(exc).__name__}: {exc}")

    def _remove_selected_roi_highlight(self):
        try:
            self.plotter.remove_actor("selected_roi_boundary", reset_camera=False, render=False)
        except Exception:
            pass
        self._selected_roi_actor = None
        self._selected_roi = None
        self.plotter.render()

    def _highlight_selected_roi(self, hemi, mesh, labels, label):
        """Draw a bright outline around the clicked ROI without hiding LORETA."""
        self._remove_selected_roi_highlight()
        try:
            tri = mesh if mesh.is_all_triangles else mesh.triangulate()
            faces = np.asarray(tri.faces).reshape(-1, 4)[:, 1:]
            edges = np.concatenate((
                faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
            ), axis=0)
            edges = np.sort(edges, axis=1)
            is_roi = labels[edges] == int(label)
            border = np.logical_xor(is_roi[:, 0], is_roi[:, 1])
            edges = np.unique(edges[border], axis=0)
            if not len(edges):
                return

            points = self.parcellation._offset_points(mesh, 0.0035)
            used, inverse = np.unique(edges.ravel(), return_inverse=True)
            segments = inverse.reshape(-1, 2)
            lines = np.column_stack((np.full(len(segments), 2), segments)).ravel()
            boundary = pv.PolyData(points[used], lines=lines)
            self._selected_roi_actor = self.plotter.add_mesh(
                boundary, color="#00E5FF", line_width=4.0,
                render_lines_as_tubes=True, lighting=False,
                reset_camera=False, name="selected_roi_boundary",
            )
            self.plotter.render()
        except Exception as exc:
            print(f"ROI highlight failed: {type(exc).__name__}: {exc}")

    def _show_eeg_panel(self, raw, file_path=None):
        """Show `raw` in the EEG trace panel to the left of the 3D view."""
        if raw is None:
            return
        name = Path(file_path).name if file_path else ""
        self.eeg_panel.set_raw(
            raw, title=f"EEG time series — {name}" if name else "EEG time series"
        )
        if not self.eeg_panel.isVisible():
            self.eeg_panel.setVisible(True)
            total_w = max(self.view_splitter.width(), 900)
            self.view_splitter.setSizes([int(total_w * 0.35), int(total_w * 0.65)])

    def _update_age_label(self, value):
        self.age_label.setText(f"{int(value)} months")

    def _set_view(self, view_name):
        if view_name == "axial":
            self.plotter.view_xy()
        elif view_name == "coronal":
            self.plotter.view_vector((0, 1, 0), viewup=(0, 0, 1))  
        elif view_name == "sagittal":
            self.plotter.view_yz()
        try:
            self._saved_camera = self.plotter.camera_position
        except Exception:
            self._saved_camera = None
        self.plotter.render()

    def save_screenshot(self):
        default_path = str(Path.home() / "infant_brain_screenshot.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            default_path,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not file_path:
            return

        path_lower = file_path.lower()
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        if not any(path_lower.endswith(ext) for ext in valid_exts):
            file_path += ".png"

        try:
            self.plotter.screenshot(file_path)
            print(f"Screenshot saved: {file_path}")
            QMessageBox.information(
                self, "Screenshot Saved",
                f"Screenshot saved to:\n{file_path}",
            )
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(
                self, "Screenshot Error",
                f"{type(exc).__name__}: {exc}",
            )

    def _prep_raw_for_effconn(self, file_path):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
        if raw.n_times < 2:
            raise ValueError("The EDF file contains too few time samples.")
        if raw.times[-1] > 60:
            raw.crop(tmax=60.0)

        montage_names = "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2".split()
        lookup = {n.lower(): n for n in montage_names}
        lookup.update({"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"})

        new_ch_names = {}
        for ch in raw.ch_names:
            clean = re.sub(r"^EEG[\s_-]*", "", ch, flags=re.IGNORECASE)
            clean = re.sub(r"[-_\s]*(REF|LE|AV|A1|A2)$", "", clean, flags=re.IGNORECASE)
            target = lookup.get(clean.strip().lower())
            if target and target not in new_ch_names.values():
                new_ch_names[ch] = target
        raw.rename_channels(new_ch_names)

        age = self._nearest_age(self.age_slider.value())
        template_name = self._pick_scalp_template(age)
        if template_name is None:
            raise ValueError("No overlay template available.")
        scalp, electrode_positions, _, _ = load_mne_infant_scalp(template_name)

        channels_to_keep = [ch for ch in raw.ch_names if ch in electrode_positions]
        if len(channels_to_keep) < 4:
            raise ValueError(
                f"Only {len(channels_to_keep)} EDF channels match the 10-20 montage "
                f"(need at least 4). EDF channels: {raw.ch_names}"
            )
        raw.pick(channels_to_keep)

        raw.set_eeg_reference("average", projection=True, verbose=False)
        raw.apply_proj(verbose=False)

        return raw

    def _fit_effconn_nodes(self):
        if self.effconn_raw is None:
            return None

        age = self._nearest_age(self.age_slider.value())
        template_name = self._pick_scalp_template(age)
        if template_name is None:
            raise ValueError("No scalp template available.")
        scalp, electrode_positions, _, _ = load_mne_infant_scalp(template_name)

        ref_parts = []
        for hemi in ("left", "right"):
            m = self._get_mesh(age, hemi, "outer")
            if m is not None:
                ref_parts.append(m)
        if not ref_parts:
            raise ValueError("No brain surface available.")
        reference = ref_parts[0] if len(ref_parts) == 1 else ref_parts[0].merge(ref_parts[1:])

        _, electrodes_fit = fit_template_to_brain(scalp, electrode_positions, reference)

        nodes = []
        for ch in self.effconn_raw.ch_names:
            if ch not in electrodes_fit:
                raise ValueError(f"Channel {ch!r} missing from electrode fit.")
            nodes.append(electrodes_fit[ch])
        return np.array(nodes)

    def _ask_effconn_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EEG File for Effective Connectivity",
            str(Path.home()),
            "EDF Files (*.edf);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not file_path:
            return False
        try:
            raw = self._prep_raw_for_effconn(file_path)
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(
                self, "Eff.Connectivity Error",
                f"{type(exc).__name__}: {exc}"
            )
            return False

        self.effconn_raw = raw
        self.effconn_source_file = file_path
        self.effconn_matrix = None
        self.effconn_nodes = None
        self._show_eeg_panel(raw, file_path)
        return True

    def on_effconn_toggled(self, checked):
        if not checked:
            self.effconn_status_label.setText("")
            self.update_view()
            return

        if self.effconn_raw is None:
            if self.loreta_raw is not None:
                self.effconn_raw = self.loreta_raw
                self.effconn_source_file = self.loreta_file
                self.effconn_matrix = None
                self.effconn_nodes = None
                self._show_eeg_panel(self.effconn_raw, self.effconn_source_file)
            elif not self._ask_effconn_file():
                self.effconn_checkbox.setChecked(False)
                return
        else:
            # Re-sync in case another workflow replaced the displayed series.
            self._show_eeg_panel(self.effconn_raw, self.effconn_source_file)

        method = (
            "granger"
            if self.effconn_combo.currentText() == "Granger causality"
            else "transfer_entropy"
        )
        if self.effconn_matrix is None or self.effconn_method != method:
            self._compute_effconn(method)

        if self.effconn_matrix is not None:
            fname = Path(self.effconn_source_file).name if self.effconn_source_file else "?"
            self.effconn_status_label.setText(
                f"{self.effconn_combo.currentText()} on {fname}"
            )
        self.update_view()

    def on_effconn_method_changed(self, index):
        if self.effconn_checkbox.isChecked() and self.effconn_raw is not None:
            method = (
                "granger"
                if self.effconn_combo.currentText() == "Granger causality"
                else "transfer_entropy"
            )
            self._compute_effconn(method)
            if self.effconn_matrix is not None:
                fname = Path(self.effconn_source_file).name if self.effconn_source_file else "?"
                self.effconn_status_label.setText(
                    f"{self.effconn_combo.currentText()} on {fname}"
                )
            self.update_view()

    def _change_effconn_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EEG File for Effective Connectivity",
            str(Path.home()),
            "EDF Files (*.edf);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not file_path:
            return
        try:
            raw = self._prep_raw_for_effconn(file_path)
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(
                self, "Eff.Connectivity Error",
                f"{type(exc).__name__}: {exc}"
            )
            return

        self.effconn_raw = raw
        self.effconn_source_file = file_path
        self.effconn_matrix = None
        self.effconn_nodes = None
        self._show_eeg_panel(raw, file_path)

        if self.effconn_checkbox.isChecked():
            method = (
                "granger"
                if self.effconn_combo.currentText() == "Granger causality"
                else "transfer_entropy"
            )
            self._compute_effconn(method)
            if self.effconn_matrix is not None:
                fname = Path(self.effconn_source_file).name
                self.effconn_status_label.setText(
                    f"{self.effconn_combo.currentText()} on {fname}"
                )
        self.update_view()

    def _compute_effconn(self, method):
        if self.effconn_raw is None:
            return

        progress = QProgressDialog(
            f"Computing {method} …", "Cancel", 0, 3, self
        )
        progress.setWindowTitle("Eff.Connectivity")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        try:
            data = self.effconn_raw.get_data().T
            if data.shape[0] < 10 or data.shape[1] < 2:
                raise ValueError("Too few samples or channels for causality analysis.")

            if method == "granger":
                matrix = compute_effective_connectivity(data, method="granger", lag=5)
            else:
                matrix = compute_effective_connectivity(
                    data, method="transfer_entropy", bins=4, lag=1
                )

            progress.setValue(2)
            QApplication.processEvents()

            self.effconn_matrix = np.asarray(matrix)
            self.effconn_method = method

            progress.setValue(3)
            progress.close()

        except Exception as exc:
            progress.close()
            traceback.print_exc()
            QMessageBox.critical(
                self, "Eff.Connectivity Error",
                f"{type(exc).__name__}: {exc}"
            )
            self.effconn_checkbox.setChecked(False)

    def _render_effconn(self):
        """Draw thin directed arrows from source to target node.

        Two-pass technique:
          - Backing pass: thin dark near-black arrow (the 'border')
          - Foreground:   even thinner bright-yellow arrow drawn on top
        Both arrows are scaled to the full inter-node distance so the tip
        lands exactly on the target node. Deliberately kept thin so the
        graph reads clearly without dominating the brain.
        """
        if self.effconn_matrix is None or self.effconn_raw is None:
            return

        try:
            nodes = self._fit_effconn_nodes()
        except Exception as exc:
            print(f"Eff.Connectivity node fit failed: {exc}")
            return
        if nodes is None or len(nodes) == 0:
            return
        self.effconn_nodes = nodes

        matrix = self.effconn_matrix
        n_nodes = min(len(nodes), matrix.shape[0])

        threshold = 1.0 if self.effconn_method == "granger" else 0.0

        points, vectors, lengths = [], [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                strength = float(matrix[i, j])
                if strength <= threshold:
                    continue
                vec = nodes[j] - nodes[i]
                dist = float(np.linalg.norm(vec))
                if dist < 1e-9:
                    continue
                points.append(nodes[i])
                vectors.append(vec / dist)
                lengths.append(dist)

        if not points:
            return

        poly = pv.PolyData(np.array(points))
        poly["vectors"] = np.array(vectors)
        poly["length"] = np.array(lengths)

        # Pass 1 — thin dark outline (border)
        outline_geom = pv.Arrow(
            tip_length=0.22, tip_radius=0.045, shaft_radius=0.016
        )
        outline_arrows = poly.glyph(
            orient="vectors", scale="length", factor=1.0, geom=outline_geom
        )
        self.plotter.add_mesh(
            outline_arrows,
            color="#0A0A14",
            lighting=False,
            opacity=1.0,
            reset_camera=False,
            name="effconn_arrows_outline",
        )

        # Pass 2 — very thin bright arrow on top
        front_geom = pv.Arrow(
            tip_length=0.22, tip_radius=0.032, shaft_radius=0.010
        )
        front_arrows = poly.glyph(
            orient="vectors", scale="length", factor=1.0, geom=front_geom
        )
        self.plotter.add_mesh(
            front_arrows,
            color="#FFEB3B",
            lighting=False,
            opacity=1.0,
            reset_camera=False,
            name="effconn_arrows",
        )

    def _add_parcellation(self, age, hemi, mesh, mode):
        store = self.parcellation
        if mode == 2:
            fill = store.filled_mesh(mesh, age, hemi)
            if fill is not None:
                self.plotter.add_mesh(
                    fill, scalars="aal_rgb", rgb=True, opacity=0.45,
                    lighting=False, show_scalar_bar=False,
                    reset_camera=False, name=f"aal_fill_{hemi}",
                )
        edges = store.boundary_mesh(mesh, age, hemi)
        if edges is not None:
            self.plotter.add_mesh(
                edges, color="#111111", line_width=1.5, lighting=False,
                reset_camera=False, name=f"aal_edges_{hemi}",
            )
            self._parc_note = f"Infant AAL ({store.source_volume(age)} labels)"
        elif store.last_error:
            print(f"Parcellation not drawn: {store.last_error}")

    def _nearest_age(self, requested_age):
        return min(self.available_ages, key=lambda a: abs(a - requested_age))

    def _ensure_template(self, name):
        if name in self.scalp_templates:
            return True
        if name in self.template_errors:
            return False
        try:
            print(f"Fetching overlay template {name} (first use)...")
            scalp, elec, montage, fiducials = load_mne_infant_scalp(name)
            self.scalp_templates[name] = scalp
            self.electrode_templates[name] = elec
            print(f"Loaded overlay template: {name}")
            return True
        except Exception as exc:
            self.template_errors[name] = str(exc)
            print(f"Overlay template {name!r} unavailable: {exc}")
            return False

    def _pick_scalp_template(self, requested_age):
        candidates = sorted(
            INFANT_SCALP_TEMPLATES.items(),
            key=lambda kv: abs(kv[1] - requested_age),
        )
        for name, _ in candidates:
            if self._ensure_template(name):
                return name
        return None

    def _get_mesh(self, age, hemisphere, surface_type):
        key = (age, hemisphere, surface_type)
        if key in self.mesh_cache:
            return self.mesh_cache[key]
        entry = self.atlas.get(str(age), {}).get(hemisphere, {})
        relative_path = entry.get(surface_type)
        if not relative_path:
            return None
        mesh = load_surface(relative_path)
        self.mesh_cache[key] = mesh
        return mesh

    def _get_wireframe_mesh(self, age, hemisphere, surface_type, mesh):
        key = (age, hemisphere, surface_type)
        if key in self.wireframe_cache:
            return self.wireframe_cache[key]
        try:
            simplified = mesh.decimate(0.99, volume_preservation=True)
            if simplified.n_points == 0 or simplified.n_cells == 0:
                simplified = mesh
        except Exception:
            simplified = mesh
        self.wireframe_cache[key] = simplified
        return simplified

    def _load_tracts(self):
        if self.tract_cache or self.tract_load_errors:
            return

        if not TRACT_DIR.is_dir():
            self.tract_load_errors["_dir"] = f"missing {TRACT_DIR.name}/"
            return

        files = sorted(TRACT_DIR.glob("cluster_*.vtp"))
        if not files:
            self.tract_load_errors["_dir"] = "no cluster files"
            return

        progress = QProgressDialog("Loading tractography...", "Cancel", 0, len(files) + 2, self)
        progress.setWindowTitle("Loading Tracts")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        parts = []
        for i, path in enumerate(files):
            progress.setValue(i)
            if progress.wasCanceled():
                progress.close()
                self.tract_load_errors["_dir"] = "canceled by user"
                return

            try:
                m = pv.read(str(path))
                if m.n_points > 0:
                    m.cell_data["cluster_id"] = np.full(m.n_cells, i, dtype=np.int32)
                    parts.append(m)
            except Exception:
                pass

        if not parts:
            progress.close()
            self.tract_load_errors["_empty"] = "all reads failed"
            return

        progress.setLabelText("Merging meshes...")
        progress.setValue(len(files))
        QApplication.processEvents()

        merged = parts[0]
        for i in range(1, len(parts), 100):
            merged = pv.merge([merged] + parts[i:i + 100], merge_points=False)

        progress.setLabelText("Subsampling fibers...")
        progress.setValue(len(files) + 1)
        QApplication.processEvents()

        TARGET_FIBERS = 2000
        if merged.n_cells > TARGET_FIBERS:
            stride = max(1, merged.n_cells // TARGET_FIBERS)
            merged = merged.extract_cells(list(range(0, merged.n_cells, stride)))

        self.tract_cache["merged"] = merged
        progress.setValue(len(files) + 2)
        progress.close()

    def _configure_lighting(self):
        try:
            self.plotter.remove_all_lights()
        except (AttributeError, TypeError):
            pass
        self.plotter.add_light(pv.Light(
            position=(1.0, 1.5, 2.0), focal_point=(0.0, 0.0, 0.0),
            color="white", intensity=0.75,
        ))
        self.plotter.add_light(pv.Light(
            position=(-1.5, 0.5, 0.8), focal_point=(0.0, 0.0, 0.0),
            color=(0.78, 0.84, 1.0), intensity=0.30,
        ))

    def on_loreta_toggled(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EEG File",
            str(Path.home()),
            "EDF Files (*.edf);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )

        if not file_path:
            return

        self.loreta_file = file_path
        progress = QProgressDialog("Running LORETA...", "Cancel", 0, 5, self)
        progress.setWindowTitle("Forward Modeling")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        def canceled():
            if progress.wasCanceled():
                progress.close()
                return True
            return False

        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")

                if raw.n_times < 2:
                    raise ValueError("The EDF file contains too few time samples.")
                if raw.times[-1] > 60:
                    raw.crop(tmax=60.0)

                montage_names = "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2".split()
                lookup = {n.lower(): n for n in montage_names}
                lookup.update({"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"})

                new_ch_names = {}
                for ch in raw.ch_names:
                    clean = re.sub(r"^EEG[\s_-]*", "", ch, flags=re.IGNORECASE)
                    clean = re.sub(r"[-_\s]*(REF|LE|AV|A1|A2)$", "", clean, flags=re.IGNORECASE)
                    target = lookup.get(clean.strip().lower())
                    if target and target not in new_ch_names.values():
                        new_ch_names[ch] = target
                raw.rename_channels(new_ch_names)

                progress.setValue(1)
                QApplication.processEvents()
                if canceled():
                    return

                age = self._nearest_age(self.age_slider.value())
                template_name = self._pick_scalp_template(age)
                if not template_name:
                    raise ValueError("No overlay template available.")

                scalp, electrode_positions, _, fiducials = load_mne_infant_scalp(template_name)

                channels_to_keep = [ch for ch in raw.ch_names if ch in electrode_positions]
                if len(channels_to_keep) < 4:
                    raise ValueError(
                        f"Only {len(channels_to_keep)} EDF channels match the 10-20 montage "
                        f"(need at least 4). EDF channels: {raw.ch_names}"
                    )
                raw.pick(channels_to_keep)

                raw_pts = np.array(
                    [np.asarray(electrode_positions[ch], dtype=float) for ch in channels_to_keep]
                )
                center = np.asarray(scalp.center, dtype=float)
                rel = raw_pts - center
                mean_r = float(np.linalg.norm(rel, axis=1).mean())
                if not np.isfinite(mean_r) or mean_r <= 0:
                    raise ValueError("Electrode positions are degenerate (all at one point).")

                HEAD_RADIUS_M = 0.07
                rel_m = rel * (HEAD_RADIUS_M / mean_r)
                ch_pos_m = {ch: rel_m[i] for i, ch in enumerate(channels_to_keep)}

                dig_montage = mne.channels.make_dig_montage(ch_pos=ch_pos_m, coord_frame="head")
                raw.set_montage(dig_montage, match_case=False, on_missing="raise", verbose=False)

                raw.set_eeg_reference("average", projection=True, verbose=False)
                raw.apply_proj(verbose=False)

                progress.setValue(2)
                QApplication.processEvents()
                if canceled():
                    return

                progress.setLabelText("Computing forward solution...")
                QApplication.processEvents()

                radius = HEAD_RADIUS_M
                bem = mne.make_sphere_model(
                    r0=(0.0, 0.0, 0.0), head_radius=radius, verbose=False
                )

                src_radius = radius * 0.75
                step = 0.007
                g = np.arange(-src_radius, src_radius + 1e-9, step)
                xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
                rr = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
                rr = rr[np.linalg.norm(rr, axis=1) <= src_radius]
                nn = np.zeros_like(rr)
                nn[:, 2] = 1.0

                src = mne.setup_volume_source_space(
                    pos=dict(rr=rr, nn=nn), verbose=False
                )
                print(f"Source space: {src[0]['nuse']} sources")

                fwd = mne.make_forward_solution(
                    raw.info, trans=None, src=src, bem=bem,
                    meg=False, eeg=True, mindist=3.0, verbose=False,
                )

                cov = mne.make_ad_hoc_cov(raw.info, verbose=False)
                cov_data = cov["data"].copy()
                if cov_data.ndim == 1:
                    cov_data = np.maximum(cov_data, 1e-10)
                else:
                    cov_data = cov_data + np.eye(cov_data.shape[0]) * 1e-10
                cov["data"] = cov_data

                inv = mne.minimum_norm.make_inverse_operator(
                    raw.info, fwd, cov, loose=1.0, depth=0.8, verbose=False
                )

                try:
                    stc = mne.minimum_norm.apply_inverse_raw(
                        raw, inv, lambda2=0.1, method="dSPM", verbose=False
                    )
                except Exception as e:
                    print(f"dSPM failed: {e}. Using MNE with high regularization.")
                    stc = mne.minimum_norm.apply_inverse_raw(
                        raw, inv, lambda2=10.0, method="MNE", verbose=False
                    )

                progress.setValue(3)
                QApplication.processEvents()
                if canceled():
                    return

                stc.data = np.nan_to_num(stc.data, nan=0.0, posinf=0.0, neginf=0.0)
                source_power = stc.data.max(axis=1)

                ref_parts = [
                    m for m in (self._get_mesh(age, h, "outer") for h in ("left", "right"))
                    if m is not None
                ]
                if not ref_parts:
                    raise ValueError("No brain surface available to align the sources to.")
                reference = ref_parts[0] if len(ref_parts) == 1 else ref_parts[0].merge(ref_parts[1:])

                src_points = src[0]["rr"][stc.vertices[0]] * 1000.0
                s_lo, s_hi = src_points.min(axis=0), src_points.max(axis=0)
                s_size = np.maximum(s_hi - s_lo, 1e-6)
                s_center = (s_lo + s_hi) / 2.0

                b = np.asarray(reference.bounds)
                b_lo, b_hi = b[[0, 2, 4]], b[[1, 3, 5]]
                b_size = b_hi - b_lo
                b_center = (b_lo + b_hi) / 2.0

                src_points = (src_points - s_center) * (b_size / s_size) * 0.95 + b_center

                tree = cKDTree(src_points)
                max_p_val = float(source_power.max())
                if max_p_val == 0 or np.isnan(max_p_val):
                    max_p_val = 1e-12

                self.loreta_data = {
                    "tree": tree,
                    "power": source_power,
                    "max_power": max_p_val,
                }

                progress.setLabelText("Computing connectivity...")
                QApplication.processEvents()

                ch_data = raw.get_data()
                n_samples = ch_data.shape[1]
                std_dev = ch_data.std(axis=1, keepdims=True)
                std_dev[std_dev == 0] = 1e-12
                ch_data = (ch_data - ch_data.mean(axis=1, keepdims=True)) / std_dev
                ch_corr = np.dot(ch_data, ch_data.T) / max(n_samples, 1)
                ch_corr = np.nan_to_num(ch_corr, nan=0.0, posinf=0.0, neginf=0.0)

                _, electrodes_fit = fit_template_to_brain(scalp, electrode_positions, reference)
                nodes_3d = np.array([electrodes_fit[ch] for ch in raw.ch_names])

                self.connectome_data = {"ch_corr": ch_corr, "nodes": nodes_3d}

                self.loreta_raw = raw

                # Show the loaded time series (channels labelled by sensor)
                # to the left of the 3D view.
                self._show_eeg_panel(raw, file_path)

                progress.setValue(5)
                progress.close()
                self.update_view()

        except Exception as exc:
            progress.close()
            traceback.print_exc()
            QMessageBox.critical(self, "LORETA Error", f"{type(exc).__name__}: {exc}")

    def on_dynamic_loreta_clicked(self):
        """Compute continuous sLORETA source time series and export to HDF5.

        This intentionally does not update or render the 3D viewer. The saved
        HDF5 contains the source time series, source coordinates and metadata.
        """
        edf_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EDF for Dynamic LORETA",
            str(Path.home()),
            "EDF Files (*.edf *.EDF);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not edf_path:
            return

        default_name = Path(edf_path).with_name(
            Path(edf_path).stem + "_dynamic_loreta.h5"
        )
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Dynamic LORETA HDF5",
            str(default_name),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not output_path:
            return
        if not output_path.lower().endswith((".h5", ".hdf5")):
            output_path += ".h5"

        progress = QProgressDialog(
            "Preparing EDF for dynamic sLORETA...", "Cancel", 0, 5, self
        )
        progress.setWindowTitle("Dynamic LORETA")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        def was_canceled():
            QApplication.processEvents()
            return progress.wasCanceled()

        try:
            import h5py

            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
            if raw.n_times < 2:
                raise ValueError("The EDF file contains too few time samples.")

            # Match the channel naming convention used by the static source-map
            # workflow, but retain the complete recording (no 60-second crop).
            montage_names = (
                "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 "
                "P3 P4 P7 P8 O1 O2"
            ).split()
            lookup = {name.lower(): name for name in montage_names}
            lookup.update({"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"})
            rename = {}
            for channel in raw.ch_names:
                clean = re.sub(r"^EEG[\s_-]*", "", channel, flags=re.IGNORECASE)
                clean = re.sub(
                    r"[-_\s]*(REF|LE|AV|A1|A2)$", "", clean, flags=re.IGNORECASE
                )
                target = lookup.get(clean.strip().lower())
                if target and target not in rename.values():
                    rename[channel] = target
            if rename:
                raw.rename_channels(rename)

            progress.setValue(1)
            if was_canceled():
                return

            age = self._nearest_age(self.age_slider.value())
            template_name = self._pick_scalp_template(age)
            if not template_name:
                raise ValueError("No infant scalp template is available.")
            scalp, electrode_positions, _, _ = load_mne_infant_scalp(template_name)

            channels_to_keep = [ch for ch in raw.ch_names if ch in electrode_positions]
            if len(channels_to_keep) < 4:
                raise ValueError(
                    f"Only {len(channels_to_keep)} EDF channels match the 10-20 montage "
                    f"(need at least 4). EDF channels: {raw.ch_names}"
                )
            raw.pick(channels_to_keep)

            raw_pts = np.array(
                [np.asarray(electrode_positions[ch], dtype=float) for ch in channels_to_keep]
            )
            center = np.asarray(scalp.center, dtype=float)
            rel = raw_pts - center
            mean_radius = float(np.linalg.norm(rel, axis=1).mean())
            if not np.isfinite(mean_radius) or mean_radius <= 0:
                raise ValueError("Electrode positions are degenerate.")

            head_radius_m = 0.07
            rel_m = rel * (head_radius_m / mean_radius)
            ch_pos_m = {ch: rel_m[i] for i, ch in enumerate(channels_to_keep)}
            montage = mne.channels.make_dig_montage(
                ch_pos=ch_pos_m, coord_frame="head"
            )
            raw.set_montage(montage, match_case=False, on_missing="raise", verbose=False)
            raw.set_eeg_reference("average", projection=True, verbose=False)
            raw.apply_proj(verbose=False)

            progress.setLabelText("Computing forward model...")
            progress.setValue(2)
            if was_canceled():
                return

            sphere = mne.make_sphere_model(
                r0=(0.0, 0.0, 0.0), head_radius=head_radius_m, verbose=False
            )
            source_radius = head_radius_m * 0.75
            step = 0.007
            grid = np.arange(-source_radius, source_radius + 1e-9, step)
            xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
            rr = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
            rr = rr[np.linalg.norm(rr, axis=1) <= source_radius]
            nn = np.zeros_like(rr)
            nn[:, 2] = 1.0
            src = mne.setup_volume_source_space(
                pos=dict(rr=rr, nn=nn), verbose=False
            )
            fwd = mne.make_forward_solution(
                raw.info, trans=None, src=src, bem=sphere,
                meg=False, eeg=True, mindist=3.0, verbose=False,
            )
            noise_cov = mne.make_ad_hoc_cov(raw.info, verbose=False)
            cov_data = noise_cov["data"].copy()
            if cov_data.ndim == 1:
                cov_data = np.maximum(cov_data, 1e-10)
            else:
                cov_data += np.eye(cov_data.shape[0]) * 1e-10
            noise_cov["data"] = cov_data
            inverse_operator = mne.minimum_norm.make_inverse_operator(
                raw.info, fwd, noise_cov, loose=1.0, depth=0.8, verbose=False
            )

            progress.setLabelText("Computing continuous sLORETA source estimates...")
            progress.setValue(3)
            if was_canceled():
                return

            # Apply one fixed inverse operator to the complete continuous EEG.
            lambda2 = 0.1
            stc = mne.minimum_norm.apply_inverse_raw(
                raw, inverse_operator, lambda2=lambda2,
                method="sLORETA", pick_ori="vector", verbose=False,
            )
            source_data = np.nan_to_num(
                stc.data, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32, copy=False)
            source_vertices = np.asarray(stc.vertices[0], dtype=np.int64)
            source_coordinates_m = np.asarray(
                src[0]["rr"][source_vertices], dtype=np.float32
            )
            times = np.asarray(stc.times, dtype=np.float64)

            progress.setLabelText("Writing dynamic source estimates to HDF5...")
            progress.setValue(4)
            if was_canceled():
                return

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(output, "w") as h5:
                h5.attrs["format"] = "infantconnect_dynamic_loreta"
                h5.attrs["format_version"] = "1.0"
                h5.attrs["method"] = "sLORETA"
                h5.attrs["inverse_method"] = "MNE-Python minimum-norm"
                h5.attrs["lambda2"] = float(lambda2)
                h5.attrs["sampling_frequency_hz"] = float(raw.info["sfreq"])
                h5.attrs["tmin_seconds"] = float(stc.tmin)
                h5.attrs["age_template_months"] = int(age)
                h5.attrs["template_name"] = str(template_name)
                h5.attrs["edf_source"] = str(Path(edf_path).resolve())
                h5.attrs["source_space"] = "volume"
                h5.attrs["source_data_note"] = (
                    "MNE sLORETA estimates; standardized source values, "
                    "not raw electrode voltages or absolute current density."
                )

                if source_data.ndim == 3:
                    # Vector source estimates: (source location, XYZ component, time).
                    data_chunks = (
                        max(1, min(source_data.shape[0], 64)),
                        source_data.shape[1],
                        max(1, min(source_data.shape[2], 2048)),
                    )
                elif source_data.ndim == 2:
                    data_chunks = (
                        max(1, min(source_data.shape[0], 128)),
                        max(1, min(source_data.shape[1], 4096)),
                    )
                else:
                    raise ValueError(
                        f"Unexpected source estimate dimensions: {source_data.shape}"
                    )
                source_dataset = h5.create_dataset(
                    "source_data", data=source_data, chunks=data_chunks,
                    compression="gzip", compression_opts=4,
                )
                source_dataset.attrs["dimensions"] = (
                    "source, component_xyz, time"
                    if source_data.ndim == 3 else "source, time"
                )
                if source_data.ndim == 3:
                    source_dataset.attrs["vector_components"] = "x,y,z"
                h5.create_dataset("times_seconds", data=times, compression="gzip")
                h5.create_dataset("source_vertices", data=source_vertices)
                h5.create_dataset(
                    "source_coordinates_m", data=source_coordinates_m,
                    compression="gzip",
                )
                source_orientation_basis = getattr(stc, "source_nn", None)
                if source_orientation_basis is not None:
                    h5.create_dataset(
                        "source_orientation_basis",
                        data=np.asarray(source_orientation_basis, dtype=np.float32),
                        compression="gzip",
                    )
                h5.create_dataset(
                    "channel_names",
                    data=np.asarray(raw.ch_names, dtype=h5py.string_dtype("utf-8")),
                )
                h5.create_dataset(
                    "bad_channels",
                    data=np.asarray(raw.info.get("bads", []), dtype=h5py.string_dtype("utf-8")),
                )

            progress.setValue(5)
            progress.close()
            QMessageBox.information(
                self,
                "Dynamic LORETA Saved",
                "Continuous sLORETA source estimates saved successfully.\n\n"
                f"File: {output}\n"
                f"Source locations: {source_data.shape[0]:,}\n"
                f"Time samples: {source_data.shape[-1]:,}\n"
                f"Sampling rate: {raw.info['sfreq']:g} Hz\n\n"
                "No cortical visualization was generated.",
            )
            self.status.showMessage(f"Dynamic sLORETA saved: {output.name}")

        except Exception as exc:
            progress.close()
            traceback.print_exc()
            QMessageBox.critical(
                self, "Dynamic LORETA Error", f"{type(exc).__name__}: {exc}"
            )

    def update_view(self, *_):
        requested_age = self.age_slider.value()
        age = self._nearest_age(requested_age)

        self._update_age_label(requested_age)

        surface_label = self.surface_combo.currentText()
        surface_type = SURFACE_TYPES[surface_label]
        hemisphere_choice = self.hemisphere_combo.currentText()
        hemispheres = (
            ["left", "right"]
            if hemisphere_choice == "Both"
            else [hemisphere_choice.lower()]
        )
        rendering = self.render_combo.currentText()
        show_axes = self.axes_checkbox.isChecked()
        show_electrodes = self.electrodes_checkbox.isChecked()
        show_tracts = self.tracts_checkbox.isChecked()
        show_animate = self.animate_checkbox.isChecked()
        show_connectome = self.connectome_checkbox.isChecked()
        show_effconn = self.effconn_checkbox.isChecked()
        parc_mode = self.parc_group.checkedId()   # 0 off, 1 boundaries, 2 filled
        self._parc_note = ""
        effconn_active = show_effconn and self.effconn_matrix is not None

        self.anim_timer.stop()
        self.anim_pulses.clear()

        self.plotter.clear()
        self._roi_pick_info.clear()
        try:
            self._roi_picker.InitializePickList()
            self._roi_picker.PickFromListOn()
        except Exception:
            pass
        self._selected_roi_actor = None
        self._selected_roi = None
        if hasattr(self, "roi_info_label"):
            if parc_mode:
                self.roi_info_label.setText(
                    "Click a cortical region to display its acronym, anatomical name and hemisphere."
                )
            else:
                self.roi_info_label.setText(
                    "Enable Infant AAL boundaries or filled mode, then click a cortical region."
                )
        self._configure_lighting()

        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                # Slightly more transparent brain when eff.conn arrows are on,
                # but not so faint that the cortex becomes a haze.
                if effconn_active:
                    loreta_opacity = 0.30
                elif show_tracts:
                    loreta_opacity = 0.40
                else:
                    loreta_opacity = 0.80

                if effconn_active:
                    # Use a neutral, clearly visible cortex while displaying
                    # effective-connectivity arrows. Keep LORETA data loaded,
                    # but temporarily hide its heatmap to avoid visual clutter.
                    brain_opacity = 0.88
                elif show_tracts or self.loreta_data is not None:
                    brain_opacity = 0.30
                else:
                    brain_opacity = 1.0

                for hemi in hemispheres:
                    mesh = self._get_mesh(age, hemi, surface_type)
                    if mesh is None:
                        continue

                    if self.loreta_data is not None and not effconn_active:
                        tree = self.loreta_data["tree"]
                        power = self.loreta_data["power"]
                        max_power = self.loreta_data["max_power"]

                        distances, indices = tree.query(mesh.points)

                        if max_power > 0:
                            power_vals = power[indices] / max_power
                        else:
                            power_vals = np.zeros_like(power[indices])

                        mesh["loreta_power"] = np.nan_to_num(power_vals, nan=0.0, posinf=0.0, neginf=0.0)

                        surface_actor = self.plotter.add_mesh(
                            mesh,
                            scalars="loreta_power",
                            cmap="hot",
                            show_scalar_bar=True,
                            scalar_bar_args={
                                'title': 'LORETA Power',
                                'color': 'white',
                                'title_font_size': 16,
                                'label_font_size': 14,
                                'n_labels': 5,
                                'position_x': 0.02,
                                'position_y': 0.05,
                            },
                            opacity=loreta_opacity,
                            smooth_shading=True,
                            lighting=True,
                            ambient=0.2,
                            diffuse=0.8,
                            specular=0.2,
                            reset_camera=False,
                            name=f"loreta_{hemi}"
                        )

                    else:
                        if rendering == "Wireframe":
                            wire = self._get_wireframe_mesh(age, hemi, surface_type, mesh)
                            self.plotter.add_mesh(
                                wire, style="wireframe", color="black",
                                line_width=1.0, lighting=False, reset_camera=False,
                            )
                            # Invisible full-resolution surface used only for ROI picking.
                            surface_actor = self.plotter.add_mesh(
                                mesh, color="#FFFFFF", opacity=0.001,
                                lighting=False, reset_camera=False,
                                name=f"roi_pick_surface_{hemi}",
                            )
                        elif rendering == "Flat":
                            surface_actor = self.plotter.add_mesh(
                                mesh, color="#9DBBD2", smooth_shading=False,
                                opacity=brain_opacity,
                                lighting=True, ambient=0.22, diffuse=0.78,
                                specular=0.12, specular_power=18, reset_camera=False,
                            )
                        else:
                            surface_actor = self.plotter.add_mesh(
                                mesh, color="#9DBBD2", smooth_shading=True,
                                opacity=brain_opacity,
                                lighting=True, ambient=0.20, diffuse=0.78,
                                specular=0.24, specular_power=28, reset_camera=False,
                            )

                    if parc_mode in (1, 2):
                        labels = self.parcellation.get_labels(age, hemi, mesh.n_points)
                        if labels is not None:
                            self._roi_pick_info[self._vtk_actor_key(surface_actor)] = {
                                "hemi": hemi,
                                "mesh": mesh,
                                "labels": labels,
                                "tree": cKDTree(np.asarray(mesh.points, dtype=float)),
                            }
                            try:
                                self._roi_picker.AddPickList(surface_actor)
                            except Exception:
                                pass

                    if parc_mode:
                        self._add_parcellation(age, hemi, mesh, parc_mode)

                if show_tracts:
                    try:
                        self._load_tracts()
                        if self.tract_cache:
                            tract_mesh = self.tract_cache["merged"]

                            reference_parts = []
                            for hemi_ref in ("left", "right"):
                                ref_mesh = self._get_mesh(age, hemi_ref, "outer")
                                if ref_mesh is not None:
                                    reference_parts.append(ref_mesh)

                            if reference_parts:
                                reference = (
                                    reference_parts[0]
                                    if len(reference_parts) == 1
                                    else reference_parts[0].merge(reference_parts[1:])
                                )
                                tract_display = fit_tracts_to_brain(
                                    tract_mesh, reference, margin=0.95
                                )
                            else:
                                tract_display = tract_mesh

                            tract_display = add_direction_colors(tract_display)

                            self.plotter.add_mesh(
                                tract_display,
                                scalars="direction_rgb",
                                rgb=True,
                                show_scalar_bar=False,
                                line_width=2.5,
                                # Dim the resting fibers while the glow is animating
                                # so the moving bundles stand out.
                                opacity=0.18 if show_animate else 0.9,
                                render_lines_as_tubes=False,
                                lighting=False,
                                reset_camera=False,
                                name="tracts",
                            )

                            if show_animate:
                                self._setup_animation(tract_display)

                    except Exception as exc:
                        print(f"Tract rendering failed: {exc}")

                tracts_ready = show_tracts and bool(self.tract_cache)
                if not tracts_ready and self.animate_checkbox.isChecked():
                    self.animate_checkbox.blockSignals(True)
                    self.animate_checkbox.setChecked(False)
                    self.animate_checkbox.blockSignals(False)
                self.animate_checkbox.setEnabled(tracts_ready)

                if self.loreta_data is None:
                    if self.connectome_checkbox.isChecked():
                        self.connectome_checkbox.blockSignals(True)
                        self.connectome_checkbox.setChecked(False)
                        self.connectome_checkbox.blockSignals(False)
                    self.connectome_checkbox.setEnabled(False)
                else:
                    self.connectome_checkbox.setEnabled(True)

                self.effconn_checkbox.setEnabled(True)
                self.effconn_combo.setEnabled(True)

                show_animate = self.animate_checkbox.isChecked()
                show_connectome = self.connectome_checkbox.isChecked()
                show_effconn = self.effconn_checkbox.isChecked()

                summary = (
                    f"Requested: {requested_age} mo  |  Displaying: {age} mo  |  "
                    f"Hemisphere: {hemisphere_choice}  |  Surface: {surface_label}  |  "
                    f"Render: {rendering}"
                )
                extras = []
                if show_electrodes:
                    extras.append("scalp+electrodes")
                if show_tracts and self.tract_cache:
                    extras.append("tracts")
                if show_animate and show_tracts:
                    extras.append("animation")
                if self.loreta_data is not None:
                    extras.append("LORETA")
                if show_connectome and self.connectome_data is not None:
                    extras.append("connectome")
                if show_effconn and self.effconn_matrix is not None:
                    extras.append(f"eff.conn({self.effconn_method})")
                if self._parc_note:
                    extras.append(self._parc_note)
                if show_axes:
                    extras.append("axes")
                if extras:
                    summary += "  |  Active: " + ", ".join(extras)
                self.status.showMessage(summary)

                if show_connectome and self.connectome_data is not None:
                    try:
                        ch_corr = self.connectome_data['ch_corr']
                        nodes_3d = self.connectome_data['nodes']

                        ref_mesh = None
                        ref_parts = []
                        for h in ("left", "right"):
                            m = self._get_mesh(age, h, surface_type)
                            if m is not None:
                                ref_parts.append(m)

                        if ref_parts:
                            ref_mesh = ref_parts[0] if len(ref_parts) == 1 else ref_parts[0].merge(ref_parts[1:])
                            mesh_tree = cKDTree(ref_mesh.points)
                            snapped_nodes = []
                            for pt in nodes_3d:
                                d, idx = mesh_tree.query(pt)
                                snapped_nodes.append(ref_mesh.points[idx])
                            nodes_3d = np.array(snapped_nodes)

                        n_ch = len(ch_corr)
                        connections = []
                        for i in range(n_ch):
                            for j in range(i + 1, n_ch):
                                connections.append((i, j, ch_corr[i, j]))

                        connections.sort(key=lambda x: x[2], reverse=True)
                        connections = connections[:50]

                        lines = []
                        for i, j, strength in connections:
                            pt1 = nodes_3d[i]
                            pt2 = nodes_3d[j]
                            if np.linalg.norm(pt1 - pt2) > 1e-4:
                                line = pv.Line(pt1, pt2, resolution=10)
                                lines.append(line)

                        if lines:
                            merged_lines = pv.merge(lines)
                            self.plotter.add_mesh(
                                merged_lines,
                                color="#FFD700",
                                line_width=4.0,
                                render_lines_as_tubes=False,
                                opacity=0.9,
                                lighting=False,
                                reset_camera=False,
                                name="connectome"
                            )
                    except Exception as exc:
                        print(f"Connectome rendering failed: {exc}")

                if show_effconn and self.effconn_matrix is not None:
                    try:
                        self._render_effconn()
                    except Exception as exc:
                        print(f"Eff.Connectivity rendering failed: {exc}")

                overlay_template = None
                if show_electrodes:
                    overlay_template = self._pick_scalp_template(requested_age)
                    if overlay_template is not None:
                        scalp_template = self.scalp_templates[overlay_template]
                        electrode_template = self.electrode_templates[overlay_template]

                        reference_parts = []
                        for hemi_ref in ("left", "right"):
                            ref_mesh = self._get_mesh(age, hemi_ref, "outer")
                            if ref_mesh is not None:
                                reference_parts.append(ref_mesh)

                        if reference_parts:
                            reference = (
                                reference_parts[0]
                                if len(reference_parts) == 1
                                else reference_parts[0].merge(reference_parts[1:])
                            )
                            scalp_fit, electrodes_fit = fit_template_to_brain(
                                scalp_template, electrode_template, reference
                            )
                            self.plotter.add_mesh(
                                scalp_fit, color="#E8C9A8", opacity=0.3,
                                smooth_shading=True, lighting=True,
                                reset_camera=False,
                            )
                            if electrodes_fit:
                                names = list(electrodes_fit.keys())
                                pts = np.vstack([electrodes_fit[n] for n in names])

                                center = np.asarray(scalp_fit.center, dtype=float)
                                radial = pts - center
                                rn = np.linalg.norm(radial, axis=1, keepdims=True)
                                rn[rn < 1e-12] = 1.0
                                sscale = float(np.linalg.norm(
                                    np.asarray(scalp_fit.bounds)[[1, 3, 5]]
                                    - np.asarray(scalp_fit.bounds)[[0, 2, 4]]
                                ))
                                pts = pts + 0.025 * sscale * radial / rn

                                marker_radius = max(sscale * 0.022, 0.004)
                                label_pts = pts + (radial / rn) * marker_radius * 1.5
                                spheres = [
                                    pv.Sphere(
                                        radius=marker_radius,
                                        center=tuple(float(v) for v in point),
                                        theta_resolution=16,
                                        phi_resolution=16,
                                    )
                                    for point in pts
                                ]
                                markers = spheres[0]
                                for sphere in spheres[1:]:
                                    markers = markers.merge(sphere, merge_points=False)

                                self.plotter.add_mesh(
                                    markers, color="#FF4D00",
                                    ambient=1.0, diffuse=0.0, specular=0.0,
                                    lighting=False, reset_camera=False,
                                    name="electrode_markers",
                                )
                                self.plotter.add_point_labels(
                                    label_pts, names, name="electrode_labels",
                                    point_size=0, font_size=14,
                                    text_color="#FFFFFF",
                                    shape="rounded_rect", fill_shape=True,
                                    shape_color="#202833", shape_opacity=0.9,
                                    always_visible=False, tolerance=0.002,
                                    show_points=False, render_points_as_spheres=False,
                                    reset_camera=False,
                                )

                if show_axes:
                    self.plotter.show_bounds(
                        grid="front", location="outer", all_edges=True,
                        xtitle="X", ytitle="Y", ztitle="Z",
                        fmt="%.3f", color="gray", font_size=10,
                    )

                if self._saved_camera is not None:
                    try:
                        self.plotter.camera_position = self._saved_camera
                    except Exception:
                        self.plotter.reset_camera()
                else:
                    self.plotter.reset_camera()
                self.plotter.render()

        except Exception as exc:
            print(f"Display error: {exc}")

    def closeEvent(self, event):
        self.anim_timer.stop()
        self.plotter.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    try:
        window = InfantBrainViewer()
        window.show()
        sys.exit(app.exec())
    except Exception as exc:
        QMessageBox.critical(None, "Infant Brain Viewer", str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
