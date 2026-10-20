from pathlib import Path
import json
import sys
import zipfile

import numpy as np
import pooch
import pyvista as pv
import mne
import vtk
from pyvistaqt import QtInteractor

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QMessageBox, QCheckBox, QProgressDialog,
)

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
    if has_scalp and has_montage:
        return subject_dir

    zip_path = Path(subjects_dir) / f"{subject}.zip"
    if not zip_path.is_file():
        print(f"[{template_name}] downloading {subject}.zip ...")
        pooch.retrieve(
            url=TEMPLATE_URL.format(subject=subject),
            known_hash=None, fname=zip_path.name,
            path=str(subjects_dir), progressbar=True,
        )

    print(f"[{template_name}] extracting scalp + montage ...")
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for suffix in SCALP_ZIP_CANDIDATES + MONTAGE_ALL_CANDIDATES:
            entry = _find_zip_entry(zf, suffix)
            if entry is None:
                continue
            dest = subject_dir / suffix
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted.append(suffix)

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
        print(f"[{template_name}] warning: using {surf_path.name} montage")
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
    return scalp, electrode_positions


def fit_template_to_brain(scalp, electrode_positions, reference_mesh,
                          margin=(1.12, 1.10), clearance=0.03):
    scalp = scalp.copy()
    E = np.vstack(list(electrode_positions.values()))
    e_lo, e_hi = E.min(axis=0), E.max(axis=0)
    b = np.asarray(reference_mesh.bounds)
    b_lo, b_hi = b[[0, 2, 4]], b[[1, 3, 5]]

    sx = margin[0] * (b_hi[0] - b_lo[0]) / (e_hi[0] - e_lo[0])
    sy = margin[1] * (b_hi[1] - b_lo[1]) / (e_hi[1] - e_lo[1])
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
    """Anisotropic display-space fit: stretch to fill X, Y, Z independently."""
    tracts = tracts.copy()

    b = np.asarray(reference_mesh.bounds)
    b_lo, b_hi = b[[0, 2, 4]], b[[1, 3, 5]]
    b_size = b_hi - b_lo
    b_center = (b_lo + b_hi) / 2

    t = np.asarray(tracts.bounds)
    t_lo, t_hi = t[[0, 2, 4]], t[[1, 3, 5]]
    t_size = t_hi - t_lo
    t_center = (t_lo + t_hi) / 2

    scale = np.where(t_size > 0, (b_size / t_size) * margin, 1.0)

    tracts.points = (tracts.points - t_center) * scale + b_center
    return tracts


class InfantBrainViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Infant Brain 3D Visualizer")
        self.resize(1400, 900)
        self.setStatusBar(None)  # Removes the default bottom status bar

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

        # Animation state
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_step)
        self.anim_actors = []
        self.anim_tracts_points = []
        self.anim_progress = []
        self.anim_speeds = []

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
        controls.addWidget(self.animate_checkbox)

        self.plotter = QtInteractor(central)
        self.plotter.set_background("#202833")
        layout.addWidget(self.plotter.interactor, stretch=1)

    def _connect_signals(self):
        self.age_slider.valueChanged.connect(self.update_view)
        self.hemisphere_combo.currentIndexChanged.connect(self.update_view)
        self.surface_combo.currentIndexChanged.connect(self.update_view)
        self.render_combo.currentIndexChanged.connect(self.update_view)
        self.axes_checkbox.toggled.connect(self.update_view)
        self.electrodes_checkbox.toggled.connect(self.update_view)
        self.tracts_checkbox.toggled.connect(self.update_view)
        self.animate_checkbox.toggled.connect(self.update_view)

    def _nearest_age(self, requested_age):
        return min(self.available_ages, key=lambda a: abs(a - requested_age))

    def _ensure_template(self, name):
        if name in self.scalp_templates:
            return True
        if name in self.template_errors:
            return False
        try:
            print(f"Fetching overlay template {name} (first use)...")
            scalp, elec = load_mne_infant_scalp(name)
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
        """Load, tag, merge, and subsample cluster_*.vtp with a UI progress dialog."""
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

    def _setup_animation(self, tract_mesh):
        """Select random fibers and prepare the glowing ellipsoids."""
        self.anim_timer.stop()
        self.anim_actors.clear()
        self.anim_tracts_points.clear()
        self.anim_progress.clear()
        self.anim_speeds.clear()

        if tract_mesh.n_cells < 1:
            return

        # Create a high-resolution sphere for smooth geometry
        sphere = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64)
        
        # Calculate distance of every point from the center
        dist = np.linalg.norm(sphere.points, axis=1)
        dist_norm = dist / np.max(dist)
        
        # Smooth Gaussian falloff for the alpha/opacity
        glow_vals = np.exp(- (dist_norm * 3.0) ** 2)
        sphere["glow"] = glow_vals

        n_pulses = min(15, tract_mesh.n_cells)
        cell_ids = np.random.choice(tract_mesh.n_cells, n_pulses, replace=False)

        for cid in cell_ids:
            pts = tract_mesh.extract_cells(int(cid)).points
            if len(pts) < 2:
                continue

            self.anim_tracts_points.append(pts)
            self.anim_progress.append(np.random.rand())
            self.anim_speeds.append(np.random.uniform(0.005, 0.015))

            # Render with standard alpha blending using the glow array
            actor = self.plotter.add_mesh(
                sphere.copy(),
                scalars="glow",
                cmap=["#00FFFF", "#FFFFFF"], # Cyan edge fading to White center
                opacity="glow",              # Maps opacity to the distance
                lighting=False,              # Disable lighting to keep it bright
                smooth_shading=True,
                reset_camera=False,
                show_scalar_bar=False,       # CRITICAL: Removes the label at the bottom!
                name=f"anim_glow_{len(self.anim_actors)}"
            )
            
            self.anim_actors.append(actor)

        if self.anim_actors:
            self.anim_timer.start(20)

    def _animate_step(self):
        """Timer callback: move and rotate the glowing ellipsoids along the fibers."""
        if not self.tracts_checkbox.isChecked() or not self.animate_checkbox.isChecked():
            self.anim_timer.stop()
            return

        for i in range(len(self.anim_tracts_points)):
            pts = self.anim_tracts_points[i]
            prog = self.anim_progress[i] + self.anim_speeds[i]
            actor = self.anim_actors[i]

            if prog >= 1.0:
                prog = 0.0 
            self.anim_progress[i] = prog

            # Interpolate position and get forward tangent
            n_pts = len(pts)
            exact_idx = prog * (n_pts - 1)
            idx = int(exact_idx)
            t = exact_idx - idx
            
            if idx >= n_pts - 1:
                pos = pts[-1]
                tangent = pts[-1] - pts[-2]
            else:
                pos = pts[idx] * (1 - t) + pts[idx+1] * t
                tangent = pts[idx+1] - pts[idx]

            # Normalize tangent
            norm = np.linalg.norm(tangent)
            if norm > 0:
                tangent = tangent / norm
            else:
                tangent = np.array([1.0, 0.0, 0.0])

            # Rotate so X-axis points along the tangent
            x_axis = np.array([1.0, 0.0, 0.0])
            dot_product = np.clip(np.dot(x_axis, tangent), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_product))
            
            transform = vtk.vtkTransform()
            transform.Identity()
            
            # Order: Translate -> Rotate -> Scale
            transform.Translate(float(pos[0]), float(pos[1]), float(pos[2]))
            
            if angle > 0.1:
                axis = np.cross(x_axis, tangent)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    transform.RotateWXYZ(angle, float(axis[0]), float(axis[1]), float(axis[2]))
                else:
                    transform.RotateWXYZ(180, 0, 1, 0)
                    
            transform.Scale(4.0, 1.5, 1.5) 
            
            actor.SetUserMatrix(transform.GetMatrix())

        self.plotter.render()

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

    def update_view(self, *_):
        requested_age = self.age_slider.value()
        age = self._nearest_age(requested_age)

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

        self.anim_timer.stop()
        self.anim_actors.clear()

        self.plotter.clear()
        self._configure_lighting()

        try:
            brain_opacity = 0.30 if show_tracts else 1.0

            for hemi in hemispheres:
                mesh = self._get_mesh(age, hemi, surface_type)
                if mesh is None:
                    continue

                if rendering == "Wireframe":
                    wire = self._get_wireframe_mesh(age, hemi, surface_type, mesh)
                    self.plotter.add_mesh(
                        wire, style="wireframe", color="black",
                        line_width=1.0, lighting=False, reset_camera=False,
                    )
                elif rendering == "Flat":
                    self.plotter.add_mesh(
                        mesh, color="#9DBBD2", smooth_shading=False,
                        opacity=brain_opacity,
                        lighting=True, ambient=0.22, diffuse=0.78,
                        specular=0.12, specular_power=18, reset_camera=False,
                    )
                else:
                    self.plotter.add_mesh(
                        mesh, color="#9DBBD2", smooth_shading=True,
                        opacity=brain_opacity,
                        lighting=True, ambient=0.20, diffuse=0.78,
                        specular=0.24, specular_power=28, reset_camera=False,
                    )

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

                        self.plotter.add_mesh(
                            tract_display,
                            scalars="cluster_id",
                            cmap="turbo",
                            show_scalar_bar=False,
                            line_width=2.5,
                            opacity=0.9,
                            render_lines_as_tubes=True,
                            lighting=False,
                            reset_camera=False,
                            name="tracts",
                        )
                        
                        if show_animate:
                            self._setup_animation(tract_display)

                except Exception as exc:
                    print(f"Tract rendering failed: {exc}")

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
