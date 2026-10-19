
from pathlib import Path
import json
import sys

import pyvista as pv
import mne
import numpy as np
from pyvistaqt import QtInteractor

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QComboBox,
    QMessageBox,
    QCheckBox,
)


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "atlas.json"

# The viewer can be used beyond the age range covered by the atlas.
# Above the last available age, the last cortical template is reused.
DISPLAY_MAX_AGE_MONTHS = 72  # 10 years; change if you want a longer slider
MNE_SUBJECTS_DIR = Path.home() / "Documents" / "infant_viewer" / "mne_templates"

SURFACE_TYPES = {
    "Outer surface": "outer",
    "Middle surface": "middle",
    "Inner surface": "inner",
    "Inflated surface": "inflated",
}


def load_surface(relative_path):
    """Load and validate a VTK cortical mesh."""
    path = ROOT / relative_path

    if not path.is_file():
        raise FileNotFoundError(f"Mesh not found:\n{path}")

    mesh = pv.read(str(path))

    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Empty surface mesh:\n{path}")

    return mesh


def load_mne_infant_scalp():
    """Load the 2-year MNE infant scalp and its 10-20 electrode positions."""
    MNE_SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    subject = mne.datasets.fetch_infant_template(
        "2yr", subjects_dir=str(MNE_SUBJECTS_DIR), verbose=True
    )
    subject_dir = MNE_SUBJECTS_DIR / subject
    bem_dir = subject_dir / "bem"

    scalp_files = list(bem_dir.glob("*outer_skin.surf"))
    if not scalp_files:
        scalp_files = list(bem_dir.glob("*head.surf"))
    if not scalp_files:
        raise FileNotFoundError(f"No outer scalp surface found in {bem_dir}")

    vertices, faces = mne.read_surface(str(scalp_files[0]))
    faces_pv = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    scalp = pv.PolyData(vertices, faces_pv)

    montage_file = subject_dir / "montages" / "10-20-montage.fif"
    montage = mne.channels.read_dig_fif(montage_file, verbose="error")
    names = "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2".split()
    montage.rename_channels({
        f"EEG{i:03d}": name for i, name in enumerate(names, start=1)
    })
    positions = montage.get_positions()["ch_pos"]
    
    '''
    electrode_positions = {
        name: np.asarray(pos, dtype=float)
        for name, pos in positions.items()
        if name in names
    }
    '''
    
    # Montage is in metres, surface is in mm -> convert, then snap to the scalp.
    electrode_positions = {}
    for name, pos in positions.items():
        if name not in names:
            continue
        p_mm = np.asarray(pos, dtype=float) * 1000.0
        idx = scalp.find_closest_point(p_mm)
        electrode_positions[name] = np.asarray(scalp.points[idx], dtype=float)
        print(f"{name}: snap distance {np.linalg.norm(scalp.points[idx] - p_mm):.1f} mm")
    return scalp, electrode_positions
 
'''
def fit_template_to_brain(scalp, electrode_positions, reference_mesh):
    """
    Uniformly scale and translate the MNE scalp/electrodes for visual overlay.

    This is a display-space fit, not anatomical registration. It is useful when
    the atlas meshes and MNE template do not share the same coordinate origin.
    """
    scalp = scalp.copy()
    ref_center = np.asarray(reference_mesh.center, dtype=float)
    ref_lengths = np.asarray(reference_mesh.bounds)[[1, 3, 5]] - np.asarray(reference_mesh.bounds)[[0, 2, 4]]
    scalp_lengths = np.asarray(scalp.bounds)[[1, 3, 5]] - np.asarray(scalp.bounds)[[0, 2, 4]]
    valid = scalp_lengths > 0
    scale = 1.12 * np.min(ref_lengths[valid] / scalp_lengths[valid])
    scalp_center = np.asarray(scalp.center, dtype=float)
    scalp.points = (scalp.points - scalp_center) * scale + ref_center

    fitted_positions = {
        name: (pos - scalp_center) * scale + ref_center
        for name, pos in electrode_positions.items()
    }
    return scalp, fitted_positions
'''
def fit_template_to_brain(scalp, electrode_positions, reference_mesh,
                          margin=(1.12, 1.10), clearance=0.03):
    """
    Display-space fit using the 10-20 cap. Scale x and y separately so the cap
    encloses the brain with a small margin, use their mean for z, centre in
    x/y, and put Cz just above the top of the brain.
    """
    scalp = scalp.copy()
    E = np.vstack(list(electrode_positions.values()))
    e_lo, e_hi = E.min(axis=0), E.max(axis=0)
    b = np.asarray(reference_mesh.bounds)
    b_lo, b_hi = b[[0, 2, 4]], b[[1, 3, 5]]

    sx = margin[0] * (b_hi[0] - b_lo[0]) / (e_hi[0] - e_lo[0])
    sy = margin[1] * (b_hi[1] - b_lo[1]) / (e_hi[1] - e_lo[1])
    scale = np.array([sx, sy, 0.5 * (sx + sy)])

    src = np.array([(e_lo[0] + e_hi[0]) / 2, (e_lo[1] + e_hi[1]) / 2, e_hi[2]])
    dst = np.array([(b_lo[0] + b_hi[0]) / 2, (b_lo[1] + b_hi[1]) / 2,
                    b_hi[2] + clearance * (b_hi[2] - b_lo[2])])

    def f(p):
        return (np.asarray(p, dtype=float) - src) * scale + dst

    scalp.points = f(scalp.points)
    return scalp, {name: f(pos) for name, pos in electrode_positions.items()}

class InfantBrainViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Infant Brain 3D Visualizer")
        self.resize(1400, 900)

        if not MANIFEST.is_file():
            raise FileNotFoundError(
                f"Manifest not found:\n{MANIFEST}\n"
                "Run generate_manifest.py first."
            )

        with MANIFEST.open("r", encoding="utf-8") as f:
            self.atlas = json.load(f)

        # Find ages with at least one outer surface.
        self.available_ages = sorted(
            int(age)
            for age, entry in self.atlas.items()
            if any(
                entry.get(hemi, {}).get("outer")
                for hemi in ("left", "right")
            )
        )

        if not self.available_ages:
            raise RuntimeError(
                "No usable outer surfaces found in atlas.json."
            )

        self.mesh_cache = {}
        self.wireframe_cache = {}
        self.electrode_template_error = None
        try:
            self.scalp_template, self.electrode_positions = load_mne_infant_scalp()
        except Exception as exc:
            # Keep the brain viewer usable even if the template cannot be fetched.
            self.scalp_template = None
            self.electrode_positions = {}
            self.electrode_template_error = str(exc)

        self._build_ui()
        self._connect_signals()
        self.update_view()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        # Age slider: continuous months, snapped to the nearest template.
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
        self.age_slider.setTickPosition(
            QSlider.TickPosition.TicksBelow
        )
        self.age_slider.setTickInterval(6)

        initial_age = min(
            self.available_ages,
            key=lambda a: abs(a - 12),
        )
        self.age_slider.setValue(initial_age)

        age_row.addWidget(self.age_slider, stretch=1)

        self.age_label = QLabel()
        self.age_label.setMinimumWidth(240)
        age_row.addWidget(self.age_label)

        # Display controls.
        controls = QHBoxLayout()
        layout.addLayout(controls)

        controls.addWidget(QLabel("Hemisphere:"))

        self.hemisphere_combo = QComboBox()
        self.hemisphere_combo.addItems(
            ["Both", "Left", "Right"]
        )
        controls.addWidget(self.hemisphere_combo)

        controls.addWidget(QLabel("Surface:"))

        self.surface_combo = QComboBox()
        self.surface_combo.addItems(SURFACE_TYPES.keys())
        controls.addWidget(self.surface_combo)

        controls.addWidget(QLabel("Rendering:"))

        self.render_combo = QComboBox()
        self.render_combo.addItems(
            ["Smooth", "Flat", "Wireframe"]
        )
        controls.addWidget(self.render_combo)

        self.axes_checkbox = QCheckBox("Show 3D axes")
        self.axes_checkbox.setChecked(False)
        controls.addWidget(self.axes_checkbox)

        self.electrodes_checkbox = QCheckBox("Show 2-year scalp + electrodes")
        self.electrodes_checkbox.setChecked(True)
        controls.addWidget(self.electrodes_checkbox)

        # 3D view.
        self.plotter = QtInteractor(central)
        self.plotter.set_background("#202833")
        layout.addWidget(self.plotter.interactor, stretch=1)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _connect_signals(self):
        # With tracking disabled, slider changes are emitted on release.
        self.age_slider.valueChanged.connect(self.update_view)

        self.hemisphere_combo.currentIndexChanged.connect(
            self.update_view
        )
        self.surface_combo.currentIndexChanged.connect(
            self.update_view
        )
        self.render_combo.currentIndexChanged.connect(
            self.update_view
        )
        self.axes_checkbox.toggled.connect(self.update_view)
        self.electrodes_checkbox.toggled.connect(self.update_view)

    def _nearest_age(self, requested_age):
        return min(
            self.available_ages,
            key=lambda age: abs(age - requested_age),
        )

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

    def _get_wireframe_mesh(
        self, age, hemisphere, surface_type, mesh
    ):
        """Return a cached, simplified mesh for wireframe rendering."""
        key = (age, hemisphere, surface_type)

        if key in self.wireframe_cache:
            return self.wireframe_cache[key]

        # Try to remove 90% of the triangles.
        # Fall back to the original mesh if decimation fails.
        try:
            simplified = mesh.decimate(
                0.99,
                volume_preservation=True,
            )

            if (
                simplified.n_points == 0
                or simplified.n_cells == 0
            ):
                simplified = mesh

        except Exception:
            simplified = mesh

        self.wireframe_cache[key] = simplified
        return simplified

    def _configure_lighting(self):
        """Install a cheap, soft three-point lighting setup."""
        # clear() may reset scene props, so explicitly replace the lights
        # on each redraw instead of allowing them to accumulate.
        try:
            self.plotter.remove_all_lights()
        except (AttributeError, TypeError):
            pass

        # A broad, soft key light reveals the cortical folds.
        key = pv.Light(
            position=(1.0, 1.5, 2.0),
            focal_point=(0.0, 0.0, 0.0),
            color="white",
            intensity=0.75,
        )
        # A weaker fill light keeps the opposite hemisphere from going dark.
        fill = pv.Light(
            position=(-1.5, 0.5, 0.8),
            focal_point=(0.0, 0.0, 0.0),
            color=(0.78, 0.84, 1.0),
            intensity=0.30,
        )
        self.plotter.add_light(key)
        self.plotter.add_light(fill)

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

        self.plotter.clear()
        self._configure_lighting()

        total_points = 0
        displayed = []
        missing = []

        try:
            for hemi in hemispheres:
                mesh = self._get_mesh(
                    age, hemi, surface_type
                )

                if mesh is None:
                    missing.append(hemi)
                    continue

                displayed.append(hemi)
                total_points += mesh.n_points

                if rendering == "Wireframe":
                    wire_mesh = self._get_wireframe_mesh(
                        age, hemi, surface_type, mesh
                    )

                    self.plotter.add_mesh(
                        wire_mesh,
                        style="wireframe",
                        color="black",
                        line_width=1.0,
                        lighting=False,
                        reset_camera=False,
                    )

                elif rendering == "Flat":
                    self.plotter.add_mesh(
                        mesh,
                        color="#9DBBD2",
                        smooth_shading=False,
                        lighting=True,
                        ambient=0.22,
                        diffuse=0.78,
                        specular=0.12,
                        specular_power=18,
                        reset_camera=False,
                    )

                else:
                    self.plotter.add_mesh(
                        mesh,
                        color="#9DBBD2",
                        smooth_shading=True,
                        lighting=True,
                        ambient=0.20,
                        diffuse=0.78,
                        specular=0.24,
                        specular_power=28,
                        reset_camera=False,
                    )

            if not displayed:
                raise ValueError(
                    f"No {surface_label.lower()} mesh is available "
                    f"for the {age}-month template."
                )

            # Overlay the MNE 2-year scalp and electrode montage. The fitting
            # uses both hemispheres so the scalp remains correctly sized when
            # the user chooses to display only one hemisphere.
            if show_electrodes and self.scalp_template is not None:
                reference_parts = []
                for hemi_ref in ("left", "right"):
                    ref_mesh = self._get_mesh(age, hemi_ref, "outer")
                    if ref_mesh is not None:
                        reference_parts.append(ref_mesh)
                if reference_parts:
                    reference = reference_parts[0] if len(reference_parts) == 1 else reference_parts[0].merge(reference_parts[1:])
                    scalp_fit, electrodes_fit = fit_template_to_brain(
                        self.scalp_template, self.electrode_positions, reference
                    )
                    self.plotter.add_mesh(
                        scalp_fit, color="#E8C9A8", opacity=0.3,
                        smooth_shading=True, lighting=True, reset_camera=False,
                    )
                    if electrodes_fit:
                        # Use ONE combined mesh for all markers and ONE label
                        # actor for all names. Repeated add_point_labels calls
                        # can replace the previous label actor, leaving only O2.
                        names = list(electrodes_fit.keys())
                        pts = np.vstack([electrodes_fit[name] for name in names])

                        # Offset points slightly outside the fitted scalp.
                        scalp_center = np.asarray(scalp_fit.center, dtype=float)
                        radial = pts - scalp_center
                        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
                        radial_norm[radial_norm < 1e-12] = 1.0
                        scalp_scale = float(np.linalg.norm(np.asarray(scalp_fit.bounds)[[1, 3, 5]] - np.asarray(scalp_fit.bounds)[[0, 2, 4]]))
                        pts = pts + 0.025 * scalp_scale * radial / radial_norm

                        # Merge explicit sphere meshes into one PolyData actor.
                        marker_radius = max(scalp_scale * 0.022, 0.004)
                        label_pts = pts + (radial / radial_norm) * marker_radius * 1.5
                        sphere_meshes = [
                            pv.Sphere(
                                radius=marker_radius,
                                center=tuple(float(v) for v in point),
                                theta_resolution=16,
                                phi_resolution=16,
                            )
                            for point in pts
                        ]
                        markers = sphere_meshes[0]
                        for sphere in sphere_meshes[1:]:
                            markers = markers.merge(sphere, merge_points=False)

                        self.plotter.add_mesh(
                            markers,
                            color="#FF4D00",
                            ambient=1.0,
                            diffuse=0.0,
                            specular=0.0,
                            lighting=False,
                            reset_camera=False,
                            name="electrode_markers",
                        )
                        self.plotter.add_point_labels(
                            label_pts,
                            names,
                            name="electrode_labels",
                            point_size=0,
                            font_size=14,
                            text_color="#FFFFFF",
                            shape="rounded_rect",
                            fill_shape=True,
                            shape_color="#202833",
                            shape_opacity=0.9,
                            always_visible=False,
                            tolerance=0.002,
                            show_points=False,
                            render_points_as_spheres=False,
                            reset_camera=False,
                        )
                        print(f"Rendered {len(names)} electrode markers: {names}")
                        print("Fitted electrode bounds:", np.round(np.c_[pts.min(axis=0), pts.max(axis=0)], 4))

            if show_axes:
                self.plotter.show_bounds(
                    grid="front",
                    location="outer",
                    all_edges=True,
                    xtitle="X",
                    ytitle="Y",
                    ztitle="Z",
                    fmt="%.3f",
                    color="gray",
                    font_size=10,
                )

            self.plotter.reset_camera()
            self.plotter.render()

            self.age_label.setText(
                f"Requested: {requested_age} mo | "
                f"Template: {age} mo"
            )

            status = (
                f"{surface_label} | "
                f"Hemisphere: {', '.join(displayed)} | "
                f"Vertices: {total_points:,} | "
                f"Rendering: {rendering}"
            )

            if missing:
                status += " | Missing: " + ", ".join(missing)

            if age != requested_age:
                status += (
                    f" | Nearest available template: {age} months"
                )

            if show_electrodes:
                if self.scalp_template is None:
                    status += " | MNE scalp/electrodes unavailable"
                    if self.electrode_template_error:
                        status += f": {self.electrode_template_error}"
                else:
                    status += " | Overlay: 2-year MNE scalp/electrodes (display-space fit)"
            self.status_label.setText(status)

        except Exception as exc:
            self.status_label.setText(
                f"Display error: {exc}"
            )

    def closeEvent(self, event):
        self.plotter.close()
        event.accept()


def main():
    app = QApplication(sys.argv)

    try:
        window = InfantBrainViewer()
        window.show()
        sys.exit(app.exec())

    except Exception as exc:
        QMessageBox.critical(
            None, "Infant Brain Viewer", str(exc)
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
