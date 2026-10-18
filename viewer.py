
from pathlib import Path
import json
import sys

import pyvista as pv
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
            max(self.available_ages),
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
        self.axes_checkbox.setChecked(True)
        controls.addWidget(self.axes_checkbox)

        # 3D view.
        self.plotter = QtInteractor(central)
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
                0.90,
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

        self.plotter.clear()

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
                        color="lightsteelblue",
                        smooth_shading=False,
                        lighting=True,
                        reset_camera=False,
                    )

                else:
                    self.plotter.add_mesh(
                        mesh,
                        color="lightsteelblue",
                        smooth_shading=True,
                        lighting=True,
                        reset_camera=False,
                    )

            if not displayed:
                raise ValueError(
                    f"No {surface_label.lower()} mesh is available "
                    f"for the {age}-month template."
                )

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

