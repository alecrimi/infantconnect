"""Stacked EEG time-series panel (one labelled trace per sensor).

Used by InfantBrainViewer to show the EDF that was loaded for static LORETA
next to the 3D view. Uses the Matplotlib Qt backend that ships with MNE's
plotting dependencies.
"""

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QPushButton,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

BG = "#202833"
FG = "#E0E0E0"
GRID = "#3A4656"
TRACE = "#7FDBFF"
SEL = "#FF9F1C"        # span being selected
APPLIED = "#4CD97B"    # span the current source map is computed from

WINDOW_CHOICES = (5, 10, 20, 30, 60)   # seconds shown at once


class EEGTracePanel(QWidget):
    # (t_start, t_end) in seconds: user wants LORETA recomputed on that span.
    refineRequested = Signal(float, float)
    # User wants LORETA on the whole recording again.
    resetRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data_uv = None      # (n_ch, n_times) in microvolts
        self._times = None
        self._names = []
        self._sfreq = 1.0
        self._window_s = 10.0
        self._scale_uv = 50.0     # spacing between traces (uV)
        self._sel = None          # pending selection (t0, t1) in seconds
        self._applied = None      # span the displayed source map uses
        self._selector = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title = QLabel("EEG time series")
        self.title.setStyleSheet("color: #9FB3C8; font-weight: bold; padding: 2px 4px;")
        layout.addWidget(self.title)

        self.figure = Figure(facecolor=BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumWidth(320)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, stretch=1)

        row = QHBoxLayout()
        layout.addLayout(row)

        row.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        for s in WINDOW_CHOICES:
            self.window_combo.addItem(f"{s} s", s)
        self.window_combo.setCurrentIndex(WINDOW_CHOICES.index(10))
        row.addWidget(self.window_combo)

        row.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox()
        for v in (10, 20, 50, 100, 200, 500):
            self.scale_combo.addItem(f"{v} µV", v)
        self.scale_combo.setCurrentIndex(2)
        row.addWidget(self.scale_combo)
        row.addStretch(1)

        self.scroll = QSlider(Qt.Orientation.Horizontal)
        self.scroll.setRange(0, 0)
        layout.addWidget(self.scroll)

        self.sel_label = QLabel("")
        self.sel_label.setWordWrap(True)
        self.sel_label.setStyleSheet("color: #9FB3C8; padding: 2px 4px;")
        layout.addWidget(self.sel_label)

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        self.refine_btn = QPushButton("Refine LORETA on selection")
        self.refine_btn.setToolTip("Recompute and plot LORETA for the highlighted time span")
        self.refine_btn.setEnabled(False)
        btn_row.addWidget(self.refine_btn)
        self.full_btn = QPushButton("Use full recording")
        self.full_btn.setEnabled(False)
        btn_row.addWidget(self.full_btn)
        self.refine_btn.clicked.connect(self._emit_refine)
        self.full_btn.clicked.connect(self._emit_reset)
        self._update_labels()

        self.window_combo.currentIndexChanged.connect(self._on_window_changed)
        self.scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        self.scroll.valueChanged.connect(self._redraw)

        self._style_axes()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ API
    def set_raw(self, raw, title=None):
        """Display an mne.io.Raw (already picked / referenced)."""
        data = raw.get_data() * 1e6                      # V -> uV
        self._data_uv = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        self._times = raw.times
        self._names = list(raw.ch_names)
        self._sfreq = float(raw.info["sfreq"])
        self._sel = None
        self._applied = None
        self._update_labels()
        if title:
            self.title.setText(title)
        self._update_scroll_range()
        self._redraw()

    def clear(self):
        self._data_uv = None
        self._names = []
        self._sel = None
        self._applied = None
        self._update_labels()
        self.title.setText("EEG time series")
        self._style_axes()
        self.canvas.draw_idle()

    def set_applied_window(self, window):
        """Mark the span the displayed source map is based on (None = whole)."""
        self._applied = window
        self._update_labels()
        self._redraw()

    # ------------------------------------------------------------- internals
    def _update_labels(self):
        self.refine_btn.setEnabled(self._sel is not None)
        self.full_btn.setEnabled(self._applied is not None)
        parts = []
        if self._applied is None:
            parts.append("Source map: full recording.")
        else:
            a0, a1 = self._applied
            parts.append(f"Source map: {a0:.2f}–{a1:.2f} s (green).")
        if self._sel is None:
            parts.append("Drag on the traces to select a time range.")
        else:
            s0, s1 = self._sel
            parts.append(f"Selected: {s0:.2f}–{s1:.2f} s ({s1 - s0:.2f} s).")
        self.sel_label.setText(" ".join(parts))

    def _on_select(self, xmin, xmax):
        if self._data_uv is None:
            return
        total = self._data_uv.shape[1] / self._sfreq
        xmin, xmax = max(0.0, float(xmin)), min(total, float(xmax))
        if xmax - xmin < 4.0 / self._sfreq:      # a click, not a drag
            return
        self._sel = (xmin, xmax)
        self._update_labels()

    def _emit_refine(self):
        if self._sel is not None:
            self.refineRequested.emit(*self._sel)

    def _emit_reset(self):
        self._sel = None
        self.resetRequested.emit()
        self.set_applied_window(None)

    def _make_selector(self):
        if self._selector is not None:
            try:
                self._selector.disconnect_events()
            except Exception:
                pass
        self._selector = SpanSelector(
            self.ax, self._on_select, "horizontal", useblit=False,
            interactive=True, drag_from_anywhere=True,
            props=dict(facecolor=SEL, alpha=0.25),
            handle_props=dict(color=SEL),
        )

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=FG, labelsize=9)

    def _on_window_changed(self, *_):
        self._window_s = float(self.window_combo.currentData())
        self._update_scroll_range()
        self._redraw()

    def _on_scale_changed(self, *_):
        self._scale_uv = float(self.scale_combo.currentData())
        self._redraw()

    def _update_scroll_range(self):
        if self._data_uv is None:
            self.scroll.setRange(0, 0)
            return
        total = self._data_uv.shape[1] / self._sfreq
        max_start_ds = max(0, int((total - self._window_s) * 10))  # 0.1 s steps
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, max_start_ds)
        self.scroll.setValue(min(self.scroll.value(), max_start_ds))
        self.scroll.blockSignals(False)

    def _redraw(self, *_):
        ax = self.ax
        ax.clear()
        self._style_axes()
        if self._data_uv is None:
            self.canvas.draw_idle()
            return

        n_ch, n_t = self._data_uv.shape
        t0 = self.scroll.value() / 10.0
        i0 = int(t0 * self._sfreq)
        i1 = min(n_t, i0 + int(self._window_s * self._sfreq))
        if i1 - i0 < 2:
            self.canvas.draw_idle()
            return

        seg = self._data_uv[:, i0:i1]
        seg = seg - seg.mean(axis=1, keepdims=True)          # remove DC per channel
        t = self._times[i0:i1]

        # Channel 0 at the top.
        offsets = (n_ch - 1 - np.arange(n_ch)) * self._scale_uv
        for k in range(n_ch):
            ax.plot(t, seg[k] + offsets[k], color=TRACE, lw=0.7)

        ax.set_yticks(offsets)
        ax.set_yticklabels(self._names, color=FG)
        ax.set_ylim(-self._scale_uv, n_ch * self._scale_uv)
        ax.set_xlim(t[0], t[0] + self._window_s)
        ax.set_xlabel("Time (s)", color=FG)
        ax.grid(axis="x", color=GRID, lw=0.5)
        if self._applied is not None:
            ax.axvspan(*self._applied, facecolor=APPLIED, alpha=0.12, lw=0)
            ax.axvline(self._applied[0], color=APPLIED, lw=1)
            ax.axvline(self._applied[1], color=APPLIED, lw=1)
        if self._sel is not None:
            ax.axvspan(*self._sel, facecolor=SEL, alpha=0.18, lw=0)
        self._make_selector()

        # Scale bar (one trace spacing).
        x_bar = t[0] + 0.02 * self._window_s
        ax.plot([x_bar, x_bar], [-0.8 * self._scale_uv, 0.2 * self._scale_uv],
                color="#FFD700", lw=2)
        ax.text(x_bar, -0.9 * self._scale_uv, f"{self._scale_uv:g} µV",
                color="#FFD700", fontsize=8, va="top")

        self.figure.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.10)
        self.canvas.draw_idle()
