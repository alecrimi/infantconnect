"""Configurable EDF preprocessing (filtering, resampling, referencing) with a dialog.

Usage from the main window:

    from edf_preprocess import show_edf_preprocess_dialog
    ...
    self.edf_preprocess_button.clicked.connect(
        lambda: show_edf_preprocess_dialog(self))

Writing EDF needs the `edfio` package:  pip install edfio
"""
import re
import traceback
from pathlib import Path

import mne
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QTextEdit, QVBoxLayout,
)

MONTAGE_NAMES = "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2".split()

DEFAULT_OPTIONS = {
    "keep_1020": True,        # keep only 10-20 channels, renamed to standard names
    "drop_flat": True,        # drop channels with (near) zero variance
    "crop": False, "crop_start": 0.0, "crop_end": 0.0,
    "notch": True, "notch_freq": 50.0, "notch_harmonics": True,
    "bandpass": True, "highpass": 0.5, "lowpass": 40.0, "filter_method": "fir",
    "resample": False, "resample_hz": 250.0,
    "reference": "average",   # "none" | "average" | "channels"
    "ref_channels": "",
}


def _match_1020(name):
    """Map an EDF label to a 10-20 name (same rules as the viewer)."""
    lookup = {n.lower(): n for n in MONTAGE_NAMES}
    lookup.update({"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"})
    clean = re.sub(r"^EEG[\s_-]*", "", name, flags=re.IGNORECASE)
    clean = re.sub(r"[-_\s]*(REF|LE|AV|A1|A2)$", "", clean, flags=re.IGNORECASE)
    return lookup.get(clean.strip().lower())


def preprocess_edf(in_path, out_path, opts, log=print):
    """Read `in_path`, apply the configured steps, write EDF to `out_path`.

    Step order: channel selection -> crop -> notch -> band-pass ->
    resample -> reference. Returns a short summary dict.
    """
    o = {**DEFAULT_OPTIONS, **opts}
    in_path, out_path = Path(in_path), Path(out_path)
    if in_path.resolve() == out_path.resolve():
        raise ValueError("Output file must differ from the input file.")

    raw = mne.io.read_raw_edf(str(in_path), preload=True, verbose="ERROR")
    sfreq_in = raw.info["sfreq"]
    n_in = len(raw.ch_names)
    log(f"Loaded {in_path.name}: {n_in} channels, {sfreq_in:g} Hz, "
        f"{raw.times[-1]:.1f} s")

    if not np.isfinite(raw.get_data()).all():
        raise ValueError("The recording contains NaN/Inf values; run the quality "
                         "check and fix the file before preprocessing.")

    # ---- channel selection
    if o["keep_1020"]:
        mapping, used = {}, set()
        for ch in raw.ch_names:
            tgt = _match_1020(ch)
            if tgt and tgt not in used:
                mapping[ch] = tgt
                used.add(tgt)
        if len(mapping) < 2:
            raise ValueError(f"Only {len(mapping)} channel(s) match the 10-20 montage.")
        raw.pick(list(mapping))
        raw.rename_channels(mapping)
        log(f"Kept {len(mapping)} 10-20 channels, dropped {n_in - len(mapping)} others.")

    if o["drop_flat"]:
        sd = raw.get_data().std(axis=1)
        flat = [ch for ch, s in zip(raw.ch_names, sd) if s < 1e-12]
        if flat:
            raw.drop_channels(flat)
            log(f"Dropped flat channels: {', '.join(flat)}")
    if len(raw.ch_names) == 0:
        raise ValueError("No channels left to process.")

    # ---- crop
    if o["crop"]:
        t0 = max(0.0, float(o["crop_start"]))
        t1 = float(o["crop_end"]) or raw.times[-1]
        t1 = min(t1, raw.times[-1])
        if t1 - t0 < 1.0:
            raise ValueError(f"Crop range {t0:g}-{t1:g} s is shorter than 1 s.")
        raw.crop(tmin=t0, tmax=t1)
        log(f"Cropped to {t0:g}-{t1:g} s.")

    nyq = raw.info["sfreq"] / 2.0

    # ---- notch
    if o["notch"]:
        f0 = float(o["notch_freq"])
        if f0 <= 0 or f0 >= nyq:
            raise ValueError(f"Notch frequency {f0:g} Hz must be between 0 and "
                             f"Nyquist ({nyq:g} Hz).")
        freqs = [f0]
        if o["notch_harmonics"]:
            freqs = [f0 * k for k in range(1, 20) if f0 * k < 0.95 * nyq]
        raw.notch_filter(freqs=freqs, verbose=False)
        log("Notch filter at " + ", ".join(f"{f:g}" for f in freqs) + " Hz.")

    # ---- band-pass
    if o["bandpass"]:
        hp = float(o["highpass"]) or None
        lp = float(o["lowpass"]) or None
        if hp is None and lp is None:
            raise ValueError("Band-pass is enabled but both cut-offs are 0.")
        if hp and lp and hp >= lp:
            raise ValueError(f"High-pass ({hp:g} Hz) must be below low-pass ({lp:g} Hz).")
        if lp and lp >= nyq:
            raise ValueError(f"Low-pass ({lp:g} Hz) must be below Nyquist ({nyq:g} Hz).")
        if o["filter_method"] == "iir":
            raw.filter(hp, lp, method="iir",
                       iir_params=dict(order=4, ftype="butter"), verbose=False)
            kind = "IIR Butterworth (order 4, zero-phase)"
        else:
            raw.filter(hp, lp, method="fir", phase="zero", verbose=False)
            kind = "FIR (zero-phase)"
        log(f"Band-pass {hp or 0:g}-{lp or nyq:g} Hz, {kind}.")

    # ---- resample
    if o["resample"]:
        new = float(o["resample_hz"])
        if new <= 0:
            raise ValueError("Target sampling rate must be positive.")
        lp = float(o["lowpass"]) if o["bandpass"] else 0.0
        if lp and lp >= new / 2:
            raise ValueError(f"Low-pass ({lp:g} Hz) must be below the new Nyquist "
                             f"({new / 2:g} Hz).")
        if new < raw.info["sfreq"] and not lp:
            log("Warning: downsampling without a low-pass filter risks aliasing.")
        old = raw.info["sfreq"]
        raw.resample(new, npad="auto", verbose=False)
        log(f"Resampled {old:g} Hz -> {raw.info['sfreq']:g} Hz.")

    # ---- reference
    if o["reference"] == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
        log("Re-referenced to the common average.")
    elif o["reference"] == "channels":
        wanted = [c.strip() for c in re.split(r"[,\s;]+", o["ref_channels"]) if c.strip()]
        if not wanted:
            raise ValueError("Enter at least one reference channel.")
        by_lower = {c.lower(): c for c in raw.ch_names}
        missing = [c for c in wanted if c.lower() not in by_lower]
        if missing:
            raise ValueError(f"Reference channel(s) not in data: {', '.join(missing)}. "
                             f"Available: {', '.join(raw.ch_names)}")
        refs = [by_lower[c.lower()] for c in wanted]
        raw.set_eeg_reference(refs, projection=False, verbose=False)
        log(f"Re-referenced to {', '.join(refs)}.")
    else:
        log("Reference left unchanged.")

    # ---- write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw.export(str(out_path), fmt="edf", overwrite=True, verbose=False)
    except ImportError as exc:
        raise RuntimeError("Writing EDF requires the 'edfio' package. "
                           "Install it with:  pip install edfio") from exc
    log(f"Wrote {out_path.name}: {len(raw.ch_names)} channels, "
        f"{raw.info['sfreq']:g} Hz, {raw.times[-1]:.1f} s.")
    return {"channels": len(raw.ch_names), "sfreq": raw.info["sfreq"],
            "duration": float(raw.times[-1]), "output": str(out_path)}


class EDFPreprocessDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EDF Preprocessing")
        self.resize(720, 760)
        root = QVBoxLayout(self)

        # ---- files
        files = QGroupBox("Files")
        ff = QFormLayout(files)
        self.in_edit = QLineEdit()
        in_btn = QPushButton("Browse…")
        in_btn.clicked.connect(self._pick_input)
        r = QHBoxLayout()
        r.addWidget(self.in_edit, 1)
        r.addWidget(in_btn)
        ff.addRow("Input EDF:", r)
        self.out_edit = QLineEdit()
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._pick_output)
        r = QHBoxLayout()
        r.addWidget(self.out_edit, 1)
        r.addWidget(out_btn)
        ff.addRow("Output EDF:", r)
        root.addWidget(files)

        # ---- channels
        chg = QGroupBox("Channels")
        cl = QVBoxLayout(chg)
        self.keep_1020 = QCheckBox("Keep only 10-20 EEG channels (rename to standard names)")
        self.keep_1020.setChecked(True)
        self.drop_flat = QCheckBox("Drop flat (zero-variance) channels")
        self.drop_flat.setChecked(True)
        cl.addWidget(self.keep_1020)
        cl.addWidget(self.drop_flat)
        root.addWidget(chg)

        # ---- crop
        self.crop_box = QGroupBox("Crop")
        self.crop_box.setCheckable(True)
        self.crop_box.setChecked(False)
        cf = QFormLayout(self.crop_box)
        self.crop_start = self._spin(0, 1e6, 0.0, 1, " s")
        self.crop_end = self._spin(0, 1e6, 0.0, 1, " s")
        self.crop_end.setToolTip("0 = until the end of the recording")
        cf.addRow("Start:", self.crop_start)
        cf.addRow("End (0 = end):", self.crop_end)
        root.addWidget(self.crop_box)

        # ---- notch
        self.notch_box = QGroupBox("Notch filter (mains)")
        self.notch_box.setCheckable(True)
        self.notch_box.setChecked(True)
        nf = QFormLayout(self.notch_box)
        self.notch_freq = QComboBox()
        self.notch_freq.setEditable(True)
        self.notch_freq.addItems(["50", "60"])
        self.notch_harm = QCheckBox("Include harmonics")
        self.notch_harm.setChecked(True)
        nf.addRow("Frequency (Hz):", self.notch_freq)
        nf.addRow("", self.notch_harm)
        root.addWidget(self.notch_box)

        # ---- band-pass
        self.bp_box = QGroupBox("Band-pass filter")
        self.bp_box.setCheckable(True)
        self.bp_box.setChecked(True)
        bf = QFormLayout(self.bp_box)
        self.hp = self._spin(0, 1000, 0.5, 2, " Hz")
        self.lp = self._spin(0, 5000, 40.0, 1, " Hz")
        self.hp.setToolTip("0 = no high-pass")
        self.lp.setToolTip("0 = no low-pass")
        self.method = QComboBox()
        self.method.addItems(["FIR (zero-phase)", "IIR Butterworth (zero-phase)"])
        bf.addRow("High-pass (0 = off):", self.hp)
        bf.addRow("Low-pass (0 = off):", self.lp)
        bf.addRow("Method:", self.method)
        root.addWidget(self.bp_box)

        # ---- resample
        self.rs_box = QGroupBox("Resample")
        self.rs_box.setCheckable(True)
        self.rs_box.setChecked(False)
        rf = QFormLayout(self.rs_box)
        self.rs_hz = self._spin(1, 10000, 250.0, 1, " Hz")
        rf.addRow("Target rate:", self.rs_hz)
        root.addWidget(self.rs_box)

        # ---- reference
        refg = QGroupBox("Reference")
        rl = QFormLayout(refg)
        self.ref_combo = QComboBox()
        self.ref_combo.addItems(["Average reference", "Channel(s)", "Keep original"])
        self.ref_edit = QLineEdit()
        self.ref_edit.setPlaceholderText("e.g. Cz   or   T7, T8")
        self.ref_edit.setEnabled(False)
        self.ref_combo.currentIndexChanged.connect(
            lambda i: self.ref_edit.setEnabled(i == 1))
        rl.addRow("Type:", self.ref_combo)
        rl.addRow("Channels:", self.ref_edit)
        root.addWidget(refg)

        root.addWidget(QLabel(
            "Order: channels → crop → notch → band-pass → resample → reference."))

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(110)
        root.addWidget(self.log_view, 1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.run_btn = QPushButton("Run preprocessing")
        self.run_btn.clicked.connect(self._run)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(self.run_btn)
        btns.addWidget(close_btn)
        root.addLayout(btns)

    @staticmethod
    def _spin(lo, hi, val, dec, suffix):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def _pick_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select EDF to Preprocess", self.in_edit.text() or str(Path.home()),
            "EDF Files (*.edf);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            self.in_edit.setText(path)
            p = Path(path)
            self.out_edit.setText(str(p.with_name(p.stem + "_preproc.edf")))

    def _pick_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preprocessed EDF", self.out_edit.text() or str(Path.home()),
            "EDF Files (*.edf);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            if not path.lower().endswith(".edf"):
                path += ".edf"
            self.out_edit.setText(path)

    def _options(self):
        try:
            notch_f = float(self.notch_freq.currentText())
        except ValueError:
            notch_f = 0.0
        return {
            "keep_1020": self.keep_1020.isChecked(),
            "drop_flat": self.drop_flat.isChecked(),
            "crop": self.crop_box.isChecked(),
            "crop_start": self.crop_start.value(),
            "crop_end": self.crop_end.value(),
            "notch": self.notch_box.isChecked(),
            "notch_freq": notch_f,
            "notch_harmonics": self.notch_harm.isChecked(),
            "bandpass": self.bp_box.isChecked(),
            "highpass": self.hp.value(),
            "lowpass": self.lp.value(),
            "filter_method": "iir" if self.method.currentIndex() == 1 else "fir",
            "resample": self.rs_box.isChecked(),
            "resample_hz": self.rs_hz.value(),
            "reference": ("average", "channels", "none")[self.ref_combo.currentIndex()],
            "ref_channels": self.ref_edit.text(),
        }

    def _log(self, msg):
        self.log_view.append(msg)
        QApplication.processEvents()

    def _run(self):
        in_path = self.in_edit.text().strip()
        out_path = self.out_edit.text().strip()
        if not in_path or not Path(in_path).is_file():
            QMessageBox.warning(self, "EDF Preprocessing", "Select a valid input EDF file.")
            return
        if not out_path:
            QMessageBox.warning(self, "EDF Preprocessing", "Choose an output file.")
            return
        if Path(out_path).exists():
            ans = QMessageBox.question(
                self, "Overwrite?", f"{Path(out_path).name} exists. Overwrite it?")
            if ans != QMessageBox.StandardButton.Yes:
                return

        self.log_view.clear()
        self.run_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            res = preprocess_edf(in_path, out_path, self._options(), log=self._log)
            self._log("\nDone.")
            QApplication.restoreOverrideCursor()
            QMessageBox.information(
                self, "EDF Preprocessing",
                f"Saved:\n{res['output']}\n\n{res['channels']} channels, "
                f"{res['sfreq']:g} Hz, {res['duration']:.1f} s")
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            traceback.print_exc()
            self._log(f"\nERROR: {type(exc).__name__}: {exc}")
            QMessageBox.critical(self, "EDF Preprocessing",
                                 f"{type(exc).__name__}: {exc}")
        finally:
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            self.run_btn.setEnabled(True)


def show_edf_preprocess_dialog(parent=None):
    EDFPreprocessDialog(parent).exec()
