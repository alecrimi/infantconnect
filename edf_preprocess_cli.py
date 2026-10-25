"""Command-line EDF preprocessing driven by a config file (no GUI dependencies).

Examples:
  python edf_preprocess_cli.py --write-default-config > preproc.json
  python edf_preprocess_cli.py rec.edf -c preproc.json
  python edf_preprocess_cli.py data/*.edf -c preproc.toml -o out/ --overwrite

Writing EDF needs:  pip install edfio
"""
import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import mne
import numpy as np


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


def load_config(path):
    """Load options from a .json or .toml file (flat keys, see --write-default-config)."""
    path = Path(path)
    text = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:          # Python < 3.11
            import tomli as tomllib
        cfg = tomllib.loads(text.decode("utf-8"))
    elif suffix == ".json":
        cfg = json.loads(text.decode("utf-8"))
    else:
        raise ValueError(f"Unsupported config type '{suffix}' (use .json or .toml).")
    cfg = cfg.get("preprocess", cfg)         # allow an optional [preprocess] table
    unknown = sorted(set(cfg) - set(DEFAULT_OPTIONS))
    if unknown:
        raise ValueError(f"Unknown config key(s): {', '.join(unknown)}. "
                         f"Valid keys: {', '.join(DEFAULT_OPTIONS)}")
    return cfg


def _parse_override(item):
    if "=" not in item:
        raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
    k, v = item.split("=", 1)
    if k not in DEFAULT_OPTIONS:
        raise ValueError(f"Unknown option '{k}'. Valid keys: {', '.join(DEFAULT_OPTIONS)}")
    try:
        return k, json.loads(v)              # numbers, true/false
    except json.JSONDecodeError:
        return k, v                          # plain string


def _build_parser():
    ap = argparse.ArgumentParser(
        description="Preprocess EDF recordings (filter / resample / reference) from a "
                    "config file and save new EDF files.")
    ap.add_argument("files", nargs="*", help="input EDF file(s)")
    ap.add_argument("-c", "--config", help="config file (.json or .toml); "
                    "omitted = built-in defaults")
    ap.add_argument("-o", "--output", help="output directory, or output file when a single "
                    "input is given (default: next to each input)")
    ap.add_argument("--suffix", default="_preproc",
                    help="suffix added to the output file name (default: _preproc)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set lowpass=45 (repeatable)")
    ap.add_argument("--write-default-config", action="store_true",
                    help="print the default config as JSON and exit")
    ap.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    return ap


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.write_default_config:
        print(json.dumps(DEFAULT_OPTIONS, indent=2))
        return 0
    if not args.files:
        print("error: no input files given", file=sys.stderr)
        return 2

    try:
        opts = load_config(args.config) if args.config else {}
        for item in args.set:
            k, v = _parse_override(item)
            opts[k] = v
    except Exception as exc:
        print(f"Config error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    log = (lambda m: None) if args.quiet else (lambda m: print("  " + m))
    out_arg = Path(args.output) if args.output else None
    single_file_out = bool(out_arg and out_arg.suffix.lower() == ".edf")
    if single_file_out and len(args.files) != 1:
        print("error: -o FILE.edf requires exactly one input file", file=sys.stderr)
        return 2

    failures = 0
    for f in args.files:
        src = Path(f)
        if single_file_out:
            dst = out_arg
        else:
            out_dir = out_arg if out_arg else src.parent
            dst = out_dir / f"{src.stem}{args.suffix}.edf"
        try:
            if not src.is_file():
                raise FileNotFoundError(f"No such file: {src}")
            if dst.exists() and not args.overwrite:
                raise FileExistsError(f"{dst} exists (use --overwrite)")
            if not args.quiet:
                print(f"{src.name} -> {dst}")
            preprocess_edf(src, dst, opts, log=log)
        except Exception as exc:
            failures += 1
            print(f"{src}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            if not isinstance(exc, (ValueError, FileNotFoundError, FileExistsError, RuntimeError)):
                traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
