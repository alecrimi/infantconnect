#!/usr/bin/env python3
"""
Command-line effective-connectivity analysis from EDF recordings.

Methods
-------
Granger causality:
    Uses effconn.compute_granger().
    Values are -log10(p); values >= 1.3 correspond approximately to p < 0.05.

Transfer entropy:
    Uses effconn.compute_transfer_entropy().
    Values are in nats.

Matrix convention
----------------
The returned matrix M follows effconn.py:

    M[target, source] = influence from source -> target

Use --source-rows to transpose the saved matrix.

Outputs
-------
By default, saves:
    <output_dir>/<recording>_granger.csv
    <output_dir>/<recording>_transfer_entropy.csv

With --edge-list, also saves:
    <output_dir>/<recording>_<method>_edges.csv

Requires:
    pip install mne numpy scipy

The effconn.py module must be in the same directory or importable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mne

from effconn import (
    compute_granger,
    compute_transfer_entropy,
    compute_effective_connectivity,
)


# ---------------------------------------------------------------------
# EDF loading
# ---------------------------------------------------------------------

def load_edf_data(edf_path: Path):
    """
    Load an EDF file and return:
        data: ndarray, shape (n_samples, n_channels)
        channel_names: list[str]
        sampling_frequency: float
        duration_seconds: float
    """
    raw = mne.io.read_raw_edf(
        str(edf_path),
        preload=True,
        verbose="ERROR",
    )

    data = raw.get_data().T.astype(float)

    channel_names = list(raw.ch_names)
    sampling_frequency = float(raw.info["sfreq"])
    duration_seconds = float(raw.n_times / sampling_frequency)

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D EEG array, got shape {data.shape}"
        )

    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("EDF contains no samples or no channels.")

    if not np.isfinite(data).all():
        print(
            "Warning: non-finite values detected in EDF data. "
            "Replacing them with zero.",
            file=sys.stderr,
        )
        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    return (
        data,
        channel_names,
        sampling_frequency,
        duration_seconds,
    )


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def save_matrix(
    matrix: np.ndarray,
    channel_names: list[str],
    output_path: Path,
    source_rows: bool = False,
):
    """
    Save a connectivity matrix to CSV.

    Default:
        row = target, column = source

    With source_rows:
        row = source, column = target
    """
    matrix_to_save = matrix.T if source_rows else matrix

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a labelled CSV so that channel identities are preserved.
    with output_path.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(channel_names) + "\n")

        for channel_name, row in zip(channel_names, matrix_to_save):
            values = ",".join(f"{float(value):.10g}" for value in row)
            f.write(f"{channel_name},{values}\n")


def save_edge_list(
    matrix: np.ndarray,
    channel_names: list[str],
    output_path: Path,
    min_strength: float = 0.0,
):
    """
    Save directed edges as a CSV.

    Matrix convention:
        matrix[target, source] = source -> target

    Edges are included when their strength is >= min_strength.
    Self-connections are excluded.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("source,target,strength\n")

        n_channels = matrix.shape[0]

        for target_idx in range(n_channels):
            for source_idx in range(n_channels):
                if source_idx == target_idx:
                    continue

                strength = float(matrix[target_idx, source_idx])

                if strength >= min_strength:
                    f.write(
                        f"{channel_names[source_idx]},"
                        f"{channel_names[target_idx]},"
                        f"{strength:.10g}\n"
                    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser(fixed_method: str | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute Granger causality or transfer entropy "
            "from an EDF recording."
        )
    )

    parser.add_argument(
        "edf",
        type=Path,
        help="Input EDF file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out"),
        help="Output directory. Default: out/",
    )

    if fixed_method is None:
        parser.add_argument(
            "--method",
            choices=["granger", "transfer_entropy"],
            default="granger",
            help="Effective-connectivity method.",
        )

    parser.add_argument(
        "--lag",
        type=int,
        default=5,
        help="Lag for Granger causality. Default: 5.",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=4,
        help="Number of bins for transfer entropy. Default: 4.",
    )

    parser.add_argument(
        "--edge-list",
        action="store_true",
        help="Also save a directed edge-list CSV.",
    )

    parser.add_argument(
        "--min-strength",
        type=float,
        default=0.0,
        help=(
            "Minimum edge strength for the edge-list output. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--source-rows",
        action="store_true",
        help=(
            "Transpose the saved matrix so rows are sources "
            "and columns are targets."
        ),
    )

    return parser


def main(fixed_method: str | None = None):
    parser = build_parser(fixed_method=fixed_method)
    args = parser.parse_args()

    if not args.edf.exists():
        parser.error(f"EDF file not found: {args.edf}")

    if args.edf.suffix.lower() != ".edf":
        parser.error("Input file must have an .edf extension.")

    if args.lag < 1:
        parser.error("--lag must be >= 1.")

    if args.bins < 2:
        parser.error("--bins must be >= 2.")

    if args.min_strength < 0:
        parser.error("--min-strength must be >= 0.")

    method = fixed_method or args.method

    if method not in ("granger", "transfer_entropy"):
        parser.error(f"Unsupported method: {method}")

    print(f"Input EDF: {args.edf}")
    print(f"Method: {method}")

    # Load EDF
    data, channel_names, sfreq, duration_seconds = load_edf_data(
        args.edf
    )

    print(f"Channels: {len(channel_names)}")
    print(f"Samples: {data.shape[0]}")
    print(f"Sampling frequency: {sfreq:g} Hz")
    print(f"Duration: {duration_seconds:.2f} seconds")

    # Compute connectivity
    if method == "granger":
        matrix = compute_granger(
            data,
            lag=args.lag,
        )
        method_suffix = "granger"

        print(f"Granger lag: {args.lag}")

    else:
        matrix = compute_transfer_entropy(
            data,
            bins=args.bins,
            lag=args.lag,
        )
        method_suffix = "transfer_entropy"

        print(f"TE bins: {args.bins}")
        print(f"TE lag: {args.lag}")

    # Output filenames
    recording_stem = args.edf.stem

    matrix_path = (
        args.output
        / f"{recording_stem}_{method_suffix}.csv"
    )

    save_matrix(
        matrix=matrix,
        channel_names=channel_names,
        output_path=matrix_path,
        source_rows=args.source_rows,
    )

    print(f"Connectivity matrix saved: {matrix_path}")

    if args.edge_list:
        edge_path = (
            args.output
            / f"{recording_stem}_{method_suffix}_edges.csv"
        )

        save_edge_list(
            matrix=matrix,
            channel_names=channel_names,
            output_path=edge_path,
            min_strength=args.min_strength,
        )

        print(f"Edge list saved: {edge_path}")
        print(f"Minimum edge strength: {args.min_strength}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
