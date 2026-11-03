#!/usr/bin/env python3
"""
Benchmark InfantConnect EDF command-line tools on every EDF recording
in EDF_format.

Tools:
    edf_quality_cli.py
    edf_preprocess_cli.py
    edf_loreta_cli.py
    edf_granger_cli.py
    edf_transfer_entropy_cli.py

Outputs:
    benchmark_results/
        per_recording_timings.csv
        summary_mean_std.csv
        system_info.txt
        logs/
            <recording>/<tool>.stdout.log
            <recording>/<tool>.stderr.log

Runtime includes subprocess startup and command-level file I/O.
Failed commands are recorded and excluded from timing summaries.

Run:
    python benchmark_edf_cli.py

Optional:
    python benchmark_edf_cli.py --edf-dir EDF_format --output-dir benchmark_results
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
except ImportError:
    np = None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_EDF_DIR = Path("EDF_format")
DEFAULT_OUTPUT_DIR = Path("benchmark_results")

LORETA_AGE = 12

# These are the command-line defaults used for the benchmark.
# Edit them if your local CLI interfaces differ.
#
# {edf}      = absolute path to the EDF file
# {out_dir}  = output directory for that recording and tool
#
# Each command is executed from the directory containing this script.
COMMANDS = {
    "quality": [
        sys.executable,
        "edf_quality_cli.py",
        "{edf}",
    ],
    "preprocess": [
        sys.executable,
        "edf_preprocess_cli.py",
        "{edf}",
    ],
    "loreta": [
        sys.executable,
        "edf_loreta_cli.py",
        "{edf}",
        "--age",
        str(LORETA_AGE),
        "-o",
        "{out_dir}",
    ],
    "granger": [
        sys.executable,
        "edf_granger_cli.py",
        "{edf}",
        "-o",
        "{out_dir}",
        "--lag",
        "5",
        "--edge-list",
    ],
    "transfer_entropy": [
        sys.executable,
        "edf_transfer_entropy_cli.py",
        "{edf}",
        "-o",
        "{out_dir}",
        "--bins",
        "4",
        "--lag",
        "1",
    ],
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def format_command(command: list[str], edf_file: Path, out_dir: Path) -> list[str]:
    """Replace command placeholders with actual paths."""
    return [
        arg.replace("{edf}", str(edf_file.resolve()))
           .replace("{out_dir}", str(out_dir.resolve()))
        for arg in command
    ]


def write_system_info(output_dir: Path) -> None:
    """Save basic machine and Python information for reproducibility."""
    info_path = output_dir / "system_info.txt"

    lines = [
        f"Benchmark date: {datetime.now().isoformat(timespec='seconds')}",
        f"Operating system: {platform.platform()}",
        f"Machine: {platform.machine()}",
        f"Processor: {platform.processor()}",
        f"Python version: {sys.version}",
        f"Python executable: {sys.executable}",
    ]

    try:
        import os
        lines.append(f"Logical CPU count: {os.cpu_count()}")
    except Exception:
        pass

    try:
        import psutil
        lines.append(
            f"Total RAM (GB): {psutil.virtual_memory().total / (1024 ** 3):.2f}"
        )
    except ImportError:
        lines.append("Total RAM (GB): not recorded (optional package psutil missing)")

    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_command(
    tool_name: str,
    command: list[str],
    recording_id: str,
    output_dir: Path,
    log_dir: Path,
) -> dict:
    """
    Run one CLI command and return its elapsed time and status.

    Timing begins immediately before subprocess execution and ends
    immediately after the process returns.
    """
    tool_output_dir = output_dir / recording_id / tool_name
    tool_output_dir.mkdir(parents=True, exist_ok=True)

    recording_log_dir = log_dir / recording_id
    recording_log_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = recording_log_dir / f"{tool_name}.stdout.log"
    stderr_path = recording_log_dir / f"{tool_name}.stderr.log"

    start = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        elapsed = time.perf_counter() - start

        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")

        status = "success" if result.returncode == 0 else "failed"

        return {
            "runtime_seconds": elapsed,
            "status": status,
            "return_code": result.returncode,
            "command": " ".join(command),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    except Exception as exc:
        elapsed = time.perf_counter() - start

        stderr_path.write_text(
            f"Exception while executing command:\n{repr(exc)}\n",
            encoding="utf-8",
        )

        return {
            "runtime_seconds": elapsed,
            "status": "exception",
            "return_code": None,
            "command": " ".join(command),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """
    Calculate arithmetic mean and sample standard deviation.

    Sample SD uses n-1 in the denominator (ddof=1).
    For one successful recording, SD is undefined and returned as None.
    """
    if not values:
        return None, None

    mean_value = statistics.mean(values)

    if len(values) < 2:
        return mean_value, None

    return mean_value, statistics.stdev(values)


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all InfantConnect EDF CLI tools."
    )
    parser.add_argument(
        "--edf-dir",
        type=Path,
        default=DEFAULT_EDF_DIR,
        help="Folder containing the Helsinki EDF recordings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for benchmark outputs.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    edf_dir = args.edf_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not edf_dir.exists():
        raise FileNotFoundError(f"EDF folder not found: {edf_dir}")

    # Find all EDF files recursively.
    edf_files = sorted(
        path for path in edf_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".edf"
    )

    if not edf_files:
        raise RuntimeError(f"No .edf files found in {edf_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    write_system_info(output_dir)

    print(f"EDF folder: {edf_dir}")
    print(f"Found {len(edf_files)} EDF files")
    print(f"Output folder: {output_dir}")
    print()

    rows = []

    for file_index, edf_file in enumerate(edf_files, start=1):
        # Preserve a unique identifier even if EDFs in different
        # subfolders share the same filename.
        relative_path = edf_file.relative_to(edf_dir)
        recording_id = str(relative_path.with_suffix("")).replace(
            "/", "__"
        ).replace("\\", "__")

        print(
            f"[{file_index}/{len(edf_files)}] "
            f"Recording: {relative_path}"
        )

        for tool_name, command_template in COMMANDS.items():
            # Each tool gets a distinct output folder for this recording.
            tool_out_dir = output_dir / "tool_outputs" / recording_id / tool_name
            tool_out_dir.mkdir(parents=True, exist_ok=True)

            command = format_command(
                command_template,
                edf_file=edf_file,
                out_dir=tool_out_dir,
            )

            # Resolve script paths relative to the benchmark script directory.
            if len(command) >= 2 and command[1].endswith(".py"):
                script_path = Path(command[1])
                if not script_path.is_absolute():
                    command[1] = str((script_dir / script_path).resolve())

            print(f"    Running {tool_name} ...", end="", flush=True)

            result = run_one_command(
                tool_name=tool_name,
                command=command,
                recording_id=recording_id,
                output_dir=output_dir / "tool_outputs",
                log_dir=logs_dir,
            )

            runtime = result["runtime_seconds"]

            print(
                f" {runtime:.2f} s "
                f"[{result['status']}]"
            )

            rows.append({
                "recording_id": recording_id,
                "edf_relative_path": str(relative_path),
                "edf_size_mb": edf_file.stat().st_size / (1024 ** 2),
                "tool": tool_name,
                "runtime_seconds": runtime,
                "status": result["status"],
                "return_code": result["return_code"],
                "command": result["command"],
                "stdout_log": result["stdout_log"],
                "stderr_log": result["stderr_log"],
            })

        print()

    # -----------------------------------------------------------------
    # Save per-recording data
    # -----------------------------------------------------------------

    per_recording_csv = output_dir / "per_recording_timings.csv"

    fieldnames = [
        "recording_id",
        "edf_relative_path",
        "edf_size_mb",
        "tool",
        "runtime_seconds",
        "status",
        "return_code",
        "command",
        "stdout_log",
        "stderr_log",
    ]

    with per_recording_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # -----------------------------------------------------------------
    # Summarize successful runs: mean, sample SD, median, min, max
    # -----------------------------------------------------------------

    summary_rows = []

    for tool_name in COMMANDS:
        successful = [
            row["runtime_seconds"]
            for row in rows
            if row["tool"] == tool_name and row["status"] == "success"
        ]

        failed_count = sum(
            1 for row in rows
            if row["tool"] == tool_name and row["status"] != "success"
        )

        mean_value, std_value = mean_std(successful)

        summary_rows.append({
            "tool": tool_name,
            "n_success": len(successful),
            "n_failed": failed_count,
            "mean_seconds": mean_value,
            "std_seconds_sample": std_value,
            "mean_plus_minus_std_seconds": (
                f"{mean_value:.2f} ± {std_value:.2f}"
                if mean_value is not None and std_value is not None
                else (
                    f"{mean_value:.2f} ± NA"
                    if mean_value is not None
                    else "NA"
                )
            ),
            "median_seconds": (
                statistics.median(successful) if successful else None
            ),
            "min_seconds": min(successful) if successful else None,
            "max_seconds": max(successful) if successful else None,
        })

    summary_csv = output_dir / "summary_mean_std.csv"

    summary_fields = [
        "tool",
        "n_success",
        "n_failed",
        "mean_seconds",
        "std_seconds_sample",
        "mean_plus_minus_std_seconds",
        "median_seconds",
        "min_seconds",
        "max_seconds",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # -----------------------------------------------------------------
    # Print a concise summary
    # -----------------------------------------------------------------

    print("=" * 78)
    print("BENCHMARK SUMMARY")
    print("=" * 78)

    for row in summary_rows:
        print(
            f"{row['tool']:20s} "
            f"n={row['n_success']:3d} "
            f"failed={row['n_failed']:3d} "
            f"mean ± SD = {row['mean_plus_minus_std_seconds']} s"
        )

    print()
    print(f"Per-recording results: {per_recording_csv}")
    print(f"Summary:               {summary_csv}")
    print(f"Logs:                  {logs_dir}")
    print(f"System info:           {output_dir / 'system_info.txt'}")


if __name__ == "__main__":
    main()
