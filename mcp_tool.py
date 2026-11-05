
"""
MCP wrappers for InfantConnect command-line tools.

All processing is delegated to the existing CLI scripts.
"""

from pathlib import Path
import subprocess
import sys
import json
from typing import Any

from infant_brain_viewer.mcp_config import (
    PACKAGE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIMEOUT,
    LORETA_TIMEOUT,
    ALLOWED_DATA_DIR,
)

PACKAGE_DIR = Path(__file__).resolve().parent

CLI_SCRIPTS = {
    "quality": "edf_quality_cli.py",
    "preprocess": "edf_preprocess_cli.py",
    "granger": "edf_granger_cli.py",
    "transfer_entropy": "edf_transfer_entropy_cli.py",
    "loreta": "edf_loreta_cli.py",
    "effconn": "edf_effconn_cli.py",
}


def _validate_edf(edf_file: str) -> Path:
    """Validate an input EDF file."""
    path = Path(edf_file).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"EDF file not found: {path}")

    if path.suffix.lower() != ".edf":
        raise ValueError("Input file must have an .edf extension")

    return path


def _validate_output(output_dir: str) -> Path:
    """Create and validate the output directory."""
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_cli(
    script_name: str,
    args: list[str],
    timeout: int = 1800,
) -> dict[str, Any]:
    """Execute an InfantConnect CLI script."""

    script = PACKAGE_DIR / script_name

    if not script.is_file():
        raise FileNotFoundError(
            f"CLI script not found: {script}"
        )

    command = [
        sys.executable,
        str(script),
        *args,
    ]

    result = subprocess.run(
        command,
        cwd=str(PACKAGE_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    return {
        "success": result.returncode == 0,
        "return_code": result.returncode,
        "command": command,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
    }


def _add_artifacts(
    result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Attach the list of generated files to the result."""

    result["output_directory"] = str(output_dir)

    result["generated_files"] = [
        str(p)
        for p in sorted(output_dir.rglob("*"))
        if p.is_file()
    ]

    return result


def eeg_quality(
    edf_file: str,
    output_dir: str,
    fail_on_warn: bool = False,
) -> dict[str, Any]:
    """
    Assess EEG recording quality.

    Returns the quality report in text and JSON formats.
    """

    edf = _validate_edf(edf_file)
    out = _validate_output(output_dir)

    args = [
        str(edf),
        "-o", str(out),
        "--format", "both",
    ]

    if fail_on_warn:
        args.append("--fail-on-warn")

    result = _run_cli(
        CLI_SCRIPTS["quality"], args
    )

    return _add_artifacts(result, out)


def eeg_preprocess(
    edf_file: str,
    output_dir: str,
    config_file: str | None = None,
    settings: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Preprocess an EDF recording.

    Settings use key=value syntax, e.g.:
    lowpass=45, resample=true, resample_hz=125.
    """

    edf = _validate_edf(edf_file)
    out = _validate_output(output_dir)

    args = [str(edf)]

    if config_file:
        config = Path(config_file).expanduser().resolve()

        if not config.is_file():
            raise FileNotFoundError(
                f"Config file not found: {config}"
            )

        args.extend(["-c", str(config)])

    for setting in settings or []:
        if "=" not in setting:
            raise ValueError(
                f"Invalid setting: {setting}. Use key=value."
            )

        args.extend(["--set", setting])

    args.extend(["-o", str(out)])

    if overwrite:
        args.append("--overwrite")

    result = _run_cli(
        CLI_SCRIPTS["preprocess"], args
    )

    return _add_artifacts(result, out)


def eeg_default_preprocessing_config(
    output_file: str,
) -> dict[str, Any]:
    """
    Generate the default preprocessing JSON configuration.
    """

    target = Path(output_file).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    script = PACKAGE_DIR / CLI_SCRIPTS["preprocess"]

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--write-default-config",
        ],
        cwd=str(PACKAGE_DIR),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    if result.returncode != 0:
        return {
            "success": False,
            "stderr": result.stderr,
        }

    # Validate that the CLI produced valid JSON.
    config = json.loads(result.stdout)

    target.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    return {
        "success": True,
        "config_file": str(target),
        "configuration": config,
    }


def eeg_granger(
    edf_file: str,
    output_dir: str,
    lag: int = 5,
    edge_list: bool = True,
) -> dict[str, Any]:
    """
    Compute Granger-causality effective connectivity.
    """

    edf = _validate_edf(edf_file)
    out = _validate_output(output_dir)

    if lag < 1:
        raise ValueError("lag must be >= 1")

    args = [
        str(edf),
        "-o", str(out),
        "--lag", str(lag),
    ]

    if edge_list:
        args.append("--edge-list")

    result = _run_cli(
        CLI_SCRIPTS["granger"], args
    )

    return _add_artifacts(result, out)


def eeg_transfer_entropy(
    edf_file: str,
    output_dir: str,
    bins: int = 4,
    lag: int = 1,
) -> dict[str, Any]:
    """
    Compute transfer-entropy effective connectivity.
    """

    edf = _validate_edf(edf_file)
    out = _validate_output(output_dir)

    if bins < 2:
        raise ValueError("bins must be >= 2")

    if lag < 1:
        raise ValueError("lag must be >= 1")

    args = [
        str(edf),
        "-o", str(out),
        "--bins", str(bins),
        "--lag", str(lag),
    ]

    result = _run_cli(
        CLI_SCRIPTS["transfer_entropy"], args
    )

    return _add_artifacts(result, out)


def eeg_loreta(
    edf_file: str,
    output_dir: str,
    age: int,
    surface: str | None = None,
    screenshot: bool = False,
) -> dict[str, Any]:
    """
    Perform age-specific LORETA source localization.
    """

    edf = _validate_edf(edf_file)
    out = _validate_output(output_dir)

    if age < 0 or age > 120:
        raise ValueError("Invalid age in months")

    args = [
        str(edf),
        "--age", str(age),
        "-o", str(out),
    ]

    if surface:
        args.extend(["--surface", surface])

    if screenshot:
        args.append("--screenshot")

    result = _run_cli(
        CLI_SCRIPTS["loreta"], args,
        timeout=3600,
    )

    return _add_artifacts(result, out)


def eeg_effective_connectivity(
    edf_file: str,
    output_dir: str,
    arguments: list[str],
) -> dict[str, Any]:
    """
    Run the combined effective-connectivity CLI.

    Arguments are passed as separate command-line arguments.
    Consult edf_effconn_cli.py --help for supported options.
    """

    edf = _validate_edf(edf_file)
    out = _validate_output(output_dir)

    args = [
        str(edf),
        "-o", str(out),
        *arguments,
    ]

    result = _run_cli(
        CLI_SCRIPTS["effconn"], args,
        timeout=1800,
    )

    return _add_artifacts(result, out)

