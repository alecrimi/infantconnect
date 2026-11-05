
"""
InfantConnect MCP Server.

Exposes EEG processing tools through the Model Context Protocol.

Default transport: STDIO
Optional transport: Streamable HTTP
"""

import os
from mcp.server.fastmcp import FastMCP

from .mcp_tools import (
    eeg_quality,
    eeg_preprocess,
    eeg_default_preprocessing_config,
    eeg_granger,
    eeg_transfer_entropy,
    eeg_loreta,
    eeg_effective_connectivity,
)


mcp = FastMCP(
    "InfantConnect",
    host=os.getenv("INFANTCONNECT_MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("INFANTCONNECT_MCP_PORT", "8000")),
)


@mcp.tool()
def check_eeg_quality(
    edf_file: str,
    output_dir: str,
    fail_on_warn: bool = False,
) -> dict:
    """
    Assess the quality of an EEG EDF recording.

    Generates text and JSON quality reports.

    Args:
        edf_file: Path to the input EDF recording.
        output_dir: Directory for generated reports.
        fail_on_warn: Return a non-zero CLI status for warnings.
    """
    return eeg_quality(
        edf_file,
        output_dir,
        fail_on_warn,
    )


@mcp.tool()
def generate_preprocessing_config(
    output_file: str,
) -> dict:
    """
    Generate the default InfantConnect preprocessing configuration.

    Returns a JSON configuration file that can subsequently
    be modified and passed to preprocess_eeg.
    """
    return eeg_default_preprocessing_config(
        output_file
    )


@mcp.tool()
def preprocess_eeg(
    edf_file: str,
    output_dir: str,
    config_file: str = "",
    settings: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Preprocess an EEG recording.

    Examples of settings:
        lowpass=45
        resample=true
        resample_hz=125

    A configuration file is optional.
    """
    return eeg_preprocess(
        edf_file,
        output_dir,
        config_file or None,
        settings,
        overwrite,
    )


@mcp.tool()
def compute_granger_connectivity(
    edf_file: str,
    output_dir: str,
    lag: int = 5,
    edge_list: bool = True,
) -> dict:
    """
    Compute Granger-causality effective connectivity.

    Returns generated connectivity matrices and edge lists.
    """
    return eeg_granger(
        edf_file,
        output_dir,
        lag,
        edge_list,
    )


@mcp.tool()
def compute_transfer_entropy(
    edf_file: str,
    output_dir: str,
    bins: int = 4,
    lag: int = 1,
) -> dict:
    """
    Compute transfer-entropy effective connectivity.

    Args:
        bins: Number of histogram bins.
        lag: Temporal lag.
    """
    return eeg_transfer_entropy(
        edf_file,
        output_dir,
        bins,
        lag,
    )


@mcp.tool()
def compute_loreta(
    edf_file: str,
    output_dir: str,
    age: int,
    surface: str = "",
    screenshot: bool = False,
) -> dict:
    """
    Estimate age-specific cortical source activity using LORETA.

    Age is specified in months.
    """
    return eeg_loreta(
        edf_file,
        output_dir,
        age,
        surface or None,
        screenshot,
    )


@mcp.tool()
def run_effective_connectivity(
    edf_file: str,
    output_dir: str,
    arguments: list[str],
) -> dict:
    """
    Run the combined effective-connectivity CLI.

    Supply arguments supported by edf_effconn_cli.py.
    Consult that CLI's --help for available options.
    """
    return eeg_effective_connectivity(
        edf_file,
        output_dir,
        arguments,
    )


def main():
    transport = os.getenv(
        "INFANTCONNECT_MCP_TRANSPORT",
        "stdio",
    ).lower()

    if transport not in ("stdio", "streamable-http"):
        raise ValueError(
            "Transport must be stdio or streamable-http"
        )

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()

