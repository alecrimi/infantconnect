
"""Configuration for the InfantConnect MCP server."""

import os
from pathlib import Path

# Directory containing the existing CLI scripts
PACKAGE_DIR = Path(__file__).resolve().parent

# Default directory for generated results
DEFAULT_OUTPUT_DIR = Path(
    os.getenv(
        "INFANTCONNECT_OUTPUT_DIR",
        str(Path.home() / "infantconnect_results")
    )
).expanduser().resolve()

# MCP server configuration
MCP_NAME = "InfantConnect"

MCP_TRANSPORT = os.getenv(
    "INFANTCONNECT_MCP_TRANSPORT",
    "stdio"
).lower()

MCP_HOST = os.getenv(
    "INFANTCONNECT_MCP_HOST",
    "127.0.0.1"
)

MCP_PORT = int(
    os.getenv("INFANTCONNECT_MCP_PORT", "8000")
)

# Execution limits (seconds)
DEFAULT_TIMEOUT = int(
    os.getenv("INFANTCONNECT_TIMEOUT", "1800")
)

LORETA_TIMEOUT = int(
    os.getenv("INFANTCONNECT_LORETA_TIMEOUT", "3600")
)

# Optional filesystem restrictions.
# If configured, only files inside this directory are accessible.
ALLOWED_DATA_DIR = os.getenv(
    "INFANTCONNECT_ALLOWED_DATA_DIR",
    ""
)

ALLOWED_DATA_DIR = (
    Path(ALLOWED_DATA_DIR).expanduser().resolve()
    if ALLOWED_DATA_DIR else None
)

DEFAULT_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

