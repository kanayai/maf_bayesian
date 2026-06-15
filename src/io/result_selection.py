"""Explicit result selection for analysis commands."""

from __future__ import annotations

import argparse
from pathlib import Path


def existing_netcdf_path(value: str) -> Path:
    """Resolve and validate an explicitly selected NetCDF result file."""
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Result file does not exist: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Result path is not a file: {path}")
    if path.suffix.lower() != ".nc":
        raise argparse.ArgumentTypeError(f"Result file must use the .nc extension: {path}")
    return path
