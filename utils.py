"""Utility functions for CLI and configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON configuration file.

    Parameters
    ----------
    path : str or Path
        Path to JSON file.

    Returns
    -------
    dict
        Parsed configuration dictionary or empty dict if file is missing.
    """
    file = Path(path)
    if not file.exists():
        return {}
    with file.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_color(value: str) -> Tuple[int, int, int]:
    """Parse BGR color from string like '255,0,0'."""
    parts = [int(x) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be B,G,R")
    for c in parts:
        if c < 0 or c > 255:
            raise argparse.ArgumentTypeError("Color values must be in 0..255")
    return tuple(parts)  # type: ignore[return-value]


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct argument parser."""
    parser = argparse.ArgumentParser(description="Irregularity detection demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device id")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--mask-out", help="Path to write binary mask image")
    parser.add_argument("--threshold", type=int, help="Initial threshold")
    parser.add_argument(
        "--color",
        type=parse_color,
        help="Mask color B,G,R (default from config)",
    )
    parser.add_argument(
        "--mode",
        choices=["laplacian", "canny"],
        default="laplacian",
        help="Detection method",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration JSON",
    )
    return parser
