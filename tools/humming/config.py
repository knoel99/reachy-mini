"""Shared paths + audio constants for the humming tooling."""
from __future__ import annotations

from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent
SOURCES_DIR = TOOL_ROOT / "data" / "sources"
SAMPLES_DIR = TOOL_ROOT / "data" / "samples"
OUTPUTS_DIR = TOOL_ROOT / "data" / "outputs"

TARGET_RATE = 22_050  # internal SR for analysis/synthesis
