"""Shared paths + audio constants for the humming lab."""
from __future__ import annotations

from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
SOURCES_DIR = LAB_ROOT / "data" / "sources"
SAMPLES_DIR = LAB_ROOT / "data" / "samples"
OUTPUTS_DIR = LAB_ROOT / "data" / "outputs"
VIZ_DIR = LAB_ROOT / "data" / "viz"
WEB_DIR = LAB_ROOT / "web"
WEB_AUDIO_DIR = WEB_DIR / "audio"
WEB_DATA_DIR = WEB_DIR / "data"

TARGET_RATE = 22_050   # internal SR for analysis/synthesis (good f0 + small files)
OUTPUT_RATE = 22_050   # WAV output rate for the demo (browser plays this fine)
