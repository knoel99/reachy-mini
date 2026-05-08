"""Export emotion WAVs from the Reachy Mini library to a local folder.

The HuggingFace dataset `pollen-robotics/reachy-mini-emotions-library`
ships a WAV next to each emotion JSON. This script walks the library
and copies every available WAV to a folder of your choice so you can
listen to / edit / reuse them outside the robot stack.

Usage:
    python scripts/export_emotion_sounds.py [output_dir]

Default output_dir: ./emotion_sounds/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from reachy_mini.motion.recorded_move import RecordedMoves


DEFAULT_DATASET = "pollen-robotics/reachy-mini-emotions-library"


def export(out_dir: Path, dataset: str = DEFAULT_DATASET) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    library = RecordedMoves(dataset)
    names = sorted(library.list_moves())

    copied: list[str] = []
    silent: list[str] = []
    for name in names:
        move = library.get(name)
        src = move.sound_path
        if src is None:
            silent.append(name)
            continue
        dest = out_dir / f"{name}.wav"
        shutil.copy2(src, dest)
        copied.append(name)

    print(f"Copied {len(copied)} WAV(s) to {out_dir}/")
    for n in copied:
        print(f"  + {n}.wav")
    if silent:
        print(f"\n{len(silent)} emotion(s) without sound (skipped):")
        for n in silent:
            print(f"  - {n}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "output_dir",
        nargs="?",
        default="emotion_sounds",
        help="Destination folder (default: ./emotion_sounds/).",
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset id (default: {DEFAULT_DATASET}).",
    )
    args = p.parse_args()
    return export(Path(args.output_dir), dataset=args.dataset)


if __name__ == "__main__":
    sys.exit(main())
