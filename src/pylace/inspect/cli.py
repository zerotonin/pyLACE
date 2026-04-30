# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — inspect.cli                                            ║
# ║  « pylace-inspect entry point »                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Command-line entry for ``pylace-inspect``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Top-level CLI. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="pylace-inspect",
        description=(
            "Open a PyQt6 inspector for a tracked video. Loads "
            "<video>.pylace_detections.csv (override with --csv) and shows "
            "a scrub view with a fading trail plus a static-trajectory "
            "overview, both colour-coded per track."
        ),
    )
    parser.add_argument("video", type=Path, help="Path to the source video.")
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Detections CSV (default: <video>.pylace_detections.csv).",
    )
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 2

    from pylace.inspect.main_window import run

    return run(video=args.video, csv_path=args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
