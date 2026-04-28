"""Command-line entry point for ``pylace-tune``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Top-level CLI. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="pylace-tune",
        description=(
            "Open a PyQt6 window for interactive tuning of detection and "
            "background parameters. Loads N preview frames from a chosen "
            "time window and updates overlays live as parameters change."
        ),
    )
    parser.add_argument("video", type=Path, help="Path to the video file.")
    parser.add_argument(
        "--sidecar", type=Path, default=None,
        help="Arena sidecar (default: <video>.pylace_arena.json).",
    )
    parser.add_argument(
        "--params", type=Path, default=None,
        help="Tuning params path (default: <video>.pylace_detect_params.json).",
    )
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 2

    from pylace.tune.main_window import run

    return run(video=args.video, sidecar_path=args.sidecar, params_path=args.params)


if __name__ == "__main__":
    raise SystemExit(main())
