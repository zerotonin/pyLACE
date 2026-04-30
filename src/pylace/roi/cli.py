# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi.cli                                                ║
# ║  « pylace-roi entry point »                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Command-line entry for ``pylace-roi``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Top-level CLI. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="pylace-roi",
        description=(
            "Open a PyQt6 window for drawing one or more regions of "
            "interest on the first frame of a video. ROIs combine into "
            "a single boolean mask (merge mode) the detector intersects "
            "with the arena mask."
        ),
    )
    parser.add_argument("video", type=Path, help="Path to the video file.")
    parser.add_argument(
        "--rois", type=Path, default=None,
        help="ROI sidecar path (default: <video>.pylace_rois.json).",
    )
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 2

    from pylace.roi.main_window import run

    return run(video=args.video, rois_path=args.rois)


if __name__ == "__main__":
    raise SystemExit(main())
