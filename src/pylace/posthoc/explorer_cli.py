# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.explorer_cli                                   ║
# ║  « pylace-explore: synchronized trajectory + time-series view »  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Command-line entry point for ``pylace-explore``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pylace.posthoc.explorer import run


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 2
    if not args.trajectory.exists():
        print(f"Trajectory CSV not found: {args.trajectory}", file=sys.stderr)
        return 2
    return run(
        video=args.video,
        trajectory=args.trajectory,
        sidecar_path=args.sidecar,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylace-explore",
        description=(
            "Open a synchronized data explorer: trajectory pane on top, "
            "stacked time-series plots (speed, yaw rate, distance to wall) "
            "below. Click anywhere to scrub time."
        ),
    )
    p.add_argument("video", type=Path,
                   help="Source video file.")
    p.add_argument("trajectory", type=Path,
                   help="pylace-clean CSV "
                        "(typically <video>.pylace_trajectory.csv).")
    p.add_argument("--sidecar", type=Path, default=None,
                   help="Arena sidecar JSON (default: alongside the video).")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
