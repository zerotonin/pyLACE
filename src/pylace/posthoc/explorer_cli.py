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
        review_mode=args.review,
        candidates_path=args.candidates,
        audit_log_path=args.audit_log,
        verdicts_path=args.verdicts,
        reviewer=args.reviewer,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylace-explore",
        description=(
            "Open a synchronized data explorer: trajectory pane on top, "
            "stacked time-series plots (speed, yaw rate, distance to wall) "
            "below. Click anywhere to scrub time. Pass --review to also "
            "dock the swap-review panel for human-in-the-loop triage."
        ),
    )
    p.add_argument("video", type=Path,
                   help="Source video file.")
    p.add_argument("trajectory", type=Path,
                   help="pylace-clean CSV "
                        "(typically <video>.pylace_trajectory.csv).")
    p.add_argument("--sidecar", type=Path, default=None,
                   help="Arena sidecar JSON (default: alongside the video).")
    p.add_argument("--review", action="store_true",
                   help="Enable the swap-review dock panel: load candidates + "
                        "audit log + verdicts and let the user assign verdicts.")
    p.add_argument("--candidates", type=Path, default=None,
                   help="Path to a pylace_candidates.csv (default: "
                        "auto-detect next to the trajectory).")
    p.add_argument("--audit-log", type=Path, default=None, dest="audit_log",
                   help="Path to a pylace_audit_swaps.csv (default: auto-detect).")
    p.add_argument("--verdicts", type=Path, default=None,
                   help="Path to a pylace_verdicts.csv (default: auto-detect; "
                        "created on first verdict).")
    p.add_argument("--reviewer", type=str, default=None,
                   help="Reviewer name recorded with each verdict "
                        "(default: $USER).")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
