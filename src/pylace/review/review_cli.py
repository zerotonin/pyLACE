# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — review.review_cli                                      ║
# ║  « pylace-candidates: scan a trajectory for swap candidates »    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Command-line entry point for ``pylace-candidates``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pylace.annotator.sidecar import default_sidecar_path, read_sidecar
from pylace.posthoc.io import video_path_from_trajectory
from pylace.review.candidates import (
    default_candidates_path,
    detect_candidates,
    write_candidates,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.trajectory.exists():
        print(f"Trajectory CSV not found: {args.trajectory}", file=sys.stderr)
        return 2

    fps, pix_per_mm = _resolve_video_params(args)
    if fps is None or pix_per_mm is None:
        return 2

    out_path = args.out if args.out is not None else default_candidates_path(args.trajectory)
    df = pd.read_csv(args.trajectory)
    n_tracks = df["track_id"].nunique()
    print(f"pylace-candidates: {args.trajectory.name} -> {out_path.name}")
    print(
        f"  fps={fps:.2f}  pix_per_mm={pix_per_mm:.4f}  "
        f"contact={args.contact_mm:.2f} mm  jump={args.jump_mm_per_frame:.2f} mm/frame  "
        f"window={args.window_s:.2f} s",
    )
    print(f"  loaded {len(df)} rows across {n_tracks} tracks")

    candidates = detect_candidates(
        df, fps=fps, pix_per_mm=pix_per_mm,
        contact_threshold_mm=args.contact_mm,
        jump_threshold_mm_per_frame=args.jump_mm_per_frame,
        coalesce_window_frames=args.coalesce_window_frames,
        window_s=args.window_s,
    )
    write_candidates(candidates, out_path)
    print(f"  {len(candidates)} candidate(s) written")
    if candidates:
        head = candidates[: min(20, len(candidates))]
        print()
        print(pd.DataFrame([
            {
                "frame_start": c.frame_start,
                "frame_end": c.frame_end,
                "animals": f"{c.animal_a},{c.animal_b}",
                "source": c.source,
                "min_d_px": round(c.min_distance_px, 2),
                "max_jump_px": round(c.max_jump_px, 2),
                "n_frames": c.n_frames,
            } for c in head
        ]).to_string(index=False))
    return 0


def _resolve_video_params(args: argparse.Namespace) -> tuple[float | None, float | None]:
    fps = args.fps
    pix_per_mm = args.pix_per_mm
    sidecar_path = (
        args.sidecar if args.sidecar
        else default_sidecar_path(video_path_from_trajectory(args.trajectory))
    )
    if sidecar_path.exists():
        sc = read_sidecar(sidecar_path)
        if fps is None:
            fps = float(sc.video.fps)
        if pix_per_mm is None:
            pix_per_mm = float(
                sc.calibration.pixel_distance / sc.calibration.physical_mm,
            )
    if fps is None:
        print(f"--fps not given and no sidecar at {sidecar_path}", file=sys.stderr)
        return None, None
    if pix_per_mm is None:
        print(
            f"--pix-per-mm not given and no sidecar at {sidecar_path}",
            file=sys.stderr,
        )
        return None, None
    return fps, pix_per_mm


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylace-candidates",
        description=(
            "Scan a cleaned trajectory CSV for swap-candidate events "
            "(close-contact, fast-jump, NaN spans) and write them next "
            "to the trajectory as <video>.pylace_candidates.csv. This "
            "is the GUI's queue source; it is independent of the audit."
        ),
    )
    p.add_argument("trajectory", type=Path,
                   help="pylace-clean CSV "
                        "(typically <video>.pylace_trajectory.csv).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV (default: <video>.pylace_candidates.csv).")
    p.add_argument("--sidecar", type=Path, default=None,
                   help="Arena sidecar JSON (for fps + calibration).")
    p.add_argument("--fps", type=float, default=None,
                   help="Override fps from the sidecar.")
    p.add_argument("--pix-per-mm", type=float, default=None, dest="pix_per_mm",
                   help="Override calibration from the sidecar.")
    p.add_argument("--contact-mm", type=float, default=5.0, dest="contact_mm",
                   help="Two tracks closer than this trigger a 'contact' event "
                        "(default 5 mm).")
    p.add_argument("--jump-mm-per-frame", type=float, default=5.0,
                   dest="jump_mm_per_frame",
                   help="Single-frame Euclidean step larger than this triggers "
                        "a 'jump' event (default 5 mm/frame).")
    p.add_argument("--window-s", type=float, default=1.0, dest="window_s",
                   help="Coalesce-window default = round(fps x window-s) frames. "
                        "Default 1 s.")
    p.add_argument("--coalesce-window-frames", type=int, default=None,
                   dest="coalesce_window_frames",
                   help="Adjacent event frames within this gap merge into one "
                        "block. Default = round(fps x window-s).")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
