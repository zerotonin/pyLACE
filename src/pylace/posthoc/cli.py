# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.cli                                            ║
# ║  « pylace-clean: detections CSV → cleaned trajectory CSV »       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Command-line entry point for ``pylace-clean``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pylace.annotator.sidecar import default_sidecar_path, read_sidecar
from pylace.posthoc.clean import clean_trajectory
from pylace.posthoc.constants import (
    DEFAULT_MAX_SPEED_MM_S,
    DEFAULT_SG_POLYORDER,
    DEFAULT_SG_WINDOW,
)
from pylace.posthoc.heading import compute_yaw, resolve_headings
from pylace.posthoc.io import (
    default_trajectory_path,
    read_detections,
    video_path_from_trajectory,
    write_trajectory,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.detections.exists():
        print(f"Detections CSV not found: {args.detections}", file=sys.stderr)
        return 2

    fps, pix_per_mm = _resolve_video_params(args)
    if fps is None or pix_per_mm is None:
        return 2

    out_path = (
        args.out if args.out is not None
        else default_trajectory_path(args.detections)
    )

    print(f"pylace-clean: {args.detections.name} -> {out_path.name}")
    print(
        f"  fps={fps:.2f}  pix_per_mm={pix_per_mm:.4f}  "
        f"max_gap_frames={args.max_gap_frames or 'auto'}  "
        f"max_speed={args.max_speed_mm_s} mm/s",
    )
    print(
        f"  smoothing: SG window={args.sg_window} order={args.sg_polyorder}",
    )

    df = read_detections(args.detections)
    print(f"  loaded {len(df)} rows across {df['track_id'].nunique()} tracks")

    df_cleaned = clean_trajectory(
        df,
        fps=fps, pix_per_mm=pix_per_mm,
        max_gap_frames=args.max_gap_frames,
        max_speed_mm_s=args.max_speed_mm_s,
        sg_window=args.sg_window,
        sg_polyorder=args.sg_polyorder,
    )
    n_outliers = int(df_cleaned["outlier_rejected"].sum())
    n_interp = int(df_cleaned["interpolated"].sum())
    print(f"  cleaned rows: {len(df_cleaned)}  "
          f"interpolated: {n_interp}  outliers rejected: {n_outliers}")

    if not args.no_heading:
        df_cleaned = resolve_headings(df_cleaned)
        df_cleaned = compute_yaw(df_cleaned, fps=fps)
        n_unresolved = int(df_cleaned["heading_unresolved"].sum())
        print(f"  headings resolved; {n_unresolved} frames had no orientation input")

    write_trajectory(df_cleaned, out_path)
    print(f"Wrote {len(df_cleaned)} rows to {out_path.name}")
    return 0


def _resolve_video_params(args: argparse.Namespace) -> tuple[float | None, float | None]:
    """Pull fps + pix_per_mm from the arena sidecar unless overridden."""
    fps = args.fps
    pix_per_mm = args.pix_per_mm
    if fps is not None and pix_per_mm is not None:
        return fps, pix_per_mm

    sidecar_path = (
        args.sidecar if args.sidecar
        else default_sidecar_path(video_path_from_trajectory(args.detections))
    )
    if not sidecar_path.exists():
        if fps is None:
            print(
                f"--fps not given and no sidecar at {sidecar_path}", file=sys.stderr,
            )
            return None, None
        if pix_per_mm is None:
            print(
                f"--pix-per-mm not given and no sidecar at {sidecar_path}",
                file=sys.stderr,
            )
            return None, None

    if sidecar_path.exists():
        sc = read_sidecar(sidecar_path)
        if fps is None:
            fps = float(sc.video.fps)
        if pix_per_mm is None:
            pix_per_mm = float(sc.calibration.pixel_distance / sc.calibration.physical_mm)
    return fps, pix_per_mm


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylace-clean",
        description=(
            "Clean a pylace-detect CSV: linear gap fill, velocity-threshold "
            "outlier rejection, Savitzky-Golay smoothing, head/tail "
            "disambiguation, and yaw + yaw-rate computation."
        ),
    )
    p.add_argument("detections", type=Path,
                   help="pylace-detect CSV (typically <video>.pylace_detections.csv).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV (default: <video>.pylace_trajectory.csv).")
    p.add_argument("--sidecar", type=Path, default=None,
                   help="Arena sidecar JSON (default: alongside the video).")
    p.add_argument("--fps", type=float, default=None,
                   help="Override fps from the sidecar.")
    p.add_argument("--pix-per-mm", type=float, default=None, dest="pix_per_mm",
                   help="Override calibration from the sidecar.")
    p.add_argument("--max-gap-frames", type=int, default=None, dest="max_gap_frames",
                   help="Largest gap (frames) bridged by interpolation. "
                        "Default ~ fps/12.")
    p.add_argument("--max-speed-mm-s", type=float,
                   default=DEFAULT_MAX_SPEED_MM_S, dest="max_speed_mm_s",
                   help="Outlier-rejection velocity ceiling.")
    p.add_argument("--sg-window", type=int, default=DEFAULT_SG_WINDOW,
                   dest="sg_window",
                   help="Savitzky-Golay window length (odd, > polyorder).")
    p.add_argument("--sg-polyorder", type=int, default=DEFAULT_SG_POLYORDER,
                   dest="sg_polyorder",
                   help="Savitzky-Golay polynomial order.")
    p.add_argument("--no-heading", action="store_true", dest="no_heading",
                   help="Skip head/tail disambiguation and yaw computation.")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
