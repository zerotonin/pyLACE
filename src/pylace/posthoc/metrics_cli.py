# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.metrics_cli                                    ║
# ║  « pylace-metrics: cleaned trajectory CSV → per-fly summary »    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Command-line entry point for ``pylace-metrics``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pylace.annotator.sidecar import default_sidecar_path, read_sidecar
from pylace.posthoc.constants import (
    DEFAULT_BOUT_MIN_DURATION_S,
    DEFAULT_BOUT_OFF_MM_S,
    DEFAULT_BOUT_ON_MM_S,
    DEFAULT_THIGMOTAXIS_OUTER_FRAC,
)
from pylace.posthoc.metrics import summarise_tracks
from pylace.posthoc.multifly import (
    nearest_neighbour_distance,
    polarisation,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.trajectory.exists():
        print(f"Trajectory CSV not found: {args.trajectory}", file=sys.stderr)
        return 2

    fps, arena, pix_per_mm = _resolve_video_params(args)
    if fps is None:
        return 2

    out_path = (
        args.out if args.out is not None
        else _default_summary_path(args.trajectory)
    )

    print(f"pylace-metrics: {args.trajectory.name} -> {out_path.name}")
    print(
        f"  fps={fps:.2f}  "
        f"bout on/off={args.on_threshold:.1f}/{args.off_threshold:.1f} mm/s  "
        f"min duration={args.min_duration:.2f} s  "
        f"thigmo outer band={args.outer_band:.2f}",
    )

    df = pd.read_csv(args.trajectory)
    n_tracks = df["track_id"].nunique()
    print(f"  loaded {len(df)} rows across {n_tracks} tracks")

    nn_long: pd.DataFrame | None = None
    if n_tracks >= 2 and pix_per_mm is not None:
        nn_long = nearest_neighbour_distance(df, pix_per_mm=pix_per_mm)
        print(f"  multi-fly: nearest-neighbour distance computed for "
              f"{n_tracks * (n_tracks - 1) // 2} pair(s)")

    summary = summarise_tracks(
        df, fps=fps, arena=arena,
        nn_distances=nn_long,
        on_threshold_mm_s=args.on_threshold,
        off_threshold_mm_s=args.off_threshold,
        min_duration_s=args.min_duration,
        outer_band_frac=args.outer_band,
    )
    summary.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Wrote {len(summary)} per-track rows to {out_path.name}")

    multifly_out_path: Path | None = None
    if args.multifly_out is not None:
        multifly_out_path = args.multifly_out
    elif args.write_multifly and n_tracks >= 2:
        multifly_out_path = _default_multifly_path(args.trajectory)

    if multifly_out_path is not None and nn_long is not None:
        pol = polarisation(df) if "heading_deg" in df.columns else None
        merged = _merge_multifly_timeseries(nn_long, pol)
        merged.to_csv(multifly_out_path, index=False, float_format="%.4f")
        print(
            f"Wrote {len(merged)} multi-fly time-series rows to "
            f"{multifly_out_path.name}",
        )

    print()
    print(summary.to_string(index=False))
    return 0


def _merge_multifly_timeseries(
    nn_long: pd.DataFrame, polar: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build a wide DataFrame: frame_idx | nn_track_<id>_mm | polarisation."""
    nn_wide = nn_long.pivot(
        index="frame_idx", columns="track_id", values="nn_distance_mm",
    )
    nn_wide.columns = [f"nn_track_{int(c)}_mm" for c in nn_wide.columns]
    nn_wide = nn_wide.reset_index()
    if polar is not None:
        return nn_wide.merge(polar, on="frame_idx", how="outer").sort_values(
            "frame_idx",
        ).reset_index(drop=True)
    return nn_wide


def _resolve_video_params(args: argparse.Namespace):
    fps = args.fps
    sidecar_path = (
        args.sidecar if args.sidecar
        else default_sidecar_path(_video_from_trajectory(args.trajectory))
    )
    arena = None
    pix_per_mm = args.pix_per_mm
    if sidecar_path.exists():
        sc = read_sidecar(sidecar_path)
        if fps is None:
            fps = float(sc.video.fps)
        arena = sc.arena
        if pix_per_mm is None:
            pix_per_mm = float(
                sc.calibration.pixel_distance / sc.calibration.physical_mm,
            )
    elif fps is None:
        print(
            f"--fps not given and no sidecar at {sidecar_path}", file=sys.stderr,
        )
        return None, None, None
    return fps, arena, pix_per_mm


def _video_from_trajectory(trajectory: Path) -> Path:
    name = trajectory.name
    if name.endswith(".pylace_trajectory.csv"):
        return trajectory.with_name(name[: -len(".pylace_trajectory.csv")])
    return trajectory.with_suffix("")


def _default_summary_path(trajectory: Path) -> Path:
    name = trajectory.name
    if name.endswith(".pylace_trajectory.csv"):
        stem = name[: -len(".pylace_trajectory.csv")]
        return trajectory.with_name(stem + ".pylace_metrics.csv")
    return trajectory.with_name(trajectory.stem + ".pylace_metrics.csv")


def _default_multifly_path(trajectory: Path) -> Path:
    name = trajectory.name
    if name.endswith(".pylace_trajectory.csv"):
        stem = name[: -len(".pylace_trajectory.csv")]
        return trajectory.with_name(stem + ".pylace_multifly.csv")
    return trajectory.with_name(trajectory.stem + ".pylace_multifly.csv")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylace-metrics",
        description=(
            "Compute per-track kinematic readouts from a cleaned trajectory "
            "CSV produced by pylace-clean: speed summary, walk/stop bouts, "
            "thigmotaxis fraction, yaw-rate summary."
        ),
    )
    p.add_argument("trajectory", type=Path,
                   help="pylace-clean CSV (typically <video>.pylace_trajectory.csv).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV (default: <video>.pylace_metrics.csv).")
    p.add_argument("--sidecar", type=Path, default=None,
                   help="Arena sidecar JSON (for fps and arena geometry).")
    p.add_argument("--fps", type=float, default=None,
                   help="Override fps from the sidecar.")
    p.add_argument("--pix-per-mm", type=float, default=None, dest="pix_per_mm",
                   help="Override calibration from the sidecar (needed for "
                        "multi-fly distance metrics).")
    p.add_argument("--multifly-out", type=Path, default=None, dest="multifly_out",
                   help="Optional output path for the per-frame multi-fly "
                        "time-series (NN distance per track + polarisation). "
                        "Implies --write-multifly.")
    p.add_argument("--write-multifly", action="store_true", dest="write_multifly",
                   help="Write the per-frame multi-fly CSV at the default "
                        "<video>.pylace_multifly.csv path.")
    p.add_argument("--on-threshold", type=float,
                   default=DEFAULT_BOUT_ON_MM_S, dest="on_threshold",
                   help="Schmitt-trigger ON threshold (mm/s).")
    p.add_argument("--off-threshold", type=float,
                   default=DEFAULT_BOUT_OFF_MM_S, dest="off_threshold",
                   help="Schmitt-trigger OFF threshold (mm/s).")
    p.add_argument("--min-duration", type=float,
                   default=DEFAULT_BOUT_MIN_DURATION_S, dest="min_duration",
                   help="Minimum walk-bout duration (s).")
    p.add_argument("--outer-band", type=float,
                   default=DEFAULT_THIGMOTAXIS_OUTER_FRAC, dest="outer_band",
                   help="Outer-band fraction for thigmotaxis (0..1).")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
