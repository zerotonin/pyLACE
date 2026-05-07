# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.audit_cli                                      ║
# ║  « pylace-audit: re-tag IDs in a cleaned trajectory CSV »        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Command-line entry point for ``pylace-audit``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pylace.annotator.sidecar import default_sidecar_path, read_sidecar
from pylace.posthoc.audit import audit_track_identities
from pylace.posthoc.io import (
    trajectory_stem,
    video_path_from_trajectory,
    write_trajectory,
)
from pylace.tracking.constants import (
    DEFAULT_KALMAN_INITIAL_V_STD,
    DEFAULT_KALMAN_Q_POS,
    DEFAULT_KALMAN_Q_VEL,
    DEFAULT_KALMAN_R_POS,
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

    out_path = (
        args.out if args.out is not None
        else _default_audited_path(args.trajectory)
    )
    swap_log_path = (
        args.swap_log if args.swap_log is not None
        else out_path.with_name(out_path.name.replace(
            ".pylace_audited.csv", ".pylace_audit_swaps.csv",
        ))
    )

    max_block_s = (
        args.max_event_block_s if args.max_event_block_s is not None
        else args.window_s * 5.0
    )
    max_jump_mm = (
        args.max_jump_mm if args.max_jump_mm is not None
        else args.contact_mm * 2.0
    )
    print(f"pylace-audit: {args.trajectory.name} -> {out_path.name}")
    print(
        f"  fps={fps:.2f}  pix_per_mm={pix_per_mm:.4f}  "
        f"contact={args.contact_mm:.2f} mm  window={args.window_s:.2f} s  "
        f"swap_cost_ratio={args.swap_cost_ratio:.2f}",
    )
    print(
        f"  gates: max_event_block={max_block_s:.2f} s  "
        f"max_jump={max_jump_mm:.2f} mm",
    )

    fingerprint_path = (
        args.fingerprint if args.fingerprint is not None
        else _default_fingerprint_path(args.trajectory)
    )
    if fingerprint_path is not None and fingerprint_path.exists():
        print(
            f"  appearance: {fingerprint_path.name}  "
            f"weights app={args.appearance_weight:.2f} "
            f"axis={args.axis_ratio_weight:.2f} "
            f"area={args.area_weight:.2f}",
        )
    else:
        print("  appearance: disabled (no fingerprint sidecar)")
        fingerprint_path = None

    df = pd.read_csv(args.trajectory)
    n_tracks = df["track_id"].nunique()
    print(f"  loaded {len(df)} rows across {n_tracks} tracks")

    audited, swap_log = audit_track_identities(
        df, fps=fps, pix_per_mm=pix_per_mm,
        contact_threshold_mm=args.contact_mm,
        window_s=args.window_s,
        swap_cost_ratio=args.swap_cost_ratio,
        coalesce_window_frames=args.coalesce_window_frames,
        max_event_block_s=max_block_s,
        max_jump_mm=max_jump_mm,
        fingerprint_path=fingerprint_path,
        appearance_weight=args.appearance_weight,
        axis_ratio_weight=args.axis_ratio_weight,
        area_weight=args.area_weight,
        kalman_q_pos=args.kalman_q_pos,
        kalman_q_vel=args.kalman_q_vel,
        kalman_r_pos=args.kalman_r_pos,
        kalman_initial_v_std=args.kalman_initial_v_std,
    )
    write_trajectory(audited, out_path)
    print(f"Wrote {len(audited)} rows to {out_path.name}")

    swap_log.to_csv(swap_log_path, index=False)
    print(
        f"  {len(swap_log)} swap event(s) — log: {swap_log_path.name}",
    )
    if not swap_log.empty:
        print()
        print(swap_log.head(20).to_string(index=False))
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


def _default_audited_path(trajectory: Path) -> Path:
    return trajectory.with_name(trajectory_stem(trajectory) + ".pylace_audited.csv")


def _default_fingerprint_path(trajectory: Path) -> Path | None:
    """Conventional fingerprint sidecar path next to the trajectory CSV's video."""
    video = video_path_from_trajectory(trajectory)
    candidate = video.with_name(video.name + ".pylace_fingerprints.npz")
    return candidate if candidate.exists() else None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylace-audit",
        description=(
            "Re-tag track_id in a cleaned trajectory CSV by globally "
            "re-optimising assignments at proximity / gap events. Catches "
            "post-merge ID swaps that local Hungarian missed."
        ),
    )
    p.add_argument("trajectory", type=Path,
                   help="pylace-clean CSV "
                        "(typically <video>.pylace_trajectory.csv).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV (default: <video>.pylace_audited.csv).")
    p.add_argument("--swap-log", type=Path, default=None, dest="swap_log",
                   help="Path for the per-event swap log CSV.")
    p.add_argument("--sidecar", type=Path, default=None,
                   help="Arena sidecar JSON (for fps + calibration).")
    p.add_argument("--fps", type=float, default=None,
                   help="Override fps from the sidecar.")
    p.add_argument("--pix-per-mm", type=float, default=None, dest="pix_per_mm",
                   help="Override calibration from the sidecar.")
    p.add_argument("--contact-mm", type=float, default=5.0, dest="contact_mm",
                   help="Two tracks closer than this trigger an audit event.")
    p.add_argument("--window-s", type=float, default=1.0, dest="window_s",
                   help="Pre/post-event window half-width in seconds.")
    p.add_argument("--swap-cost-ratio", type=float, default=0.7,
                   dest="swap_cost_ratio",
                   help="Commit a swap when its cost is below this fraction "
                        "of the identity-permutation cost. Default 0.7 "
                        "(= require 30%% improvement).")
    p.add_argument("--coalesce-window-frames", type=int, default=None,
                   dest="coalesce_window_frames",
                   help="Adjacent event-frames within this gap are merged into "
                        "one block. Default = round(fps x window-s). Set to 1 "
                        "to disable coalescing — useful when the trajectory "
                        "has long NaN spans that would otherwise swallow the "
                        "real merge events.")
    p.add_argument("--max-event-block-s", type=float, default=None,
                   dest="max_event_block_s",
                   help="Skip event blocks longer than this many seconds "
                        "(default = 5 x window-s). Coalesced contact / NaN "
                        "spans this long bridge the Kalman filter through "
                        "too many predict-only steps and the cost ratio "
                        "becomes numerical noise.")
    p.add_argument("--max-jump-mm", type=float, default=None,
                   dest="max_jump_mm",
                   help="Refuse any non-identity permutation whose worst-case "
                        "track jump from last pre-event position to first "
                        "post-event position exceeds this distance (default = "
                        "2 x contact-mm). Catches obvious teleports the "
                        "Kalman cost can rank favourably under degenerate "
                        "covariance.")
    p.add_argument("--fingerprint", type=Path, default=None,
                   help="Path to a pylace_fingerprints.npz sidecar produced by "
                        "pylace-fingerprint. Default: auto-detect "
                        "<video>.pylace_fingerprints.npz next to the "
                        "trajectory's video.")
    p.add_argument("--appearance-weight", type=float, default=1.0,
                   dest="appearance_weight",
                   help="Weight on the pose-normalised intensity-patch RMSE "
                        "term in the permutation cost. Set to 0 to disable. "
                        "Default 1.0.")
    p.add_argument("--axis-ratio-weight", type=float, default=1.0,
                   dest="axis_ratio_weight",
                   help="Weight on the axis-ratio (major/minor) continuity "
                        "term. Default 1.0.")
    p.add_argument("--area-weight", type=float, default=1.0,
                   dest="area_weight",
                   help="Weight on the area continuity term. Default 1.0.")
    p.add_argument("--kalman-q-pos", type=float, default=DEFAULT_KALMAN_Q_POS,
                   dest="kalman_q_pos",
                   help=f"Audit Kalman position-drift std (px). "
                        f"Default {DEFAULT_KALMAN_Q_POS}.")
    p.add_argument("--kalman-q-vel", type=float, default=DEFAULT_KALMAN_Q_VEL,
                   dest="kalman_q_vel",
                   help=f"Audit Kalman velocity-jitter std (px/frame). "
                        f"Default {DEFAULT_KALMAN_Q_VEL}.")
    p.add_argument("--kalman-r-pos", type=float, default=DEFAULT_KALMAN_R_POS,
                   dest="kalman_r_pos",
                   help=f"Audit Kalman measurement-noise std (px). "
                        f"Default {DEFAULT_KALMAN_R_POS}.")
    p.add_argument("--kalman-initial-v-std", type=float,
                   default=DEFAULT_KALMAN_INITIAL_V_STD,
                   dest="kalman_initial_v_std",
                   help=f"Audit Kalman initial velocity prior (px/frame). "
                        f"Default {DEFAULT_KALMAN_INITIAL_V_STD}.")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
