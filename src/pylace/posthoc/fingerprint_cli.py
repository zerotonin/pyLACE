# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.fingerprint_cli                                ║
# ║  « pylace-fingerprint: per-detection pose-normalised patches »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Reads a video + its pylace_detections.csv and writes a          ║
# ║  ``<video>.pylace_fingerprints.npz`` with one pose-normalised    ║
# ║  intensity patch per detection plus a per-frame "is this a       ║
# ║  trustworthy frame for medians" boolean. The audit consumes this ║
# ║  file when scoring permutations after a merge.                   ║
# ║                                                                  ║
# ║  Output layout (all 1-D arrays parallel to one another):         ║
# ║    frame_idx       (N,)  int32                                   ║
# ║    track_id        (N,)  int32   — the detection's tracker ID    ║
# ║    patch           (N, h, w) uint8                               ║
# ║    is_confident    (N,)  bool    — frame-level confidence flag   ║
# ║    cx_px, cy_px, area_px, major_axis_px, minor_axis_px,          ║
# ║    orientation_deg                                               ║
# ║                                                                  ║
# ║  Plus scalars: patch_h, patch_w, n_animals, expected_area_px,    ║
# ║                contact_threshold_px, video_sha256_hex.           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""``pylace-fingerprint``: extract pose-normalised intensity patches per detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from pylace.annotator.sidecar import default_sidecar_path, read_sidecar
from pylace.posthoc.appearance import (
    DEFAULT_CONFIDENT_AREA_TOL,
    DEFAULT_CONFIDENT_SEPARATION_FACTOR,
    DEFAULT_PATCH_H,
    DEFAULT_PATCH_PAD_FACTOR,
    DEFAULT_PATCH_W,
    extract_patch,
    is_confident_frame,
)
from pylace.posthoc.io import read_detections, video_path_from_trajectory
from pylace.tune.params import (
    TuningParamsSchemaError,
    default_params_path,
    read_params,
)

FINGERPRINT_SUFFIX = ".pylace_fingerprints.npz"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.detections.exists():
        print(f"Detections CSV not found: {args.detections}", file=sys.stderr)
        return 2

    video_path = (
        args.video if args.video is not None
        else video_path_from_trajectory(args.detections)
    )
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 2

    sidecar_path = (
        args.sidecar if args.sidecar is not None
        else default_sidecar_path(video_path)
    )
    if not sidecar_path.exists():
        print(f"Arena sidecar not found: {sidecar_path}", file=sys.stderr)
        return 2
    sidecar = read_sidecar(sidecar_path)

    params_path = (
        args.params if args.params is not None
        else default_params_path(video_path)
    )
    n_animals, expected_area_px = _resolve_pipeline_params(
        params_path, args.n_animals, args.expected_area,
    )

    pix_per_mm = float(sidecar.calibration.pixel_distance / sidecar.calibration.physical_mm)
    contact_threshold_px = float(args.contact_mm) * pix_per_mm
    min_pairwise_distance_px = (
        contact_threshold_px * DEFAULT_CONFIDENT_SEPARATION_FACTOR
    )

    out_path = (
        args.out if args.out is not None
        else video_path.with_name(video_path.name + FINGERPRINT_SUFFIX)
    )

    print(f"pylace-fingerprint: {video_path.name} -> {out_path.name}")
    print(
        f"  patch={args.patch_h}x{args.patch_w}  "
        f"n_animals={n_animals if n_animals else 'auto'}  "
        f"expected_area={expected_area_px:.0f} px²  "
        f"contact={args.contact_mm:.2f} mm  "
        f"min_pair_sep={min_pairwise_distance_px:.1f} px",
    )

    df = read_detections(args.detections)
    print(f"  loaded {len(df)} detection rows across "
          f"{df['frame_idx'].nunique()} frames")

    # If n_animals was not provided, estimate from the CSV (mode of
    # per-frame counts).
    if n_animals is None:
        per_frame = df.groupby("frame_idx").size()
        n_animals = int(per_frame.mode().iloc[0])
        print(f"  estimated n_animals={n_animals} from detections")

    # If expected_area_px was not provided, use the median of all
    # detection areas — a robust estimator that survives the few
    # outsized merged blobs.
    if expected_area_px <= 0:
        expected_area_px = float(df["area_px"].median())
        print(f"  estimated expected_area_px={expected_area_px:.0f} from detections")

    rows = _extract_all(
        video_path=video_path,
        df=df,
        n_animals=n_animals,
        expected_area_px=expected_area_px,
        min_pairwise_distance_px=min_pairwise_distance_px,
        area_tol=args.area_tol,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        pad_factor=args.pad_factor,
    )

    np.savez_compressed(
        out_path,
        frame_idx=rows["frame_idx"].astype(np.int32),
        track_id=rows["track_id"].astype(np.int32),
        patch=rows["patch"],
        is_confident=rows["is_confident"].astype(bool),
        cx_px=rows["cx_px"].astype(np.float32),
        cy_px=rows["cy_px"].astype(np.float32),
        area_px=rows["area_px"].astype(np.float32),
        major_axis_px=rows["major_axis_px"].astype(np.float32),
        minor_axis_px=rows["minor_axis_px"].astype(np.float32),
        orientation_deg=rows["orientation_deg"].astype(np.float32),
        patch_h=np.array(args.patch_h, dtype=np.int32),
        patch_w=np.array(args.patch_w, dtype=np.int32),
        n_animals=np.array(n_animals, dtype=np.int32),
        expected_area_px=np.array(expected_area_px, dtype=np.float32),
        contact_threshold_px=np.array(contact_threshold_px, dtype=np.float32),
        min_pairwise_distance_px=np.array(min_pairwise_distance_px, dtype=np.float32),
    )
    n_conf = int(np.count_nonzero(rows["is_confident"]))
    print(
        f"Wrote {len(rows['frame_idx'])} patches "
        f"({n_conf} on confident frames) to {out_path.name}",
    )
    return 0


def _resolve_pipeline_params(
    params_path: Path,
    cli_n_animals: int | None,
    cli_expected_area: float,
) -> tuple[int | None, float]:
    """Pull n_animals + expected_area from the tuning sidecar, with CLI overrides."""
    n_animals: int | None = cli_n_animals
    expected_area: float = float(cli_expected_area)
    if params_path.exists():
        try:
            params, _ = read_params(params_path)
            tp = params.tracking
            if n_animals is None and tp.n_animals is not None:
                n_animals = int(tp.n_animals)
            if expected_area <= 0 and tp.expected_animal_area_px is not None:
                expected_area = float(tp.expected_animal_area_px)
        except TuningParamsSchemaError:
            pass
    return n_animals, expected_area


def _extract_all(
    *,
    video_path: Path,
    df: pd.DataFrame,
    n_animals: int,
    expected_area_px: float,
    min_pairwise_distance_px: float,
    area_tol: float,
    patch_h: int,
    patch_w: int,
    pad_factor: float,
) -> dict[str, np.ndarray]:
    """Stream through the video once, extracting every per-detection patch."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    df_sorted = df.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
    by_frame: dict[int, pd.DataFrame] = {
        int(idx): grp for idx, grp in df_sorted.groupby("frame_idx", sort=True)
    }
    frame_idx_list = sorted(by_frame.keys())

    n_total = len(df_sorted)
    out_frame_idx = np.empty(n_total, dtype=np.int32)
    out_track_id = np.empty(n_total, dtype=np.int32)
    out_patch = np.empty((n_total, patch_h, patch_w), dtype=np.uint8)
    out_is_conf = np.empty(n_total, dtype=bool)
    out_cx = np.empty(n_total, dtype=np.float32)
    out_cy = np.empty(n_total, dtype=np.float32)
    out_area = np.empty(n_total, dtype=np.float32)
    out_major = np.empty(n_total, dtype=np.float32)
    out_minor = np.empty(n_total, dtype=np.float32)
    out_orient = np.empty(n_total, dtype=np.float32)

    write_idx = 0
    try:
        for frame_idx in tqdm(
            frame_idx_list, desc="Fingerprinting", unit="frame",
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            grp = by_frame[frame_idx]
            cx_arr = grp["cx_px"].to_numpy(dtype=np.float32)
            cy_arr = grp["cy_px"].to_numpy(dtype=np.float32)
            area_arr = grp["area_px"].to_numpy(dtype=np.float32)
            confident = is_confident_frame(
                cx_arr, cy_arr, area_arr,
                n_animals=n_animals,
                min_pairwise_distance_px=min_pairwise_distance_px,
                expected_area_px=expected_area_px,
                area_tol=area_tol,
            )
            for _, row in grp.iterrows():
                patch = extract_patch(
                    frame_gray,
                    cx=float(row["cx_px"]),
                    cy=float(row["cy_px"]),
                    orientation_deg=float(row["orientation_deg"]),
                    major_axis_px=float(row["major_axis_px"]),
                    minor_axis_px=float(row["minor_axis_px"]),
                    patch_h=patch_h,
                    patch_w=patch_w,
                    pad_factor=pad_factor,
                )
                out_frame_idx[write_idx] = int(row["frame_idx"])
                out_track_id[write_idx] = int(row["track_id"])
                out_patch[write_idx] = patch
                out_is_conf[write_idx] = bool(confident)
                out_cx[write_idx] = float(row["cx_px"])
                out_cy[write_idx] = float(row["cy_px"])
                out_area[write_idx] = float(row["area_px"])
                out_major[write_idx] = float(row["major_axis_px"])
                out_minor[write_idx] = float(row["minor_axis_px"])
                out_orient[write_idx] = float(row["orientation_deg"])
                write_idx += 1
    finally:
        cap.release()

    return {
        "frame_idx": out_frame_idx[:write_idx],
        "track_id": out_track_id[:write_idx],
        "patch": out_patch[:write_idx],
        "is_confident": out_is_conf[:write_idx],
        "cx_px": out_cx[:write_idx],
        "cy_px": out_cy[:write_idx],
        "area_px": out_area[:write_idx],
        "major_axis_px": out_major[:write_idx],
        "minor_axis_px": out_minor[:write_idx],
        "orientation_deg": out_orient[:write_idx],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylace-fingerprint",
        description=(
            "Extract pose-normalised intensity patches for each detection in a "
            "pylace_detections.csv. The output sidecar feeds the audit's "
            "appearance-cost term."
        ),
    )
    p.add_argument("detections", type=Path,
                   help="pylace-detect CSV (typically <video>.pylace_detections.csv).")
    p.add_argument("--video", type=Path, default=None,
                   help="Source video (default: derive from detections path).")
    p.add_argument("--sidecar", type=Path, default=None,
                   help="Arena sidecar JSON (default: <video>.pylace_arena.json).")
    p.add_argument("--params", type=Path, default=None,
                   help="Tuning params JSON (default: "
                        "<video>.pylace_detect_params.json).")
    p.add_argument("--out", type=Path, default=None,
                   help=f"Output NPZ (default: <video>{FINGERPRINT_SUFFIX}).")
    p.add_argument("--n-animals", type=int, default=None, dest="n_animals",
                   help="Expected fly count (default: from tuning params or "
                        "auto-estimated from the detections CSV).")
    p.add_argument("--expected-area", type=float, default=0.0,
                   dest="expected_area",
                   help="Expected per-fly area in px² (default: from tuning "
                        "params or auto-estimated from the detections CSV).")
    p.add_argument("--contact-mm", type=float, default=5.0, dest="contact_mm",
                   help="Contact-distance threshold in mm (default 5.0). "
                        f"Confidence requires pairwise separation > "
                        f"{DEFAULT_CONFIDENT_SEPARATION_FACTOR} x contact-mm.")
    p.add_argument("--area-tol", type=float, default=DEFAULT_CONFIDENT_AREA_TOL,
                   dest="area_tol",
                   help=f"Confidence area tolerance fraction (default "
                        f"{DEFAULT_CONFIDENT_AREA_TOL}).")
    p.add_argument("--patch-h", type=int, default=DEFAULT_PATCH_H, dest="patch_h",
                   help=f"Patch height in pixels (default {DEFAULT_PATCH_H}).")
    p.add_argument("--patch-w", type=int, default=DEFAULT_PATCH_W, dest="patch_w",
                   help=f"Patch width in pixels (default {DEFAULT_PATCH_W}).")
    p.add_argument("--pad-factor", type=float, default=DEFAULT_PATCH_PAD_FACTOR,
                   dest="pad_factor",
                   help=f"Source-region size as a multiple of the fitted axes "
                        f"(default {DEFAULT_PATCH_PAD_FACTOR}).")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
