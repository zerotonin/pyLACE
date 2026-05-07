"""Command-line entry point for ``pylace-detect``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from pylace.annotator.sidecar import default_sidecar_path, read_sidecar
from pylace.detect.frame import (
    DEFAULT_DILATE_ITERS,
    DEFAULT_ERODE_ITERS,
    DEFAULT_MAX_AREA,
    DEFAULT_MAX_AXIS_RATIO,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_SOLIDITY,
    DEFAULT_MORPH_KERNEL,
    DEFAULT_THRESHOLD,
)
from pylace.detect.chain import ChainSplitter
from pylace.detect.hough_rescue import HoughRescuer
from pylace.detect.watershed import WatershedSplitter
from pylace.detect.pipeline import run_detection, write_detections_csv
from pylace.roi.mask import build_combined_mask, build_split_masks
from pylace.roi.sidecar import default_rois_path, read_rois
from pylace.tracking.constants import (
    DEFAULT_AREA_COST_WEIGHT,
    DEFAULT_KALMAN_INITIAL_V_STD,
    DEFAULT_KALMAN_Q_POS,
    DEFAULT_KALMAN_Q_VEL,
    DEFAULT_KALMAN_R_POS,
    DEFAULT_MAX_DISTANCE_PX,
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_PERIMETER_COST_WEIGHT,
)
from pylace.tracking.tracks import Tracker
from pylace.tune.params import default_params_path, read_params

DETECTIONS_SUFFIX = ".pylace_detections.csv"


def main(argv: list[str] | None = None) -> int:
    """Top-level CLI. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 2

    sidecar_path = args.sidecar if args.sidecar else default_sidecar_path(args.video)
    if not sidecar_path.exists():
        print(f"Sidecar not found: {sidecar_path}", file=sys.stderr)
        print("Run pylace-annotate first.", file=sys.stderr)
        return 2
    sidecar = read_sidecar(sidecar_path)

    detection, background, tracking = _resolve_tuning_params(args)

    out_path = (
        args.out
        if args.out is not None
        else args.video.with_name(args.video.name + DETECTIONS_SUFFIX)
    )

    fps = sidecar.video.fps
    # CLI flags override; sidecar.trim provides the default; otherwise
    # whole video.
    trim = sidecar.trim
    start_s = (
        args.start if args.start is not None
        else (trim.start_s if trim is not None else None)
    )
    end_s = (
        args.end if args.end is not None
        else (trim.end_s if trim is not None else None)
    )
    start_frame = int(round(start_s * fps)) if start_s is not None else 0
    end_frame = int(round(end_s * fps)) if end_s is not None else None
    if end_frame is not None and end_frame <= start_frame:
        print("--end must be after --start.", file=sys.stderr)
        return 2

    print(f"pylace-detect: {args.video.name} -> {out_path.name}")
    if start_s is not None or end_s is not None:
        source = (
            "CLI flag" if (args.start is not None or args.end is not None)
            else "sidecar.trim"
        )
        end_repr = f"{end_s:.2f}s" if end_s is not None else "end"
        start_repr = f"{start_s:.2f}s" if start_s is not None else "start"
        print(
            f"  window: {start_repr} -> {end_repr}  "
            f"(frames {start_frame}..{end_frame})  [{source}]",
        )
    print(
        f"  detection: threshold={detection['threshold']} "
        f"min_area={detection['min_area']} max_area={detection['max_area']} "
        f"morph={detection['morph_kernel']} "
        f"dilate={detection['dilate_iters']} erode={detection['erode_iters']}"
    )
    quality_bits: list[str] = []
    if detection["min_solidity"] > 0.0:
        quality_bits.append(f"min_solidity={detection['min_solidity']:.2f}")
    if detection["max_axis_ratio"] > 0.0:
        quality_bits.append(f"max_axis_ratio={detection['max_axis_ratio']:.2f}")
    if quality_bits:
        print(f"  shape filter: {' '.join(quality_bits)}")
    else:
        print("  shape filter: disabled")
    print(
        f"  background: n_frames={background['n_frames']} "
        f"start_frac={background['start_frac']} end_frac={background['end_frac']} "
        f"polarity={background['polarity']}"
    )
    if tracking["enabled"]:
        if tracking["n_animals"] is not None:
            print(
                f"  tracking: fixed-N mode, N={tracking['n_animals']} "
                "(max_distance ignored, max_missed ignored)",
            )
        else:
            print(
                f"  tracking: dynamic-N "
                f"max_distance={tracking['max_distance_px']:.1f} px "
                f"max_missed={tracking['max_missed_frames']}",
            )
        print(
            f"  cost weights: area={tracking['area_cost_weight']:.4f} "
            f"perimeter={tracking['perimeter_cost_weight']:.4f}",
        )
        print(
            f"  kalman: q_pos={tracking['kalman_q_pos']:.2f} "
            f"q_vel={tracking['kalman_q_vel']:.2f} "
            f"r_pos={tracking['kalman_r_pos']:.2f} "
            f"init_v_std={tracking['kalman_initial_v_std']:.1f}",
        )
    else:
        print("  tracking: disabled (track_id falls back to per-frame index)")
    if tracking["chain_split_enabled"]:
        mode = tracking.get("splitter_mode", "watershed")
        extra = ""
        if mode == "watershed":
            extra = (
                f"  (watershed, peak_distance="
                f"{tracking.get('watershed_peak_distance_px', 8)} px)"
            )
        if tracking["expected_animal_area_px"] is not None:
            print(
                f"  splitter ({mode}): expected_animal_area="
                f"{tracking['expected_animal_area_px']:.0f} px² "
                f"(threshold 1.5×){extra}",
            )
        else:
            print(
                f"  splitter ({mode}): enabled, expected area "
                f"auto-learned from first 50 frames{extra}",
            )
    else:
        print("  splitter: disabled")
    if tracking.get("hough_rescue_enabled", True) and tracking.get("n_animals"):
        print(
            f"  hough rescue: enabled (target N={tracking['n_animals']}, "
            f"area_tol=±{tracking.get('hough_area_tolerance', 0.5):.2f})",
        )
    else:
        print("  hough rescue: disabled")
    from pylace.detect.background import load_or_build_background_pair

    bg, _trail, bg_source = load_or_build_background_pair(
        args.video,
        n_frames=background["n_frames"],
        start_frac=background["start_frac"],
        end_frac=background["end_frac"],
        force_rebuild=args.rebuild_background,
        polarity=background["polarity"],
    )
    print(f"  background: {bg_source}")

    plan = _resolve_roi_plan(args, sidecar)
    for label, mask in plan:
        if mask is not None:
            kept = int(mask.sum())
            total = int(mask.size)
            print(
                f"  rois[{label}]: {kept}/{total} pixels "
                f"({100.0 * kept / total:.1f}%) inside ROI mask",
            )

    rows = _run_plan(
        plan=plan,
        video=args.video,
        sidecar=sidecar,
        out_path=out_path,
        bg=bg,
        detection=detection,
        tracking=tracking,
        start_frame=start_frame,
        end_frame=end_frame,
        every=args.every,
        max_frames=args.max_frames,
    )
    print(f"Wrote {rows} detection rows.")
    return 0


def _run_plan(
    *, plan, video, sidecar, out_path, bg, detection, tracking,
    start_frame, end_frame, every, max_frames,
) -> int:
    """Run detection for each (label, mask) entry in the ROI plan, into one CSV."""
    total = _estimate_total_frames(
        video=video, start_frame=start_frame, end_frame=end_frame,
        every=every, max_frames=max_frames, n_rois=len(plan),
    )

    def gen():
        for label, mask in plan:
            tracker = (
                Tracker(
                    max_distance_px=tracking["max_distance_px"],
                    max_missed_frames=tracking["max_missed_frames"],
                    n_animals=tracking["n_animals"],
                    area_cost_weight=tracking["area_cost_weight"],
                    perimeter_cost_weight=tracking["perimeter_cost_weight"],
                    kalman_q_pos=tracking["kalman_q_pos"],
                    kalman_q_vel=tracking["kalman_q_vel"],
                    kalman_r_pos=tracking["kalman_r_pos"],
                    kalman_initial_v_std=tracking["kalman_initial_v_std"],
                )
                if tracking["enabled"]
                else None
            )
            chain_splitter = (
                _build_splitter(tracking)
                if tracking["chain_split_enabled"]
                else None
            )
            hough_rescuer = _build_hough_rescuer(tracking)
            for fr in run_detection(
                video, sidecar,
                threshold=detection["threshold"],
                min_area=detection["min_area"],
                max_area=detection["max_area"],
                morph_kernel=detection["morph_kernel"],
                dilate_iters=detection["dilate_iters"],
                erode_iters=detection["erode_iters"],
                min_solidity=detection["min_solidity"],
                max_axis_ratio=detection["max_axis_ratio"],
                every=every,
                max_frames=max_frames,
                start_frame=start_frame,
                end_frame=end_frame,
                background=bg,
                extra_mask=mask,
                tracker=tracker,
                chain_splitter=chain_splitter,
                hough_rescuer=hough_rescuer,
            ):
                fr.roi_label = label
                yield fr
            if (
                chain_splitter is not None
                and chain_splitter.expected_animal_area_px is not None
                and tracking["expected_animal_area_px"] is None
            ):
                print(
                    f"  chain split [{label}]: learned expected_animal_area="
                    f"{chain_splitter.expected_animal_area_px:.0f} px²",
                )

    progress = tqdm(
        gen(),
        total=total,
        desc="Detecting",
        unit="frame",
        smoothing=0.05,
        dynamic_ncols=True,
    )
    return write_detections_csv(progress, sidecar, out_path)


def _build_hough_rescuer(tracking: dict):
    """Build a HoughRescuer if enabled and we have a known target N."""
    if not tracking.get("hough_rescue_enabled", True):
        return None
    n_animals = tracking.get("n_animals")
    if n_animals is None:
        # Hough rescue only makes sense in fixed-N mode — without N
        # there's no way to know the per-frame deficit.
        return None
    return HoughRescuer(
        target_n=int(n_animals),
        expected_animal_area_px=tracking.get("expected_animal_area_px"),
        area_tolerance=tracking.get("hough_area_tolerance", 0.50),
    )


def _build_splitter(tracking: dict):
    """Pick the splitter implementation honouring the ``splitter_mode`` knob."""
    mode = tracking.get("splitter_mode", "watershed")
    if mode == "chain":
        return ChainSplitter(
            expected_animal_area_px=tracking["expected_animal_area_px"],
        )
    if mode == "watershed":
        return WatershedSplitter(
            expected_animal_area_px=tracking["expected_animal_area_px"],
            peak_min_distance_px=tracking.get("watershed_peak_distance_px", 8),
        )
    raise ValueError(
        f"Unknown splitter_mode: {mode!r} (expected 'watershed' or 'chain').",
    )


def _estimate_total_frames(
    *,
    video: Path,
    start_frame: int,
    end_frame: int | None,
    every: int,
    max_frames: int | None,
    n_rois: int,
) -> int | None:
    """Estimate processed-frame count for tqdm; ``None`` if the video can't be probed."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if n_total <= 0:
        return None
    real_end = end_frame if end_frame is not None else n_total
    if real_end <= start_frame:
        return None
    per_roi = (real_end - start_frame + every - 1) // every
    if max_frames is not None:
        per_roi = min(per_roi, max_frames)
    return per_roi * max(1, n_rois)


def _resolve_tuning_params(args: argparse.Namespace) -> tuple[dict, dict]:
    """Resolve detection + background params from --params, sibling, or CLI flags.

    Precedence (highest first):
      1. Per-flag CLI arguments that the user explicitly provided
         (detected via the parser's ``default=None`` sentinel).
      2. The tuning-params JSON specified by --params, or the conventional
         sibling ``<video>.pylace_detect_params.json`` if it exists.
      3. The hard-coded detector defaults.
    """
    detection: dict = {
        "threshold": DEFAULT_THRESHOLD,
        "min_area": DEFAULT_MIN_AREA,
        "max_area": DEFAULT_MAX_AREA,
        "morph_kernel": DEFAULT_MORPH_KERNEL,
        "dilate_iters": DEFAULT_DILATE_ITERS,
        "erode_iters": DEFAULT_ERODE_ITERS,
        "min_solidity": DEFAULT_MIN_SOLIDITY,
        "max_axis_ratio": DEFAULT_MAX_AXIS_RATIO,
    }
    background: dict = {
        "n_frames": 50,
        "start_frac": 0.1,
        "end_frac": 0.9,
        "polarity": "dark_on_light",
    }
    tracking: dict = {
        "enabled": True,
        "max_distance_px": DEFAULT_MAX_DISTANCE_PX,
        "max_missed_frames": DEFAULT_MAX_MISSED_FRAMES,
        "n_animals": None,
        "expected_animal_area_px": None,
        "chain_split_enabled": True,
        "splitter_mode": "watershed",
        "watershed_peak_distance_px": 8,
        "hough_rescue_enabled": True,
        "hough_area_tolerance": 0.50,
        "area_cost_weight": DEFAULT_AREA_COST_WEIGHT,
        "perimeter_cost_weight": DEFAULT_PERIMETER_COST_WEIGHT,
        "kalman_q_pos": DEFAULT_KALMAN_Q_POS,
        "kalman_q_vel": DEFAULT_KALMAN_Q_VEL,
        "kalman_r_pos": DEFAULT_KALMAN_R_POS,
        "kalman_initial_v_std": DEFAULT_KALMAN_INITIAL_V_STD,
    }

    candidate = args.params if args.params else default_params_path(args.video)
    if candidate.exists():
        tp, _ = read_params(candidate)
        detection.update(
            threshold=tp.detection.threshold,
            min_area=tp.detection.min_area,
            max_area=tp.detection.max_area,
            morph_kernel=tp.detection.morph_kernel,
            dilate_iters=tp.detection.dilate_iters,
            erode_iters=tp.detection.erode_iters,
            min_solidity=tp.detection.min_solidity,
            max_axis_ratio=tp.detection.max_axis_ratio,
        )
        background = {
            "n_frames": tp.background.n_frames,
            "start_frac": tp.background.start_frac,
            "end_frac": tp.background.end_frac,
            "polarity": tp.background.polarity,
        }
        tracking = {
            "enabled": tp.tracking.enabled,
            "max_distance_px": tp.tracking.max_distance_px,
            "max_missed_frames": tp.tracking.max_missed_frames,
            "n_animals": tp.tracking.n_animals,
            "expected_animal_area_px": tp.tracking.expected_animal_area_px,
            "chain_split_enabled": True,
            "splitter_mode": tp.tracking.splitter_mode,
            "watershed_peak_distance_px": tp.tracking.watershed_peak_distance_px,
            "hough_rescue_enabled": tp.tracking.hough_rescue_enabled,
            "hough_area_tolerance": tp.tracking.hough_area_tolerance,
            "area_cost_weight": tp.tracking.area_cost_weight,
            "perimeter_cost_weight": tp.tracking.perimeter_cost_weight,
            "kalman_q_pos": tp.tracking.kalman_q_pos,
            "kalman_q_vel": tp.tracking.kalman_q_vel,
            "kalman_r_pos": tp.tracking.kalman_r_pos,
            "kalman_initial_v_std": tp.tracking.kalman_initial_v_std,
        }
        print(f"Loaded tuned params from {candidate.name}.")

    overrides = {
        "threshold": args.threshold,
        "min_area": args.min_area,
        "max_area": args.max_area,
        "morph_kernel": args.morph_kernel,
        "dilate_iters": args.dilate_iters,
        "erode_iters": args.erode_iters,
        "min_solidity": args.min_solidity,
        "max_axis_ratio": args.max_axis_ratio,
    }
    for key, value in overrides.items():
        if value is not None:
            detection[key] = value

    if args.max_track_distance is not None:
        tracking["max_distance_px"] = args.max_track_distance
    if args.max_missed_frames is not None:
        tracking["max_missed_frames"] = args.max_missed_frames
    if args.no_track:
        tracking["enabled"] = False
    if args.n_animals is not None:
        tracking["n_animals"] = args.n_animals
    if args.expected_animal_area is not None:
        tracking["expected_animal_area_px"] = args.expected_animal_area
    if args.no_chain_split:
        tracking["chain_split_enabled"] = False
    if args.splitter is not None:
        tracking["splitter_mode"] = args.splitter
    if args.watershed_peak_distance is not None:
        tracking["watershed_peak_distance_px"] = args.watershed_peak_distance
    if args.no_hough_rescue:
        tracking["hough_rescue_enabled"] = False
    if args.hough_area_tolerance is not None:
        tracking["hough_area_tolerance"] = args.hough_area_tolerance
    if args.cost_area_weight is not None:
        tracking["area_cost_weight"] = args.cost_area_weight
    if args.cost_perimeter_weight is not None:
        tracking["perimeter_cost_weight"] = args.cost_perimeter_weight
    if args.kalman_q_pos is not None:
        tracking["kalman_q_pos"] = args.kalman_q_pos
    if args.kalman_q_vel is not None:
        tracking["kalman_q_vel"] = args.kalman_q_vel
    if args.kalman_r_pos is not None:
        tracking["kalman_r_pos"] = args.kalman_r_pos
    if args.kalman_initial_v_std is not None:
        tracking["kalman_initial_v_std"] = args.kalman_initial_v_std
    return detection, background, tracking


def _resolve_roi_plan(args: argparse.Namespace, sidecar):
    """Decide what to detect against, returning a list of ``(label, mask)``.

    - ``--no-rois`` or no sidecar: ``[("_merged", None)]`` (use arena mask only).
    - merge mode: ``[("_merged", combined_mask)]``.
    - split mode: one entry per add-ROI, label from ROI.label or ``roi_<i>``.
    """
    from pylace.detect.pipeline import MERGED_ROI_LABEL

    if args.no_rois:
        return [(MERGED_ROI_LABEL, None)]
    candidate = args.rois if args.rois else default_rois_path(args.video)
    if not candidate.exists():
        print(
            f"  rois: no sidecar at {candidate.name}; "
            "running on whole-arena mask (use pylace-roi to add ROIs).",
        )
        return [(MERGED_ROI_LABEL, None)]

    rois_sidecar = read_rois(candidate)
    print(f"Loaded ROI sidecar from {candidate.name} (mode={rois_sidecar.roi_set.mode}).")
    if rois_sidecar.roi_set.mode == "split":
        pairs = build_split_masks(rois_sidecar.roi_set, sidecar.video.frame_size)
        if not pairs:
            print("  split mode but no add-ROIs found; falling back to whole arena.")
            return [(MERGED_ROI_LABEL, None)]
        return pairs
    return [(
        MERGED_ROI_LABEL,
        build_combined_mask(rois_sidecar.roi_set, sidecar.video.frame_size),
    )]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylace-detect",
        description=(
            "Run per-frame blob detection on a video against an arena "
            "sidecar JSON; write a per-detection CSV."
        ),
    )
    p.add_argument("video", type=Path, help="Path to the video file.")
    p.add_argument(
        "--sidecar", type=Path, default=None,
        help="Sidecar path (default: <video>.pylace_arena.json).",
    )
    p.add_argument(
        "--params", type=Path, default=None,
        help=(
            "Tuning-params JSON path. Defaults to <video>.pylace_detect_params.json "
            "if that exists; per-flag CLI arguments still override."
        ),
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="CSV output path (default: <video>.pylace_detections.csv).",
    )
    p.add_argument(
        "--threshold", type=int, default=None,
        help=f"Foreground intensity threshold (default: {DEFAULT_THRESHOLD}).",
    )
    p.add_argument(
        "--min-area", type=int, default=None, dest="min_area",
        help=f"Minimum blob area in pixels (default: {DEFAULT_MIN_AREA}).",
    )
    p.add_argument(
        "--max-area", type=int, default=None, dest="max_area",
        help=f"Maximum blob area in pixels (default: {DEFAULT_MAX_AREA}).",
    )
    p.add_argument(
        "--morph-kernel", type=int, default=None,
        dest="morph_kernel",
        help="Side length of the morphology kernel; set 0 to skip morphology.",
    )
    p.add_argument(
        "--dilate-iters", type=int, default=None, dest="dilate_iters",
        help=(
            "Extra dilation iterations after open+close (default: "
            f"{DEFAULT_DILATE_ITERS}). Useful to merge fragmented blobs."
        ),
    )
    p.add_argument(
        "--erode-iters", type=int, default=None, dest="erode_iters",
        help=(
            "Extra erosion iterations after dilation (default: "
            f"{DEFAULT_ERODE_ITERS}). Useful to trim peripheral noise."
        ),
    )
    p.add_argument(
        "--every", type=int, default=1,
        help="Process every Nth frame (default: 1, i.e. every frame).",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, dest="max_frames",
        help="Stop after this many processed frames.",
    )
    p.add_argument(
        "--start", type=_parse_time_spec, default=None,
        help="Start time as seconds, MM:SS, or HH:MM:SS (default: video start).",
    )
    p.add_argument(
        "--end", type=_parse_time_spec, default=None,
        help="End time as seconds, MM:SS, or HH:MM:SS (default: video end).",
    )
    p.add_argument(
        "--rebuild-background", action="store_true",
        dest="rebuild_background",
        help=(
            "Force max-projection recompute even if a "
            "<video>.pylace_background.png sidecar already exists."
        ),
    )
    p.add_argument(
        "--rois", type=Path, default=None,
        help=(
            "ROI sidecar path. Defaults to <video>.pylace_rois.json if it "
            "exists; the combined ROI mask is intersected with the arena "
            "mask before detection runs."
        ),
    )
    p.add_argument(
        "--no-rois", action="store_true", dest="no_rois",
        help=(
            "Ignore any ROI sidecar and run detection on the whole arena."
        ),
    )
    p.add_argument(
        "--max-track-distance", type=float, default=None,
        dest="max_track_distance",
        help=(
            f"Hungarian-tracker rejection threshold in pixels "
            f"(default: {DEFAULT_MAX_DISTANCE_PX})."
        ),
    )
    p.add_argument(
        "--max-missed-frames", type=int, default=None,
        dest="max_missed_frames",
        help=(
            f"Tracks unmatched for more than this many frames are retired "
            f"(default: {DEFAULT_MAX_MISSED_FRAMES})."
        ),
    )
    p.add_argument(
        "--no-track", action="store_true", dest="no_track",
        help=(
            "Disable identity tracking; the track_id column falls back to "
            "the per-frame detection index."
        ),
    )
    p.add_argument(
        "--n-animals", type=int, default=None, dest="n_animals",
        help=(
            "Number of animals known to be present; switches the tracker "
            "to fixed-N mode. Tracks are born up to N then never retired "
            "or replaced; max-track-distance is ignored. Recommended for "
            "the LACE-paper workflow."
        ),
    )
    p.add_argument(
        "--expected-animal-area", type=float, default=None,
        dest="expected_animal_area",
        help=(
            "Expected single-animal area in px². Enables LACE-paper chain "
            "splitting: any contour above 1.5× this area is cut "
            "perpendicular to its major axis at the centroid and refit "
            "as two ellipses. Auto-learned from the median of the first "
            "50 frames if omitted."
        ),
    )
    p.add_argument(
        "--no-chain-split", action="store_true", dest="no_chain_split",
        help="Disable splitting of merged contours entirely.",
    )
    p.add_argument(
        "--splitter", choices=("watershed", "chain"), default=None, dest="splitter",
        help=(
            "Algorithm used to split a merged-fly blob whose area exceeds "
            "1.5× expected. 'watershed' (default) seeds a distance-transform "
            "watershed at local peaks — handles N≥3 contacts and side-by-"
            "side merges. 'chain' is the legacy LACE-paper rule: cut "
            "perpendicular to the major axis at the centroid."
        ),
    )
    p.add_argument(
        "--watershed-peak-distance", type=int, default=None,
        dest="watershed_peak_distance",
        help=(
            "Watershed local-maximum suppression radius in pixels. Set well "
            "below the inter-fly centroid distance and above the half-width "
            "of a single fly. Default 8."
        ),
    )
    p.add_argument(
        "--no-hough-rescue", action="store_true", dest="no_hough_rescue",
        help=(
            "Disable the LACE-paper Hough rescue. When enabled (the default "
            "in fixed-N mode), under-counted frames get extra ellipse "
            "candidates fitted to contour sub-arcs before Hungarian runs."
        ),
    )
    p.add_argument(
        "--hough-area-tolerance", type=float, default=None,
        dest="hough_area_tolerance",
        help=(
            "± fraction around the expected animal area that a rescue "
            "candidate's fitted ellipse must fall within. Default 0.5 "
            "(half-to-1.5× expected)."
        ),
    )
    p.add_argument(
        "--cost-area-weight", type=float, default=None,
        dest="cost_area_weight",
        help=(
            "Weight on |Δarea_px| in the Hungarian cost matrix. 0 disables "
            "(legacy distance-only). Try 0.02 to make a 100 px² area "
            "difference cost ~2 pixels-equivalent of distance."
        ),
    )
    p.add_argument(
        "--cost-perimeter-weight", type=float, default=None,
        dest="cost_perimeter_weight",
        help=(
            "Weight on |Δperimeter_px| in the Hungarian cost matrix. "
            "0 disables. Try 0.5 (1 px-equivalent per 2 px perimeter "
            "difference)."
        ),
    )
    p.add_argument(
        "--min-solidity", type=float, default=None,
        dest="min_solidity",
        help=(
            "Reject contours whose solidity (area / convex-hull area) "
            "falls below this. 0 disables. A real fly is ≈ 0.9; "
            "shadows / streaks are < 0.7 — start with 0.85."
        ),
    )
    p.add_argument(
        "--max-axis-ratio", type=float, default=None,
        dest="max_axis_ratio",
        help=(
            "Reject contours whose major / minor axis ratio exceeds this. "
            "0 disables. A real fly is ≈ 2–3; thin streak shadows are "
            "5+ — start with 5.0."
        ),
    )
    p.add_argument(
        "--kalman-q-pos", type=float, default=None, dest="kalman_q_pos",
        help=(
            f"Kalman per-frame position-drift std (px). Larger = more "
            f"responsive but less smooth. Default {DEFAULT_KALMAN_Q_POS}."
        ),
    )
    p.add_argument(
        "--kalman-q-vel", type=float, default=None, dest="kalman_q_vel",
        help=(
            f"Kalman per-frame velocity-jitter std (px/frame). Roughly the "
            f"rms acceleration the filter expects. Default {DEFAULT_KALMAN_Q_VEL}."
        ),
    )
    p.add_argument(
        "--kalman-r-pos", type=float, default=None, dest="kalman_r_pos",
        help=(
            f"Kalman per-axis measurement-noise std (px). The detector's "
            f"centroid is accurate to ~ 1 px on a 1080p arena view; raise "
            f"this if your detections look noisier. Default {DEFAULT_KALMAN_R_POS}."
        ),
    )
    p.add_argument(
        "--kalman-initial-v-std", type=float, default=None,
        dest="kalman_initial_v_std",
        help=(
            f"Initial velocity prior at track birth (px/frame std). "
            f"Generous enough to cover the fastest plausible per-frame "
            f"motion. Default {DEFAULT_KALMAN_INITIAL_V_STD}."
        ),
    )
    return p


def _parse_time_spec(s: str) -> float:
    """Convert ``'90'`` / ``'15:00'`` / ``'0:15:00'`` to seconds."""
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Bad time spec: {s!r}") from exc
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * 60.0 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
    raise argparse.ArgumentTypeError(f"Bad time spec: {s!r}")


if __name__ == "__main__":
    raise SystemExit(main())
