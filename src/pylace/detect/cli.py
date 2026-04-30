"""Command-line entry point for ``pylace-detect``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pylace.annotator.sidecar import default_sidecar_path, read_sidecar
from pylace.detect.frame import (
    DEFAULT_DILATE_ITERS,
    DEFAULT_ERODE_ITERS,
    DEFAULT_MAX_AREA,
    DEFAULT_MIN_AREA,
    DEFAULT_MORPH_KERNEL,
    DEFAULT_THRESHOLD,
)
from pylace.detect.chain import ChainSplitter
from pylace.detect.pipeline import run_detection, write_detections_csv
from pylace.roi.mask import build_combined_mask, build_split_masks
from pylace.roi.sidecar import default_rois_path, read_rois
from pylace.tracking.constants import (
    DEFAULT_MAX_DISTANCE_PX,
    DEFAULT_MAX_MISSED_FRAMES,
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
    start_frame = (
        int(round(args.start * fps)) if args.start is not None else 0
    )
    end_frame = (
        int(round(args.end * fps)) if args.end is not None else None
    )
    if end_frame is not None and end_frame <= start_frame:
        print("--end must be after --start.", file=sys.stderr)
        return 2

    print(f"pylace-detect: {args.video.name} -> {out_path.name}")
    if args.start is not None or args.end is not None:
        end_repr = f"{args.end:.2f}s" if args.end is not None else "end"
        start_repr = f"{args.start:.2f}s" if args.start is not None else "start"
        print(f"  window: {start_repr} -> {end_repr}  (frames {start_frame}..{end_frame})")
    print(
        f"  detection: threshold={detection['threshold']} "
        f"min_area={detection['min_area']} max_area={detection['max_area']} "
        f"morph={detection['morph_kernel']} "
        f"dilate={detection['dilate_iters']} erode={detection['erode_iters']}"
    )
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
    else:
        print("  tracking: disabled (track_id falls back to per-frame index)")
    if tracking["chain_split_enabled"]:
        if tracking["expected_animal_area_px"] is not None:
            print(
                f"  chain split: expected_animal_area="
                f"{tracking['expected_animal_area_px']:.0f} px² "
                "(threshold 1.5×)",
            )
        else:
            print(
                "  chain split: enabled, expected area auto-learned from "
                "first 50 frames",
            )
    else:
        print("  chain split: disabled")
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
    def gen():
        for label, mask in plan:
            tracker = (
                Tracker(
                    max_distance_px=tracking["max_distance_px"],
                    max_missed_frames=tracking["max_missed_frames"],
                    n_animals=tracking["n_animals"],
                )
                if tracking["enabled"]
                else None
            )
            chain_splitter = (
                ChainSplitter(
                    expected_animal_area_px=tracking["expected_animal_area_px"],
                )
                if tracking["chain_split_enabled"]
                else None
            )
            for fr in run_detection(
                video, sidecar,
                threshold=detection["threshold"],
                min_area=detection["min_area"],
                max_area=detection["max_area"],
                morph_kernel=detection["morph_kernel"],
                dilate_iters=detection["dilate_iters"],
                erode_iters=detection["erode_iters"],
                every=every,
                max_frames=max_frames,
                start_frame=start_frame,
                end_frame=end_frame,
                background=bg,
                extra_mask=mask,
                tracker=tracker,
                chain_splitter=chain_splitter,
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

    return write_detections_csv(gen(), sidecar, out_path)


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
        }
        print(f"Loaded tuned params from {candidate.name}.")

    overrides = {
        "threshold": args.threshold,
        "min_area": args.min_area,
        "max_area": args.max_area,
        "morph_kernel": args.morph_kernel,
        "dilate_iters": args.dilate_iters,
        "erode_iters": args.erode_iters,
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
        help="Disable chain splitting even if --expected-animal-area is set.",
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
