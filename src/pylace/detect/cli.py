"""Command-line entry point for ``pylace-detect``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pylace.annotator.sidecar import default_sidecar_path, read_sidecar
from pylace.detect.frame import (
    DEFAULT_MAX_AREA,
    DEFAULT_MIN_AREA,
    DEFAULT_MORPH_KERNEL,
    DEFAULT_THRESHOLD,
)
from pylace.detect.pipeline import run_detection, write_detections_csv

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
    results = run_detection(
        args.video, sidecar,
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        morph_kernel=args.morph_kernel,
        every=args.every,
        max_frames=args.max_frames,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    rows = write_detections_csv(results, sidecar, out_path)
    print(f"Wrote {rows} detection rows.")
    return 0


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
        "--out", type=Path, default=None,
        help="CSV output path (default: <video>.pylace_detections.csv).",
    )
    p.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help=f"Foreground intensity threshold (default: {DEFAULT_THRESHOLD}).",
    )
    p.add_argument(
        "--min-area", type=int, default=DEFAULT_MIN_AREA, dest="min_area",
        help=f"Minimum blob area in pixels (default: {DEFAULT_MIN_AREA}).",
    )
    p.add_argument(
        "--max-area", type=int, default=DEFAULT_MAX_AREA, dest="max_area",
        help=f"Maximum blob area in pixels (default: {DEFAULT_MAX_AREA}).",
    )
    p.add_argument(
        "--morph-kernel", type=int, default=DEFAULT_MORPH_KERNEL,
        dest="morph_kernel",
        help="Side length of the morphology kernel; set 0 to skip morphology.",
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
