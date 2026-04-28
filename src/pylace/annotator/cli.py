"""Command-line entry point for ``pylace-annotate``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pylace.annotator.geometry import (
    Arena,
    Calibration,
    Circle,
    Polygon,
    Rectangle,
    WorldFrame,
    edge_length,
)
from pylace.annotator.sidecar import (
    Sidecar,
    VideoMeta,
    default_sidecar_path,
    probe_video,
    video_sha256,
    write_sidecar,
)


def main(argv: list[str] | None = None) -> int:
    """Top-level CLI entry. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 2

    if args.headless:
        return _run_headless(args)
    return _run_gui(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pylace-annotate",
        description="Annotate the arena boundary, world frame, and pix-to-mm "
        "calibration for a single video file.",
    )
    parser.add_argument("video", type=Path, help="Path to the video file.")
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Frame index to display in the GUI (default: 0).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Sidecar output path (default: <video>.pylace_arena.json).",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Skip GUI; build the sidecar from the headless arguments below.",
    )

    g = parser.add_argument_group("headless geometry")
    g.add_argument("--shape", choices=("circle", "rectangle", "polygon"))
    g.add_argument("--cx", type=float)
    g.add_argument("--cy", type=float)
    g.add_argument("--r", type=float)
    g.add_argument(
        "--vertices",
        help="Vertices for rectangle/polygon: 'x1,y1;x2,y2;...'",
    )

    g = parser.add_argument_group("headless world frame")
    g.add_argument("--origin-x", type=float, dest="origin_x")
    g.add_argument("--origin-y", type=float, dest="origin_y")
    g.add_argument(
        "--y-down", action="store_true",
        help="Y axis points down (default: Y up).",
    )
    g.add_argument(
        "--x-left", action="store_true",
        help="X axis points left (default: X right).",
    )

    g = parser.add_argument_group("headless calibration")
    g.add_argument("--diameter-mm", type=float, dest="diameter_mm")
    g.add_argument("--edge-mm", type=float, dest="edge_mm")
    g.add_argument(
        "--edge-vertices", dest="edge_vertices",
        help="Two vertex indices defining the calibration edge: 'i,j'.",
    )
    return parser


def _run_headless(args: argparse.Namespace) -> int:
    arena = _build_arena(args)
    world = _build_world_frame(args)
    cal = _build_calibration(args, arena)

    frame_size, fps = probe_video(args.video)
    video = VideoMeta(
        path=str(args.video),
        sha256=video_sha256(args.video),
        frame_size=frame_size,
        fps=fps,
    )
    sidecar = Sidecar(video=video, arena=arena, world_frame=world, calibration=cal)

    out = args.out or default_sidecar_path(args.video)
    write_sidecar(sidecar, out)
    print(f"Wrote sidecar: {out}")
    return 0


def _run_gui(args: argparse.Namespace) -> int:
    from pylace.annotator.main_window import run

    return run(args.video, args.frame, args.out)


def _build_arena(args: argparse.Namespace) -> Arena:
    if args.shape is None:
        sys.exit("--headless requires --shape circle|rectangle|polygon.")

    if args.shape == "circle":
        if args.cx is None or args.cy is None or args.r is None:
            sys.exit("--shape circle requires --cx, --cy, --r.")
        return Circle(args.cx, args.cy, args.r)

    if not args.vertices:
        sys.exit(f"--shape {args.shape} requires --vertices 'x1,y1;x2,y2;...'.")
    verts = _parse_vertices(args.vertices)
    if args.shape == "rectangle":
        return Rectangle(verts)
    return Polygon(verts)


def _build_world_frame(args: argparse.Namespace) -> WorldFrame:
    if args.origin_x is None or args.origin_y is None:
        sys.exit("--headless requires --origin-x and --origin-y.")
    return WorldFrame(
        origin_pixel=(args.origin_x, args.origin_y),
        y_axis="down" if args.y_down else "up",
        x_axis="left" if args.x_left else "right",
    )


def _build_calibration(args: argparse.Namespace, arena: Arena) -> Calibration:
    if isinstance(arena, Circle) and args.diameter_mm is not None:
        return Calibration(
            reference_kind="diameter",
            physical_mm=args.diameter_mm,
            pixel_distance=2.0 * arena.r,
        )
    if args.edge_mm is not None and args.edge_vertices:
        i, j = (int(s) for s in args.edge_vertices.split(","))
        verts = arena.vertices  # type: ignore[union-attr]
        return Calibration(
            reference_kind="edge",
            physical_mm=args.edge_mm,
            pixel_distance=edge_length(verts, i, j),
            reference_vertices=(i, j),
        )
    sys.exit(
        "--headless requires either --diameter-mm (for circle) or "
        "--edge-mm with --edge-vertices i,j (for rectangle/polygon)."
    )


def _parse_vertices(spec: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for chunk in spec.split(";"):
        parts = chunk.split(",")
        if len(parts) != 2:
            sys.exit(f"Bad vertex spec: {chunk!r} (expected 'x,y').")
        out.append((float(parts[0]), float(parts[1])))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
