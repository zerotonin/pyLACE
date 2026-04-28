"""JSON sidecar I/O for per-video arena annotations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pylace.annotator.geometry import (
    Arena,
    Calibration,
    Circle,
    Polygon,
    Rectangle,
    WorldFrame,
    shape_name,
)

SCHEMA_VERSION = 1
SIDECAR_SUFFIX = ".pylace_arena.json"


class SidecarSchemaError(ValueError):
    """Raised when a sidecar file cannot be parsed at the current schema version."""


@dataclass
class VideoMeta:
    """Identifying information about the source video."""

    path: str
    sha256: str
    frame_size: tuple[int, int]
    fps: float


@dataclass
class Sidecar:
    """Full annotation payload written next to a video."""

    video: VideoMeta
    arena: Arena
    world_frame: WorldFrame
    calibration: Calibration


def write_sidecar(sidecar: Sidecar, out_path: Path) -> None:
    """Write a sidecar to JSON, overwriting any existing file."""
    out_path.write_text(json.dumps(_sidecar_to_dict(sidecar), indent=2))


def read_sidecar(in_path: Path) -> Sidecar:
    """Load and validate a sidecar JSON written at the current schema version."""
    payload = json.loads(in_path.read_text())
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SidecarSchemaError(
            f"Unsupported sidecar schema_version: {version!r} "
            f"(this build expects {SCHEMA_VERSION})."
        )
    return _sidecar_from_dict(payload)


def video_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> tuple[tuple[int, int], float]:
    """Return ((width, height), fps) for a video file via OpenCV.

    Raises:
        IOError: If the file cannot be opened.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return ((width, height), fps)
    finally:
        cap.release()


def default_sidecar_path(video: Path) -> Path:
    """Sidecar path that lives next to the video, preserving its extension."""
    return video.with_name(video.name + SIDECAR_SUFFIX)


def _sidecar_to_dict(sidecar: Sidecar) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "video": {
            "path": sidecar.video.path,
            "sha256": sidecar.video.sha256,
            "frame_size": list(sidecar.video.frame_size),
            "fps": sidecar.video.fps,
        },
        "arena": _arena_to_dict(sidecar.arena),
        "world_frame": {
            "origin_pixel": list(sidecar.world_frame.origin_pixel),
            "y_axis": sidecar.world_frame.y_axis,
            "x_axis": sidecar.world_frame.x_axis,
        },
        "calibration": {
            "reference_kind": sidecar.calibration.reference_kind,
            "reference_vertices": (
                list(sidecar.calibration.reference_vertices)
                if sidecar.calibration.reference_vertices is not None
                else None
            ),
            "physical_mm": sidecar.calibration.physical_mm,
            "pixel_distance": sidecar.calibration.pixel_distance,
            "mm_per_pixel": sidecar.calibration.mm_per_pixel,
        },
    }


def _sidecar_from_dict(payload: dict) -> Sidecar:
    v = payload["video"]
    w = payload["world_frame"]
    c = payload["calibration"]
    ref = c.get("reference_vertices")
    return Sidecar(
        video=VideoMeta(
            path=v["path"],
            sha256=v["sha256"],
            frame_size=tuple(v["frame_size"]),
            fps=float(v["fps"]),
        ),
        arena=_arena_from_dict(payload["arena"]),
        world_frame=WorldFrame(
            origin_pixel=tuple(w["origin_pixel"]),
            y_axis=w.get("y_axis", "up"),
            x_axis=w.get("x_axis", "right"),
        ),
        calibration=Calibration(
            reference_kind=c["reference_kind"],
            physical_mm=float(c["physical_mm"]),
            pixel_distance=float(c["pixel_distance"]),
            reference_vertices=tuple(ref) if ref is not None else None,
        ),
    )


def _arena_to_dict(arena: Arena) -> dict:
    if isinstance(arena, Circle):
        return {
            "shape": "circle",
            "geometry": {"cx": arena.cx, "cy": arena.cy, "r": arena.r},
        }
    return {
        "shape": shape_name(arena),
        "geometry": {"vertices": [list(v) for v in arena.vertices]},
    }


def _arena_from_dict(d: dict) -> Arena:
    shape = d["shape"]
    g = d["geometry"]
    if shape == "circle":
        return Circle(float(g["cx"]), float(g["cy"]), float(g["r"]))
    if shape == "rectangle":
        return Rectangle([(float(x), float(y)) for x, y in g["vertices"]])
    if shape == "polygon":
        return Polygon([(float(x), float(y)) for x, y in g["vertices"]])
    raise SidecarSchemaError(f"Unknown shape: {shape!r}")
