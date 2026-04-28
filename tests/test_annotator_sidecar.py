"""Sidecar JSON read / write round-trips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pylace.annotator.geometry import (
    Calibration,
    Circle,
    Polygon,
    Rectangle,
    WorldFrame,
)
from pylace.annotator.sidecar import (
    SCHEMA_VERSION,
    Sidecar,
    SidecarSchemaError,
    VideoMeta,
    default_sidecar_path,
    read_sidecar,
    video_sha256,
    write_sidecar,
)


@pytest.fixture
def video_meta() -> VideoMeta:
    return VideoMeta(
        path="/dev/null/example.mp4",
        sha256="0" * 64,
        frame_size=(640, 480),
        fps=25.0,
    )


@pytest.fixture
def world_frame() -> WorldFrame:
    return WorldFrame(origin_pixel=(100.0, 200.0), y_axis="up", x_axis="right")


def test_circle_sidecar_roundtrips(tmp_path, video_meta, world_frame):
    sc = Sidecar(
        video=video_meta,
        arena=Circle(50.0, 50.0, 25.0),
        world_frame=world_frame,
        calibration=Calibration(
            reference_kind="diameter", physical_mm=10.0, pixel_distance=50.0,
        ),
    )
    out = tmp_path / "side.json"
    write_sidecar(sc, out)
    loaded = read_sidecar(out)
    assert loaded == sc


def test_rectangle_sidecar_roundtrips(tmp_path, video_meta, world_frame):
    rect = Rectangle.from_two_points((10.0, 10.0), (90.0, 60.0))
    sc = Sidecar(
        video=video_meta,
        arena=rect,
        world_frame=world_frame,
        calibration=Calibration(
            reference_kind="edge", physical_mm=40.0, pixel_distance=80.0,
            reference_vertices=(0, 1),
        ),
    )
    out = tmp_path / "side.json"
    write_sidecar(sc, out)
    loaded = read_sidecar(out)
    assert loaded == sc


def test_polygon_sidecar_roundtrips(tmp_path, video_meta, world_frame):
    poly = Polygon([(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0), (5.0, 30.0)])
    sc = Sidecar(
        video=video_meta,
        arena=poly,
        world_frame=world_frame,
        calibration=Calibration(
            reference_kind="edge", physical_mm=20.0, pixel_distance=40.0,
            reference_vertices=(0, 1),
        ),
    )
    out = tmp_path / "side.json"
    write_sidecar(sc, out)
    loaded = read_sidecar(out)
    assert loaded == sc


def test_schema_version_mismatch_raises(tmp_path, video_meta, world_frame):
    sc = Sidecar(
        video=video_meta,
        arena=Circle(0, 0, 1),
        world_frame=world_frame,
        calibration=Calibration(
            reference_kind="diameter", physical_mm=1.0, pixel_distance=1.0,
        ),
    )
    out = tmp_path / "side.json"
    write_sidecar(sc, out)
    payload = json.loads(out.read_text())
    payload["schema_version"] = SCHEMA_VERSION + 99
    out.write_text(json.dumps(payload))
    with pytest.raises(SidecarSchemaError):
        read_sidecar(out)


def test_unknown_shape_raises(tmp_path, video_meta, world_frame):
    sc = Sidecar(
        video=video_meta,
        arena=Circle(0, 0, 1),
        world_frame=world_frame,
        calibration=Calibration(
            reference_kind="diameter", physical_mm=1.0, pixel_distance=1.0,
        ),
    )
    out = tmp_path / "side.json"
    write_sidecar(sc, out)
    payload = json.loads(out.read_text())
    payload["arena"]["shape"] = "trapezoid"
    out.write_text(json.dumps(payload))
    with pytest.raises(SidecarSchemaError):
        read_sidecar(out)


def test_default_sidecar_path_appends_suffix_to_full_name():
    p = Path("/tmp/recording_arena_00.mp4")
    assert default_sidecar_path(p).name == "recording_arena_00.mp4.pylace_arena.json"


def test_video_sha256_is_hex_and_deterministic(tmp_path):
    f = tmp_path / "fake.bin"
    f.write_bytes(b"hello, pylace")
    digest = video_sha256(f)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    assert video_sha256(f) == digest


def test_video_sha256_detects_byte_change(tmp_path):
    f = tmp_path / "fake.bin"
    f.write_bytes(b"hello, pylace")
    a = video_sha256(f)
    f.write_bytes(b"hello, pylace!")
    b = video_sha256(f)
    assert a != b
