"""Trim sidecar field round-trip."""

from __future__ import annotations

from pathlib import Path

from pylace.annotator.geometry import Calibration, Circle, WorldFrame
from pylace.annotator.sidecar import (
    Sidecar,
    Trim,
    VideoMeta,
    read_sidecar,
    write_sidecar,
)


def _toy_sidecar(trim: Trim | None = None) -> Sidecar:
    return Sidecar(
        video=VideoMeta(
            path="/tmp/x.mp4", sha256="0" * 64,
            frame_size=(64, 64), fps=25.0,
        ),
        arena=Circle(cx=32.0, cy=32.0, r=30.0),
        world_frame=WorldFrame(origin_pixel=(32.0, 32.0)),
        calibration=Calibration(
            reference_kind="diameter",
            physical_mm=20.0, pixel_distance=60.0,
        ),
        trim=trim,
    )


def test_sidecar_round_trip_with_trim(tmp_path: Path):
    p = tmp_path / "x.json"
    write_sidecar(_toy_sidecar(Trim(start_s=12.5, end_s=300.0)), p)
    sc = read_sidecar(p)
    assert sc.trim is not None
    assert sc.trim.start_s == 12.5
    assert sc.trim.end_s == 300.0


def test_sidecar_round_trip_with_partial_trim(tmp_path: Path):
    p = tmp_path / "x.json"
    write_sidecar(_toy_sidecar(Trim(start_s=5.0, end_s=None)), p)
    sc = read_sidecar(p)
    assert sc.trim is not None
    assert sc.trim.start_s == 5.0
    assert sc.trim.end_s is None


def test_sidecar_round_trip_without_trim_omits_field(tmp_path: Path):
    p = tmp_path / "x.json"
    write_sidecar(_toy_sidecar(trim=None), p)
    payload = p.read_text()
    assert "trim" not in payload
    sc = read_sidecar(p)
    assert sc.trim is None


def test_sidecar_round_trip_empty_trim_omits_field(tmp_path: Path):
    """An all-None Trim is equivalent to no trim and shouldn't be written."""
    p = tmp_path / "x.json"
    write_sidecar(_toy_sidecar(Trim(start_s=None, end_s=None)), p)
    payload = p.read_text()
    assert "trim" not in payload


def test_trim_is_empty():
    assert Trim().is_empty()
    assert Trim(start_s=None, end_s=None).is_empty()
    assert not Trim(start_s=1.0).is_empty()
    assert not Trim(end_s=10.0).is_empty()
