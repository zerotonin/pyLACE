"""End-to-end CLI test for ``pylace-annotate --headless``.

This is the whitepaper-mandated CLI test: it builds a tiny synthetic mp4,
invokes ``cli.main([...])`` for each shape, and parses the resulting sidecar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pylace.annotator import cli, read_sidecar


def _write_synthetic_mp4(path: Path, n_frames: int = 3, size: tuple[int, int] = (64, 64)) -> None:
    """Write a tiny mp4 of solid grey frames so cv2.VideoCapture can probe it."""
    cv2 = pytest.importorskip("cv2")
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (width, height))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot open mp4v on this platform.")
    try:
        frame = np.full((height, width, 3), 200, dtype=np.uint8)
        for _ in range(n_frames):
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    video = tmp_path / "synthetic.mp4"
    _write_synthetic_mp4(video)
    return video


def test_headless_circle_writes_sidecar(synthetic_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "circle.json"
    rc = cli.main([
        str(synthetic_video),
        "--headless",
        "--shape", "circle",
        "--cx", "32", "--cy", "32", "--r", "20",
        "--origin-x", "32", "--origin-y", "32",
        "--diameter-mm", "10",
        "--out", str(out),
    ])
    assert rc == 0
    sidecar = read_sidecar(out)
    assert sidecar.calibration.reference_kind == "diameter"
    assert sidecar.calibration.mm_per_pixel == pytest.approx(10.0 / 40.0)
    assert sidecar.world_frame.origin_pixel == (32.0, 32.0)
    assert sidecar.world_frame.y_axis == "up"


def test_headless_rectangle_writes_sidecar(synthetic_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "rect.json"
    rc = cli.main([
        str(synthetic_video),
        "--headless",
        "--shape", "rectangle",
        "--vertices", "10,10;50,10;50,40;10,40",
        "--origin-x", "10", "--origin-y", "40",
        "--y-down",
        "--edge-mm", "20",
        "--edge-vertices", "0,1",
        "--out", str(out),
    ])
    assert rc == 0
    sidecar = read_sidecar(out)
    assert sidecar.calibration.reference_kind == "edge"
    assert sidecar.calibration.reference_vertices == (0, 1)
    assert sidecar.calibration.pixel_distance == pytest.approx(40.0)
    assert sidecar.calibration.mm_per_pixel == pytest.approx(0.5)
    assert sidecar.world_frame.y_axis == "down"


def test_headless_polygon_writes_sidecar(synthetic_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "poly.json"
    rc = cli.main([
        str(synthetic_video),
        "--headless",
        "--shape", "polygon",
        "--vertices", "10,10;30,10;40,30;20,40;5,25",
        "--origin-x", "10", "--origin-y", "10",
        "--edge-mm", "15",
        "--edge-vertices", "0,1",
        "--out", str(out),
    ])
    assert rc == 0
    sidecar = read_sidecar(out)
    assert sidecar.arena.vertices == [
        (10.0, 10.0), (30.0, 10.0), (40.0, 30.0), (20.0, 40.0), (5.0, 25.0),
    ]
    assert sidecar.calibration.pixel_distance == pytest.approx(20.0)
    assert sidecar.calibration.mm_per_pixel == pytest.approx(0.75)


def test_headless_default_sidecar_path(synthetic_video: Path) -> None:
    rc = cli.main([
        str(synthetic_video),
        "--headless",
        "--shape", "circle",
        "--cx", "32", "--cy", "32", "--r", "20",
        "--origin-x", "32", "--origin-y", "32",
        "--diameter-mm", "10",
    ])
    assert rc == 0
    expected = synthetic_video.with_name(synthetic_video.name + ".pylace_arena.json")
    assert expected.exists()


def test_missing_video_returns_nonzero(tmp_path: Path) -> None:
    rc = cli.main([
        str(tmp_path / "does-not-exist.mp4"),
        "--headless",
        "--shape", "circle",
        "--cx", "0", "--cy", "0", "--r", "1",
        "--origin-x", "0", "--origin-y", "0",
        "--diameter-mm", "1",
    ])
    assert rc != 0
