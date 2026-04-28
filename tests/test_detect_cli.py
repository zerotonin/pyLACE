"""End-to-end CLI test for ``pylace-detect``."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pylace.annotator import (
    Calibration,
    Circle,
    Sidecar,
    VideoMeta,
    WorldFrame,
    default_sidecar_path,
    video_sha256,
    write_sidecar,
)
from pylace.detect import cli


def _write_video_with_walking_blob(path: Path, n_frames: int = 12) -> tuple[int, int]:
    cv2 = pytest.importorskip("cv2")
    h, w = 80, 80
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot open mp4v on this platform.")
    try:
        for i in range(n_frames):
            frame = np.full((h, w, 3), 220, dtype=np.uint8)
            cx = 20 + i * 3
            cv2.ellipse(frame, (cx, 40), (8, 4), 0, 0, 360, (30, 30, 30), -1)
            writer.write(frame)
    finally:
        writer.release()
    return w, h


def _write_circle_sidecar(video: Path, w: int, h: int) -> Path:
    sidecar = Sidecar(
        video=VideoMeta(
            path=str(video),
            sha256=video_sha256(video),
            frame_size=(w, h),
            fps=25.0,
        ),
        arena=Circle(cx=40.0, cy=40.0, r=35.0),
        world_frame=WorldFrame(origin_pixel=(40.0, 40.0), y_axis="up", x_axis="right"),
        calibration=Calibration(
            reference_kind="diameter", physical_mm=10.0, pixel_distance=70.0,
        ),
    )
    out = default_sidecar_path(video)
    write_sidecar(sidecar, out)
    return out


@pytest.fixture
def video_and_sidecar(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "blob.mp4"
    w, h = _write_video_with_walking_blob(video)
    sidecar = _write_circle_sidecar(video, w, h)
    return video, sidecar


def test_pylace_detect_writes_one_detection_per_frame(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    video, _ = video_and_sidecar
    out = tmp_path / "detections.csv"
    rc = cli.main([str(video), "--out", str(out)])
    assert rc == 0

    rows = list(csv.DictReader(out.read_text().splitlines()))
    # All rows are detection_idx == 0 (one blob per frame).
    assert {int(r["detection_idx"]) for r in rows} == {0}
    # First row's centroid is near (20, 40); last row's is near (53, 40).
    first = rows[0]
    last = rows[-1]
    assert abs(float(first["cx_px"]) - 20.0) < 4
    assert abs(float(first["cy_px"]) - 40.0) < 3
    assert float(last["cx_px"]) > float(first["cx_px"])


def test_pylace_detect_world_coords_origin_at_arena_centre(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    video, _ = video_and_sidecar
    out = tmp_path / "detections.csv"
    cli.main([str(video), "--out", str(out)])

    rows = list(csv.DictReader(out.read_text().splitlines()))
    first = rows[0]
    # Pixel (20, 40) → mm relative to origin (40, 40) at scale 10/70 mm/px,
    # with y-up flipping the y sign. (cx - 40) = -20; (cy - 40) = 0.
    expected_x_mm = (20.0 - 40.0) * (10.0 / 70.0)
    assert float(first["x_mm"]) == pytest.approx(expected_x_mm, abs=0.5)
    assert abs(float(first["y_mm"])) < 0.5


def test_pylace_detect_max_frames_caps_output(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    video, _ = video_and_sidecar
    out = tmp_path / "limited.csv"
    rc = cli.main([str(video), "--out", str(out), "--max-frames", "3"])
    assert rc == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    frame_indices = {int(r["frame_idx"]) for r in rows}
    assert len(frame_indices) <= 3


def test_pylace_detect_missing_video_returns_nonzero(tmp_path: Path) -> None:
    rc = cli.main([str(tmp_path / "no-such.mp4")])
    assert rc != 0


def test_pylace_detect_missing_sidecar_returns_nonzero(
    tmp_path: Path,
) -> None:
    video = tmp_path / "blob.mp4"
    _write_video_with_walking_blob(video)
    rc = cli.main([str(video)])
    assert rc != 0


def test_pylace_detect_time_window_restricts_frames(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """``--start`` / ``--end`` should bound the absolute frame indices written."""
    video, _ = video_and_sidecar  # 12 frames at 25 fps = 0.48 s total
    out = tmp_path / "windowed.csv"
    # 0.16 s -> frame 4; 0.32 s -> frame 8.
    rc = cli.main([str(video), "--out", str(out), "--start", "0.16", "--end", "0.32"])
    assert rc == 0

    rows = list(csv.DictReader(out.read_text().splitlines()))
    frame_indices = sorted({int(r["frame_idx"]) for r in rows})
    assert frame_indices, "expected at least one detection in the window"
    assert all(4 <= idx < 8 for idx in frame_indices), frame_indices


def test_pylace_detect_end_before_start_returns_nonzero(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    video, _ = video_and_sidecar
    rc = cli.main([
        str(video), "--out", str(tmp_path / "x.csv"),
        "--start", "0.3", "--end", "0.1",
    ])
    assert rc != 0


def test_pylace_detect_time_spec_accepts_mm_ss_and_hh_mm_ss(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    video, _ = video_and_sidecar
    # 0:00:00.20 -> 0.20 s -> frame 5; 0:00.32 -> 0.32 s -> frame 8.
    rc = cli.main([
        str(video), "--out", str(tmp_path / "fmt.csv"),
        "--start", "0:00:00.20", "--end", "0:00.32",
    ])
    assert rc == 0


def test_pipeline_start_end_frame_yields_absolute_indices(
    video_and_sidecar: tuple[Path, Path],
) -> None:
    """Direct pipeline call with frame-based bounds yields the right slice."""
    from pylace.annotator.sidecar import default_sidecar_path, read_sidecar
    from pylace.detect.pipeline import run_detection

    video, _ = video_and_sidecar
    sidecar = read_sidecar(default_sidecar_path(video))

    results = list(run_detection(video, sidecar, start_frame=4, end_frame=8))
    assert [r.frame_idx for r in results] == [4, 5, 6, 7]
