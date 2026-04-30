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
    # One blob per frame and tracking on by default → all rows share track_id.
    assert len({int(r["track_id"]) for r in rows}) == 1
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


def test_pylace_detect_loads_sibling_tuning_params(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """An auto-detected ``.pylace_detect_params.json`` overrides flag defaults."""
    from pylace.tune.params import (
        BackgroundParams,
        DetectionParams,
        TuningParams,
        default_params_path,
        write_params,
    )

    video, _ = video_and_sidecar
    # Aggressive min_area to suppress all detections.
    write_params(
        TuningParams(
            detection=DetectionParams(
                threshold=25, min_area=10_000, max_area=100_000, morph_kernel=3,
            ),
            background=BackgroundParams(),
        ),
        video_path=video, video_sha256_hex="0" * 64,
        out_path=default_params_path(video),
    )

    out = tmp_path / "with_params.csv"
    rc = cli.main([str(video), "--out", str(out)])
    assert rc == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert rows == []  # tuned min_area suppressed everything


def test_pylace_detect_explicit_flag_overrides_sibling_tuning_params(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """User-provided CLI flags win over the params sidecar."""
    from pylace.tune.params import (
        BackgroundParams,
        DetectionParams,
        TuningParams,
        default_params_path,
        write_params,
    )

    video, _ = video_and_sidecar
    write_params(
        TuningParams(
            detection=DetectionParams(min_area=10_000),
            background=BackgroundParams(),
        ),
        video_path=video, video_sha256_hex="0" * 64,
        out_path=default_params_path(video),
    )

    out = tmp_path / "override.csv"
    rc = cli.main([str(video), "--out", str(out), "--min-area", "20"])
    assert rc == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert rows  # the explicit --min-area lets detections through


def test_pylace_detect_writes_background_sidecar_and_reuses_it(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """First run writes the max + min PNG pair; second run reads them back."""
    from pylace.detect.background import default_background_paths

    video, _ = video_and_sidecar
    max_path, min_path = default_background_paths(video)
    for p in (max_path, min_path):
        if p.exists():
            p.unlink()

    rc = cli.main([str(video), "--out", str(tmp_path / "first.csv")])
    assert rc == 0
    assert max_path.exists() and min_path.exists()
    first_mtime = max_path.stat().st_mtime

    rc = cli.main([str(video), "--out", str(tmp_path / "second.csv")])
    assert rc == 0
    assert max_path.stat().st_mtime == first_mtime


def test_pylace_detect_rebuild_background_flag_overwrites(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    from pylace.detect.background import default_background_paths

    video, _ = video_and_sidecar
    max_path, _ = default_background_paths(video)
    cli.main([str(video), "--out", str(tmp_path / "first.csv")])
    assert max_path.exists()

    import os
    import time

    old_time = time.time() - 60
    os.utime(max_path, (old_time, old_time))
    rc = cli.main([
        str(video), "--out", str(tmp_path / "rebuild.csv"),
        "--rebuild-background",
    ])
    assert rc == 0
    assert max_path.stat().st_mtime > old_time + 1


def test_pylace_detect_consumes_sibling_roi_sidecar(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """A subtractive ROI over the blob's path should suppress detections."""
    from pylace.annotator.geometry import Circle, Rectangle
    from pylace.roi.geometry import ROI, ROISet
    from pylace.roi.sidecar import ROISidecar, default_rois_path, write_rois

    video, _ = video_and_sidecar
    # ROI = full arena MINUS the strip the blob walks across.
    rois = ROISet(
        rois=[
            ROI(shape=Rectangle.from_two_points((0.0, 0.0), (80.0, 80.0))),
            ROI(
                shape=Rectangle.from_two_points((0.0, 30.0), (80.0, 50.0)),
                operation="subtract",
            ),
        ],
    )
    write_rois(
        ROISidecar(video_path=str(video), video_sha256="0" * 64, roi_set=rois),
        default_rois_path(video),
    )

    out = tmp_path / "with_rois.csv"
    rc = cli.main([str(video), "--out", str(out)])
    assert rc == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert rows == []  # the blob stripe is masked out


def test_pylace_detect_split_mode_tags_rows_with_roi_label(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """Split mode runs detection per add-ROI and tags rows with the label."""
    from pylace.annotator.geometry import Rectangle
    from pylace.roi.geometry import ROI, ROISet
    from pylace.roi.sidecar import ROISidecar, default_rois_path, write_rois

    video, _ = video_and_sidecar
    rois = ROISet(
        rois=[
            ROI(
                shape=Rectangle.from_two_points((0.0, 0.0), (40.0, 80.0)),
                label="left",
            ),
            ROI(
                shape=Rectangle.from_two_points((40.0, 0.0), (80.0, 80.0)),
                label="right",
            ),
        ],
        mode="split",
    )
    write_rois(
        ROISidecar(video_path=str(video), video_sha256="0" * 64, roi_set=rois),
        default_rois_path(video),
    )

    out = tmp_path / "split.csv"
    rc = cli.main([str(video), "--out", str(out)])
    assert rc == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    labels = {r["roi_label"] for r in rows}
    assert "left" in labels
    assert "right" in labels


def test_pylace_detect_no_rois_flag_bypasses_sidecar(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """``--no-rois`` should ignore an existing ROI sidecar."""
    from pylace.annotator.geometry import Circle
    from pylace.roi.geometry import ROI, ROISet
    from pylace.roi.sidecar import ROISidecar, default_rois_path, write_rois

    video, _ = video_and_sidecar
    rois = ROISet(rois=[ROI(shape=Circle(0.0, 0.0, 1.0))])  # tiny corner ROI
    write_rois(
        ROISidecar(video_path=str(video), video_sha256="0" * 64, roi_set=rois),
        default_rois_path(video),
    )

    out = tmp_path / "ignore_rois.csv"
    rc = cli.main([str(video), "--out", str(out), "--no-rois"])
    assert rc == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert rows  # detections happen because ROI was bypassed


def test_pylace_detect_tracking_assigns_stable_track_id_across_frames(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """The walking blob keeps the same track_id across every frame."""
    video, _ = video_and_sidecar
    out = tmp_path / "tracked.csv"
    rc = cli.main([str(video), "--out", str(out)])
    assert rc == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert len({int(r["track_id"]) for r in rows}) == 1
    # And the centroid actually walked: cx_px monotonically increases.
    cx_values = [float(r["cx_px"]) for r in rows]
    assert cx_values[-1] > cx_values[0]


def test_pylace_detect_no_track_falls_back_to_per_frame_index(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """``--no-track`` produces track_id == per-frame detection index."""
    video, _ = video_and_sidecar
    out = tmp_path / "untracked.csv"
    rc = cli.main([str(video), "--out", str(out), "--no-track"])
    assert rc == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    # One blob per frame; per-frame index is 0 each time.
    assert {int(r["track_id"]) for r in rows} == {0}


def _make_video_with_two_blobs_one_chained(
    path: Path, n_frames: int = 6,
) -> tuple[int, int]:
    """First N-1 frames: two distinct blobs. Last frame: a single 2-fly chain.

    Returns ``(width, height)``.
    """
    cv2 = pytest.importorskip("cv2")
    h, w = 80, 100
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot open mp4v on this platform.")
    try:
        for i in range(n_frames - 1):
            frame = np.full((h, w, 3), 220, dtype=np.uint8)
            cv2.ellipse(frame, (30, 40), (8, 4), 0, 0, 360, (30, 30, 30), -1)
            cv2.ellipse(frame, (70, 40), (8, 4), 0, 0, 360, (30, 30, 30), -1)
            writer.write(frame)
        # Final frame: the two flies pile up into one elongated blob.
        chain_frame = np.full((h, w, 3), 220, dtype=np.uint8)
        cv2.ellipse(chain_frame, (50, 40), (16, 4), 0, 0, 360, (30, 30, 30), -1)
        writer.write(chain_frame)
    finally:
        writer.release()
    return w, h


def test_pylace_detect_chain_split_recovers_two_detections_in_chain(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """A chained frame produces 2 detections after splitting (1 without)."""
    from pylace.annotator.geometry import Circle
    from pylace.annotator.sidecar import (
        Sidecar, VideoMeta, WorldFrame,
        Calibration, default_sidecar_path, video_sha256, write_sidecar,
    )

    video = tmp_path / "chain.mp4"
    w, h = _make_video_with_two_blobs_one_chained(video)
    sidecar = Sidecar(
        video=VideoMeta(path=str(video), sha256=video_sha256(video),
                        frame_size=(w, h), fps=25.0),
        arena=Circle(cx=50.0, cy=40.0, r=45.0),
        world_frame=WorldFrame(origin_pixel=(50.0, 40.0)),
        calibration=Calibration(
            reference_kind="diameter", physical_mm=10.0, pixel_distance=90.0,
        ),
    )
    write_sidecar(sidecar, default_sidecar_path(video))

    # Without chain split: the chain frame yields 1 contour → 1 detection.
    out_no_split = tmp_path / "no_split.csv"
    rc = cli.main([
        str(video), "--out", str(out_no_split),
        "--no-chain-split",
        "--no-track",  # keep track_id deterministic per-frame
    ])
    assert rc == 0
    rows = list(csv.DictReader(out_no_split.read_text().splitlines()))
    last_frame = max(int(r["frame_idx"]) for r in rows)
    last_rows_no_split = [r for r in rows if int(r["frame_idx"]) == last_frame]
    assert len(last_rows_no_split) == 1

    # With chain split + an explicit expected area, the chain splits.
    out_split = tmp_path / "split.csv"
    rc = cli.main([
        str(video), "--out", str(out_split),
        "--expected-animal-area", "100",
        "--no-track",
    ])
    assert rc == 0
    rows_split = list(csv.DictReader(out_split.read_text().splitlines()))
    last_rows_split = [
        r for r in rows_split if int(r["frame_idx"]) == last_frame
    ]
    assert len(last_rows_split) == 2


def test_pylace_detect_n_animals_caps_unique_track_ids(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """``--n-animals`` clamps the number of distinct track_ids in the CSV."""
    video, _ = video_and_sidecar
    out = tmp_path / "fixed_n.csv"
    rc = cli.main([str(video), "--out", str(out), "--n-animals", "1"])
    assert rc == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    track_ids = {int(r["track_id"]) for r in rows}
    assert track_ids == {0}


def test_pylace_detect_max_track_distance_zero_births_id_per_frame(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """An aggressive max_track_distance prevents any cross-frame matches."""
    video, _ = video_and_sidecar
    out = tmp_path / "no_match.csv"
    rc = cli.main([
        str(video), "--out", str(out),
        "--max-track-distance", "0",
        "--max-missed-frames", "0",
    ])
    assert rc == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    track_ids = [int(r["track_id"]) for r in rows]
    assert len(set(track_ids)) == len(rows)


def test_pylace_detect_dilate_and_erode_flags_propagate(
    video_and_sidecar: tuple[Path, Path], tmp_path: Path,
) -> None:
    """``--dilate-iters`` / ``--erode-iters`` reach the pipeline; --dilate-iters
    grows blobs into a clearly larger area than the default run."""
    video, _ = video_and_sidecar

    out_base = tmp_path / "base.csv"
    cli.main([str(video), "--out", str(out_base)])
    base_rows = list(csv.DictReader(out_base.read_text().splitlines()))
    base_areas = [float(r["area_px"]) for r in base_rows]

    out_dilated = tmp_path / "dilated.csv"
    cli.main([str(video), "--out", str(out_dilated), "--dilate-iters", "3"])
    dilated_rows = list(csv.DictReader(out_dilated.read_text().splitlines()))
    dilated_areas = [float(r["area_px"]) for r in dilated_rows]

    assert base_areas and dilated_areas
    assert max(dilated_areas) > max(base_areas)
