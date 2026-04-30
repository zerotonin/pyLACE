"""Max-projection background recovers the bright static scene."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pylace.detect.background import build_max_projection_background


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 25.0) -> None:
    cv2 = pytest.importorskip("cv2")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot open mp4v on this platform.")
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()


def test_max_projection_recovers_bright_background(tmp_path: Path) -> None:
    h, w = 64, 64
    bright = np.full((h, w, 3), 220, dtype=np.uint8)
    moving_blob_frames: list[np.ndarray] = []
    # The dark blob walks across the frame; the background is bright.
    for i in range(10):
        f = bright.copy()
        cx = 8 + i * 5
        f[20:40, cx:cx + 10] = 30  # dark patch
        moving_blob_frames.append(f)

    video = tmp_path / "moving.mp4"
    _write_video(video, moving_blob_frames)

    bg = build_max_projection_background(video, n_frames=10)
    assert bg.shape == (h, w)
    # The static bright background should dominate; mean pixel close to 220.
    assert bg.mean() > 200


def test_n_frames_must_be_positive(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    with pytest.raises(ValueError):
        build_max_projection_background(tmp_path / "x.mp4", n_frames=0)


def test_invalid_fraction_window_raises(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    with pytest.raises(ValueError):
        build_max_projection_background(
            tmp_path / "x.mp4", n_frames=5, start_frac=0.8, end_frac=0.2,
        )


def test_missing_video_raises(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    with pytest.raises(OSError):
        build_max_projection_background(tmp_path / "does-not-exist.mp4")


def test_save_load_background_round_trip(tmp_path: Path) -> None:
    from pylace.detect.background import (
        load_background_png,
        save_background_png,
    )

    bg = np.full((40, 40), 200, dtype=np.uint8)
    bg[10:20, 10:20] = 50
    out = tmp_path / "bg.png"
    save_background_png(bg, out)

    loaded = load_background_png(out)
    assert loaded.shape == bg.shape
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, bg)


def test_load_or_build_uses_sidecar_pair_when_present(tmp_path: Path) -> None:
    from pylace.detect.background import (
        default_background_paths,
        load_or_build_background,
        save_background_png,
    )

    video = tmp_path / "v.mp4"
    h, w = 32, 32
    bright = np.full((h, w, 3), 220, dtype=np.uint8)
    _write_video(video, [bright] * 3)

    fake_max = np.full((h, w), 99, dtype=np.uint8)
    fake_min = np.full((h, w), 33, dtype=np.uint8)
    max_path, min_path = default_background_paths(video)
    save_background_png(fake_max, max_path)
    save_background_png(fake_min, min_path)

    bg, source = load_or_build_background(video, polarity="dark_on_light")
    assert source == "sidecar"
    assert np.array_equal(bg, fake_max)

    bg2, source2 = load_or_build_background(video, polarity="light_on_dark")
    assert source2 == "sidecar"
    assert np.array_equal(bg2, fake_min)


def test_load_or_build_force_rebuild_overwrites_pair(tmp_path: Path) -> None:
    from pylace.detect.background import (
        default_background_paths,
        load_or_build_background,
        save_background_png,
    )

    video = tmp_path / "v.mp4"
    h, w = 32, 32
    bright = np.full((h, w, 3), 220, dtype=np.uint8)
    _write_video(video, [bright] * 5)

    max_path, min_path = default_background_paths(video)
    fake = np.full((h, w), 99, dtype=np.uint8)
    save_background_png(fake, max_path)
    save_background_png(fake, min_path)

    bg, source = load_or_build_background(video, force_rebuild=True)
    assert source == "computed"
    assert max_path.exists() and min_path.exists()


def test_min_projection_recovers_dark_surround_with_light_animal(tmp_path: Path) -> None:
    from pylace.detect.background import build_min_projection_background

    h, w = 64, 64
    dark = np.full((h, w, 3), 30, dtype=np.uint8)
    frames: list[np.ndarray] = []
    for i in range(10):
        f = dark.copy()
        cx = 8 + i * 5
        f[20:40, cx:cx + 10] = 220  # bright "animal"
        frames.append(f)
    video = tmp_path / "light_on_dark.mp4"
    _write_video(video, frames)

    bg = build_min_projection_background(video, n_frames=10)
    assert bg.mean() < 60  # the dark background dominates


def test_compute_projection_pair_returns_both(tmp_path: Path) -> None:
    from pylace.detect.background import compute_projection_pair

    h, w = 32, 32
    frames: list[np.ndarray] = []
    for value in (50, 100, 200):
        frames.append(np.full((h, w, 3), value, dtype=np.uint8))
    video = tmp_path / "pair.mp4"
    _write_video(video, frames)

    bg_max, bg_min = compute_projection_pair(video, n_frames=3)
    assert bg_max.max() >= bg_min.max()
    assert bg_max.mean() > bg_min.mean()


def test_detection_and_trail_routes_by_polarity():
    from pylace.detect.background import detection_and_trail

    bg_max = np.full((4, 4), 200, dtype=np.uint8)
    bg_min = np.full((4, 4), 50, dtype=np.uint8)
    det, trail = detection_and_trail(bg_max, bg_min, "dark_on_light")
    assert np.array_equal(det, bg_max)
    assert np.array_equal(trail, bg_min)
    det2, trail2 = detection_and_trail(bg_max, bg_min, "light_on_dark")
    assert np.array_equal(det2, bg_min)
    assert np.array_equal(trail2, bg_max)
