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


def test_load_or_build_uses_sidecar_when_present(tmp_path: Path) -> None:
    from pylace.detect.background import (
        default_background_path,
        load_or_build_background,
        save_background_png,
    )

    video = tmp_path / "v.mp4"
    h, w = 32, 32
    bright = np.full((h, w, 3), 220, dtype=np.uint8)
    _write_video(video, [bright] * 3)

    # Pre-stage a deliberately distinctive saved background.
    fake_bg = np.full((h, w), 99, dtype=np.uint8)
    save_background_png(fake_bg, default_background_path(video))

    bg, source = load_or_build_background(video)
    assert source == "sidecar"
    assert np.array_equal(bg, fake_bg)


def test_load_or_build_force_rebuild_overwrites_sidecar(tmp_path: Path) -> None:
    from pylace.detect.background import (
        default_background_path,
        load_or_build_background,
        save_background_png,
    )

    video = tmp_path / "v.mp4"
    h, w = 32, 32
    bright = np.full((h, w, 3), 220, dtype=np.uint8)
    _write_video(video, [bright] * 5)

    fake_bg = np.full((h, w), 99, dtype=np.uint8)
    save_background_png(fake_bg, default_background_path(video))

    bg, source = load_or_build_background(video, force_rebuild=True)
    assert source == "computed"
    assert not np.array_equal(bg, fake_bg)
    # Sidecar overwritten with the freshly computed bg.
    on_disk = np.array(
        __import__("cv2").imread(
            str(default_background_path(video)),
            __import__("cv2").IMREAD_GRAYSCALE,
        )
    )
    assert np.array_equal(on_disk, bg)
