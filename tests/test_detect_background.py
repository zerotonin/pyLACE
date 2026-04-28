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
