"""Sample-preview-frames behaviour on synthetic videos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pylace.tune.frame_loader import sample_preview_frames


def _write_video(path: Path, n: int) -> tuple[int, int]:
    cv2 = pytest.importorskip("cv2")
    h, w = 32, 32
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot open mp4v on this platform.")
    try:
        for i in range(n):
            frame = np.full((h, w, 3), 200, dtype=np.uint8)
            # Stamp a per-frame pattern so we can verify ordering.
            frame[0, 0] = (i, i, i)
            writer.write(frame)
    finally:
        writer.release()
    return w, h


def test_sample_returns_n_grayscale_frames(tmp_path: Path):
    video = tmp_path / "v.mp4"
    _write_video(video, n=20)
    frames, idx = sample_preview_frames(video, n=5)
    assert len(frames) == 5
    assert len(idx) == 5
    assert all(f.ndim == 2 and f.dtype == np.uint8 for f in frames)


def test_sample_indices_are_in_window_and_evenly_spaced(tmp_path: Path):
    video = tmp_path / "v.mp4"
    _write_video(video, n=30)
    _, idx = sample_preview_frames(video, n=4, start_frame=10, end_frame=22)
    assert idx[0] >= 10
    assert idx[-1] < 22
    # Within rounding, gaps are uniform.
    gaps = [idx[i + 1] - idx[i] for i in range(len(idx) - 1)]
    assert max(gaps) - min(gaps) <= 1


def test_sample_n_one_returns_first_frame_in_window(tmp_path: Path):
    video = tmp_path / "v.mp4"
    _write_video(video, n=10)
    _, idx = sample_preview_frames(video, n=1, start_frame=4)
    assert idx == [4]


def test_sample_rejects_bad_inputs(tmp_path: Path):
    pytest.importorskip("cv2")
    with pytest.raises(ValueError):
        sample_preview_frames(tmp_path / "x.mp4", n=0)
    with pytest.raises(ValueError):
        sample_preview_frames(tmp_path / "x.mp4", n=3, start_frame=-1)


def test_sample_missing_video_raises(tmp_path: Path):
    pytest.importorskip("cv2")
    with pytest.raises(OSError):
        sample_preview_frames(tmp_path / "no.mp4", n=3)
