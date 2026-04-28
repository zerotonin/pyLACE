"""Sample evenly-spaced grayscale preview frames from a video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_preview_frames(
    video_path: Path,
    n: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """Read ``n`` grayscale frames evenly across ``[start_frame, end_frame)``.

    Args:
        video_path: Source video.
        n: Number of frames to load. ``n >= 1``.
        start_frame: First eligible frame index (inclusive).
        end_frame: One past the last eligible frame index. ``None`` means
            run to the last frame in the file.

    Returns:
        ``(frames, indices)`` — a list of grayscale numpy arrays and the
        absolute frame indices they were read from. If a seek fails for
        any index it is silently skipped, so the lists may be shorter than
        ``n``.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise OSError(f"Video has zero frames: {video_path}")
        end = end_frame if end_frame is not None else total
        if end <= start_frame:
            raise ValueError("end_frame must be > start_frame.")
        end = min(end, total)
        indices = _evenly_spaced_indices(start_frame, end, n)
        return _read_frames(cap, indices)
    finally:
        cap.release()


def _evenly_spaced_indices(start: int, end: int, n: int) -> list[int]:
    if n == 1 or end - start <= 1:
        return [start]
    step = (end - 1 - start) / (n - 1)
    return [int(round(start + i * step)) for i in range(n)]


def _read_frames(
    cap: cv2.VideoCapture, indices: list[int],
) -> tuple[list[np.ndarray], list[int]]:
    frames: list[np.ndarray] = []
    used: list[int] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        used.append(idx)
    return frames, used
