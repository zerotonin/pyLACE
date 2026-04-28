"""Background model via per-pixel max-projection across sampled frames.

Dark animals on a bright arena background: the per-pixel maximum across a
sample of frames recovers the bright static background, even when animals
move only a little. This is the same trick thermokourt and similar tools
use; the algorithm is well-known and not specific to LACE.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build_max_projection_background(
    video_path: Path,
    n_frames: int = 50,
    start_frac: float = 0.1,
    end_frac: float = 0.9,
) -> np.ndarray:
    """Per-pixel max-projection grayscale background.

    Args:
        video_path: Path to the source video.
        n_frames: Number of frames to sample, evenly spaced.
        start_frac: Skip the first ``start_frac`` of the video.
        end_frac: Stop sampling at ``end_frac`` of the video.

    Returns:
        Grayscale ``uint8`` array shaped ``(height, width)``.

    Raises:
        IOError: If the video cannot be opened or no frames can be read.
    """
    if not 0.0 <= start_frac < end_frac <= 1.0:
        raise ValueError("Need 0 <= start_frac < end_frac <= 1.")
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise OSError(f"Video has zero frames: {video_path}")
        indices = _sample_frame_indices(total, n_frames, start_frac, end_frac)
        return _max_projection_from_indices(cap, indices)
    finally:
        cap.release()


def _sample_frame_indices(
    total: int, n: int, start_frac: float, end_frac: float,
) -> list[int]:
    start = int(total * start_frac)
    end = max(start + 1, int(total * end_frac))
    n = min(n, end - start)
    if n <= 1:
        return [start]
    return [int(start + i * (end - 1 - start) / (n - 1)) for i in range(n)]


def _max_projection_from_indices(
    cap: cv2.VideoCapture, indices: list[int],
) -> np.ndarray:
    bg: np.ndarray | None = None
    used = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bg = gray if bg is None else np.maximum(bg, gray)
        used += 1
    if bg is None or used == 0:
        raise OSError("No frames could be read for background estimation.")
    return bg
