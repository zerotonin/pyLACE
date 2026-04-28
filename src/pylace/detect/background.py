"""Background model via per-pixel max-projection across sampled frames.

Dark animals on a bright arena background: the per-pixel maximum across a
sample of frames recovers the bright static background, even when animals
move only a little. This is the same trick thermokourt and similar tools
use; the algorithm is well-known and not specific to LACE.

Backgrounds can be persisted next to the video as a PNG sidecar so the
expensive max-projection runs once, downstream tools all see the same
arena lighting, and an operator can edit out a stationary fly that was
baked into the background.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

BACKGROUND_SUFFIX = ".pylace_background.png"


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


def default_background_path(video: Path) -> Path:
    """Conventional sidecar path for a saved background next to the video."""
    return video.with_name(video.name + BACKGROUND_SUFFIX)


def save_background_png(bg: np.ndarray, out_path: Path) -> None:
    """Write a grayscale background to a PNG sidecar.

    Raises:
        OSError: If the file cannot be written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), bg):
        raise OSError(f"Cannot write background PNG to {out_path}")


def load_background_png(in_path: Path) -> np.ndarray:
    """Read a previously-saved grayscale background PNG.

    Raises:
        OSError: If the file cannot be read or is not a valid image.
    """
    bg = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
    if bg is None:
        raise OSError(f"Cannot read background PNG from {in_path}")
    return bg


def load_or_build_background(
    video_path: Path,
    *,
    n_frames: int = 50,
    start_frac: float = 0.1,
    end_frac: float = 0.9,
    force_rebuild: bool = False,
    save_to: Path | None = None,
) -> tuple[np.ndarray, str]:
    """Return ``(bg, source)`` where ``source`` is ``'sidecar'`` or ``'computed'``.

    Loads the conventional ``<video>.pylace_background.png`` sidecar when
    one is present and ``force_rebuild`` is False; otherwise computes a
    fresh max-projection and (unless ``save_to`` is None) writes it to
    ``save_to``. Default ``save_to`` is the conventional sidecar path.
    """
    sidecar = save_to if save_to is not None else default_background_path(video_path)
    if sidecar.exists() and not force_rebuild:
        return load_background_png(sidecar), "sidecar"
    bg = build_max_projection_background(
        video_path,
        n_frames=n_frames,
        start_frac=start_frac,
        end_frac=end_frac,
    )
    if save_to is not None or not sidecar.exists() or force_rebuild:
        save_background_png(bg, sidecar)
    return bg, "computed"
