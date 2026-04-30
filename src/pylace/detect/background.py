"""Background model via per-pixel max- or min-projection across sampled frames.

Two projections, two purposes:

- **max**-projection recovers the *bright* component of the scene. With
  dark animals on a light arena, this is the clean background used for
  bg-subtraction detection. With light animals on a dark arena, the
  same projection becomes a "where did the animals go" trail image
  (the brightest pixels are wherever the animal sat).
- **min**-projection is the symmetric opposite. With light animals on
  a dark arena it is the detection background; with dark animals on a
  light arena it is the trail image.

The two are computed in a single pass over the sampled frames and
saved as separate PNG sidecars next to the video, so flipping the
polarity does not require any recomputation. Backgrounds are
persisted next to the video so the expensive projection runs once,
downstream tools all see the same arena lighting, and an operator
can edit out a stationary fly that was baked into the background.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

# Legacy single-bg name (pre-polarity).  Kept so old sidecars are still
# readable; new code reads / writes the polarity-tagged pair below.
BACKGROUND_SUFFIX = ".pylace_background.png"
BACKGROUND_MAX_SUFFIX = ".pylace_background_max.png"
BACKGROUND_MIN_SUFFIX = ".pylace_background_min.png"

Polarity = Literal["dark_on_light", "light_on_dark"]


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
    bg, _ = _projection_pair_from_indices(cap, indices)
    return bg


def _projection_pair_from_indices(
    cap: cv2.VideoCapture, indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass max + min projection over sampled frames."""
    bg_max: np.ndarray | None = None
    bg_min: np.ndarray | None = None
    used = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bg_max = gray if bg_max is None else np.maximum(bg_max, gray)
        bg_min = gray if bg_min is None else np.minimum(bg_min, gray)
        used += 1
    if bg_max is None or bg_min is None or used == 0:
        raise OSError("No frames could be read for background estimation.")
    return bg_max, bg_min


def build_min_projection_background(
    video_path: Path,
    n_frames: int = 50,
    start_frac: float = 0.1,
    end_frac: float = 0.9,
) -> np.ndarray:
    """Per-pixel min-projection grayscale background.

    Use when animals are *brighter* than the arena background; the
    per-pixel minimum recovers the dark static background.
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
        _, bg = _projection_pair_from_indices(cap, indices)
        return bg
    finally:
        cap.release()


def compute_projection_pair(
    video_path: Path,
    n_frames: int = 50,
    start_frac: float = 0.1,
    end_frac: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass return of ``(max_bg, min_bg)`` for the sampled frames.

    Cheaper than calling the two builders separately because each frame
    is decoded only once.
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
        return _projection_pair_from_indices(cap, indices)
    finally:
        cap.release()


def default_background_path(video: Path) -> Path:
    """Legacy single-bg sidecar path. Prefer :func:`default_background_paths`."""
    return video.with_name(video.name + BACKGROUND_SUFFIX)


def default_background_paths(video: Path) -> tuple[Path, Path]:
    """Return ``(max_path, min_path)`` for the dual-projection sidecars."""
    return (
        video.with_name(video.name + BACKGROUND_MAX_SUFFIX),
        video.with_name(video.name + BACKGROUND_MIN_SUFFIX),
    )


def detection_and_trail(
    bg_max: np.ndarray, bg_min: np.ndarray, polarity: Polarity,
) -> tuple[np.ndarray, np.ndarray]:
    """Route ``(max, min)`` to ``(detection, trail)`` based on polarity.

    - ``dark_on_light``: detection = max (bright background, animals are
      darker), trail = min (where the animal sat — the dark spots).
    - ``light_on_dark``: the symmetric opposite.
    """
    if polarity == "dark_on_light":
        return bg_max, bg_min
    if polarity == "light_on_dark":
        return bg_min, bg_max
    raise ValueError(f"Unknown polarity: {polarity!r}.")


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
    polarity: Polarity = "dark_on_light",
    save_to: Path | None = None,
) -> tuple[np.ndarray, str]:
    """Return ``(detection_bg, source)`` for the given ``polarity``.

    Polarity-aware wrapper around :func:`load_or_build_background_pair`:
    routes ``(max, min)`` to detection-bg via :func:`detection_and_trail`
    and discards the trail. ``save_to`` overrides the *max*-bg path
    only — used by tests that pre-stage a single sidecar.
    """
    detection_bg, _trail, source = load_or_build_background_pair(
        video_path,
        n_frames=n_frames,
        start_frac=start_frac,
        end_frac=end_frac,
        force_rebuild=force_rebuild,
        polarity=polarity,
        save_to_max=save_to,
    )
    return detection_bg, source


def load_or_build_background_pair(
    video_path: Path,
    *,
    n_frames: int = 50,
    start_frac: float = 0.1,
    end_frac: float = 0.9,
    force_rebuild: bool = False,
    polarity: Polarity = "dark_on_light",
    save_to_max: Path | None = None,
    save_to_min: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(detection_bg, trail_bg, source)`` honouring ``polarity``.

    Loads ``<video>.pylace_background_max.png`` and
    ``<video>.pylace_background_min.png`` when both sidecars exist and
    ``force_rebuild`` is False; otherwise recomputes the pair in a
    single pass and writes both. ``source`` is ``'sidecar'`` or
    ``'computed'``.
    """
    max_path, min_path = default_background_paths(video_path)
    if save_to_max is not None:
        max_path = save_to_max
    if save_to_min is not None:
        min_path = save_to_min

    if max_path.exists() and min_path.exists() and not force_rebuild:
        bg_max = load_background_png(max_path)
        bg_min = load_background_png(min_path)
        detection_bg, trail_bg = detection_and_trail(bg_max, bg_min, polarity)
        return detection_bg, trail_bg, "sidecar"

    bg_max, bg_min = compute_projection_pair(
        video_path,
        n_frames=n_frames,
        start_frac=start_frac,
        end_frac=end_frac,
    )
    save_background_png(bg_max, max_path)
    save_background_png(bg_min, min_path)
    detection_bg, trail_bg = detection_and_trail(bg_max, bg_min, polarity)
    return detection_bg, trail_bg, "computed"
