# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — detect.chain                                           ║
# ║  « split a "chained" two-fly blob back into two ellipses »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  When two flies pile up they merge into a single contour with    ║
# ║  area ≈ 2× a single fly. cv2.fitEllipse on that merged contour   ║
# ║  yields one big ellipse covering both, so the tracker loses one  ║
# ║  fly until the chain breaks. This module enacts the LACE-paper   ║
# ║  chaining solution (Problems 5 / 6 / 7): when a contour's area   ║
# ║  exceeds ``area_ratio_threshold × expected_animal_area_px``, cut ║
# ║  it perpendicular to the major axis at the centroid and refit    ║
# ║  one ellipse per half.                                           ║
# ║                                                                  ║
# ║  ``expected_animal_area_px`` can be supplied up front or learned ║
# ║  online from the first ``learn_frames`` frames' median contour   ║
# ║  area — chains are rare in any specific 50-frame window so the   ║
# ║  median is robust to the few that slip in.                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Chain-blob splitter (LACE Problems 5 / 6 / 7)."""

from __future__ import annotations

import math

import cv2
import numpy as np

DEFAULT_AREA_RATIO_THRESHOLD = 1.5
DEFAULT_LEARN_FRAMES = 50
ELLIPSE_FIT_MIN_POINTS = 5


class ChainSplitter:
    """Stateful wrapper that splits oversized contours, optionally auto-learning."""

    def __init__(
        self,
        *,
        expected_animal_area_px: float | None = None,
        area_ratio_threshold: float = DEFAULT_AREA_RATIO_THRESHOLD,
        learn_frames: int = DEFAULT_LEARN_FRAMES,
    ) -> None:
        if area_ratio_threshold <= 1.0:
            raise ValueError("area_ratio_threshold must be > 1.")
        if learn_frames < 1:
            raise ValueError("learn_frames must be >= 1.")
        if expected_animal_area_px is not None and expected_animal_area_px <= 0:
            raise ValueError("expected_animal_area_px must be > 0.")

        self._area_ratio_threshold = float(area_ratio_threshold)
        self._learn_frames = int(learn_frames)
        self.expected_animal_area_px: float | None = (
            float(expected_animal_area_px)
            if expected_animal_area_px is not None
            else None
        )
        self._learning = expected_animal_area_px is None
        self._learned_areas: list[float] = []
        self._frames_seen = 0

    @property
    def is_learning(self) -> bool:
        return self._learning

    def maybe_split(self, contours: list[np.ndarray]) -> list[np.ndarray]:
        """Return contours with any oversized one replaced by its two halves."""
        if self._learning:
            self._observe(contours)
            return contours
        threshold = self._area_ratio_threshold * float(self.expected_animal_area_px)
        out: list[np.ndarray] = []
        for c in contours:
            if cv2.contourArea(c) > threshold:
                halves = split_along_major_axis(c)
                out.extend(halves if halves else [c])
            else:
                out.append(c)
        return out

    # ── Internal ───────────────────────────────────────────────────────

    def _observe(self, contours: list[np.ndarray]) -> None:
        for c in contours:
            self._learned_areas.append(float(cv2.contourArea(c)))
        self._frames_seen += 1
        if self._frames_seen >= self._learn_frames and self._learned_areas:
            self.expected_animal_area_px = float(np.median(self._learned_areas))
            self._learning = False


def split_along_major_axis(contour: np.ndarray) -> list[np.ndarray]:
    """Cut a contour perpendicular to its major axis at the centroid.

    Returns ``[half_a, half_b]`` if both halves have at least
    ``ELLIPSE_FIT_MIN_POINTS`` points. Returns ``[]`` when the split
    is degenerate (too few points, zero area, axes ratio invalid).
    The caller decides how to handle ``[]`` — typically falling back
    to keeping the original contour unsplit.
    """
    if len(contour) < ELLIPSE_FIT_MIN_POINTS * 2:
        return []
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return []
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    (_centre, axes, angle_deg) = cv2.fitEllipse(contour)
    if axes[0] <= 0 or axes[1] <= 0:
        return []
    # cv2.fitEllipse: (axes[0], axes[1]) = (width, height) along the
    # rotated rectangle's axes; ``angle_deg`` is the rotation of the
    # width axis from horizontal. The major axis is whichever of the
    # two is larger.
    major_angle_deg = angle_deg if axes[0] >= axes[1] else angle_deg + 90.0
    angle_rad = math.radians(major_angle_deg)
    axis_dx = math.cos(angle_rad)
    axis_dy = math.sin(angle_rad)

    pts = contour.reshape(-1, 2).astype(np.float64)
    projection = (pts[:, 0] - cx) * axis_dx + (pts[:, 1] - cy) * axis_dy
    half_a_pts = pts[projection >= 0]
    half_b_pts = pts[projection < 0]
    if (
        len(half_a_pts) < ELLIPSE_FIT_MIN_POINTS
        or len(half_b_pts) < ELLIPSE_FIT_MIN_POINTS
    ):
        return []
    half_a = half_a_pts.reshape(-1, 1, 2).astype(np.int32)
    half_b = half_b_pts.reshape(-1, 1, 2).astype(np.int32)
    return [half_a, half_b]


__all__ = [
    "ChainSplitter",
    "DEFAULT_AREA_RATIO_THRESHOLD",
    "DEFAULT_LEARN_FRAMES",
    "split_along_major_axis",
]
