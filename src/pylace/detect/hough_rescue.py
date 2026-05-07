# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — detect.hough_rescue                                    ║
# ║  « LACE-paper rescue: ellipse candidates from contour sub-arcs » ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  When the ordinary detector finds fewer than N detections in a   ║
# ║  fixed-N run — typically because two flies merged into one       ║
# ║  contour and neither watershed nor the chain rule could split    ║
# ║  them — this module generates additional candidate ellipses by   ║
# ║  sliding a window across each foreground contour and fitting     ║
# ║  cv2.fitEllipse to the windowed point cloud. Candidates whose    ║
# ║  area sits within the tolerance band around                      ║
# ║  ``expected_animal_area_px`` are retained, then non-maximum-     ║
# ║  suppressed against each other and against the existing          ║
# ║  detections, and finally appended (up to the per-frame deficit). ║
# ║                                                                  ║
# ║  This is the simplified-but-faithful version of Geurten 2022's   ║
# ║  three-vote candidate scheme: position fit (covered by NMS),     ║
# ║  surface match (the area-tolerance gate), and contour            ║
# ║  similarity (implicit in fitEllipse's residual). The full        ║
# ║  parameter-space Hough accumulator is overkill for fixed-N       ║
# ║  arena tracking and would slow the inner loop; this candidate-   ║
# ║  and-NMS variant captures the same intent at a fraction of the  ║
# ║  cost.                                                           ║
# ║                                                                  ║
# ║  Stage order in the per-frame pipeline:                          ║
# ║    bg-sub → threshold → morph → contours → splitter →            ║
# ║    shape filter → mask gate → **HoughRescue (this module)** →    ║
# ║    Tracker.step (Hungarian)                                      ║
# ║  Running before Hungarian is the whole point: by giving the      ║
# ║  Hungarian N observations instead of N-1, the assignment doesn't ║
# ║  have to leave one track unmatched and can place the rescued     ║
# ║  ellipse with the right ID.                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Hough-style rescue: ellipse candidates from foreground sub-arcs."""

from __future__ import annotations

import cv2
import numpy as np

from pylace.detect.frame import Detection

DEFAULT_WINDOW_FRACTION = 0.30
"""Sub-arc window size as a fraction of the contour's point count."""

DEFAULT_WINDOW_MIN_POINTS = 30
"""Hard minimum window size in contour points (overrides the fraction
when the contour is short)."""

DEFAULT_AREA_TOLERANCE = 0.50
"""± fraction around ``expected_animal_area_px`` that a candidate's
fitted-ellipse area must fall within to be considered a fly. 0.50 means
half-to-1.5× expected — generous, because fitEllipse on a partial arc
estimates the *full* ellipse, and a fly's silhouette area in pixels
varies with morphology."""

DEFAULT_LEARN_FRAMES = 50
"""Frames-of-observation before auto-learning ``expected_animal_area_px``
from the median of all observed detection areas."""


class HoughRescuer:
    """Add up to ``target_n - len(detections)`` ellipse candidates per frame.

    Stateful: like ChainSplitter, the rescuer auto-learns the expected
    animal area from the first ``learn_frames`` frames if none was
    supplied. During the learning window the rescuer is a pass-through.
    """

    def __init__(
        self,
        target_n: int,
        *,
        expected_animal_area_px: float | None = None,
        window_fraction: float = DEFAULT_WINDOW_FRACTION,
        window_min_points: int = DEFAULT_WINDOW_MIN_POINTS,
        area_tolerance: float = DEFAULT_AREA_TOLERANCE,
        learn_frames: int = DEFAULT_LEARN_FRAMES,
    ) -> None:
        if target_n < 1:
            raise ValueError("target_n must be >= 1.")
        if window_min_points < 5:
            raise ValueError("window_min_points must be >= 5 (ellipse fit minimum).")
        if not (0.0 < window_fraction <= 1.0):
            raise ValueError("window_fraction must be in (0, 1].")
        if not (0.0 < area_tolerance < 1.0):
            raise ValueError("area_tolerance must be in (0, 1).")
        if learn_frames < 1:
            raise ValueError("learn_frames must be >= 1.")
        if expected_animal_area_px is not None and expected_animal_area_px <= 0:
            raise ValueError("expected_animal_area_px must be > 0.")
        self._target_n = int(target_n)
        self._window_fraction = float(window_fraction)
        self._window_min_points = int(window_min_points)
        self._area_tolerance = float(area_tolerance)
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

    def maybe_rescue(
        self,
        detections: list[Detection],
        fg_mask: np.ndarray,
    ) -> list[Detection]:
        """Append rescue candidates if ``len(detections) < target_n``."""
        if self._learning:
            self._observe(detections)
            return detections
        if len(detections) >= self._target_n:
            return detections
        n_missing = self._target_n - len(detections)
        expected = float(self.expected_animal_area_px)
        # NMS radius: roughly the expected fly's body radius.
        nms_radius_px = float(np.sqrt(max(1.0, expected) / np.pi))

        # Generate candidates from contour sub-arcs.
        binary = fg_mask if fg_mask.dtype == np.uint8 else (
            fg_mask.astype(np.uint8) * 255
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )
        cands = generate_candidates(
            contours,
            expected_area=expected,
            area_tolerance=self._area_tolerance,
            window_fraction=self._window_fraction,
            window_min_points=self._window_min_points,
        )
        # Score by closeness to expected area (1.0 = perfect fit).
        cands.sort(
            key=lambda d: abs(d.area_px - expected),
        )
        # NMS against existing detections + already-accepted candidates.
        accepted: list[Detection] = []
        keep_centres: list[tuple[float, float]] = [
            (d.cx, d.cy) for d in detections
        ]
        for cand in cands:
            if any(
                _hypot(cand.cx - x, cand.cy - y) < nms_radius_px
                for x, y in keep_centres
            ):
                continue
            accepted.append(cand)
            keep_centres.append((cand.cx, cand.cy))
            if len(accepted) >= n_missing:
                break
        return list(detections) + accepted

    def _observe(self, detections: list[Detection]) -> None:
        for d in detections:
            self._learned_areas.append(float(d.area_px))
        self._frames_seen += 1
        if self._frames_seen >= self._learn_frames and self._learned_areas:
            self.expected_animal_area_px = float(np.median(self._learned_areas))
            self._learning = False


# ─────────────────────────────────────────────────────────────────
#  Pure helpers
# ─────────────────────────────────────────────────────────────────


def generate_candidates(
    contours: list[np.ndarray],
    *,
    expected_area: float,
    area_tolerance: float,
    window_fraction: float = DEFAULT_WINDOW_FRACTION,
    window_min_points: int = DEFAULT_WINDOW_MIN_POINTS,
) -> list[Detection]:
    """Slide a window over each contour and fit ellipses; filter by area."""
    out: list[Detection] = []
    lo = expected_area * (1.0 - area_tolerance)
    hi = expected_area * (1.0 + area_tolerance)
    for c in contours:
        n = len(c)
        if n < 5:
            continue
        # Whole-contour fit first — covers the case where the contour
        # itself IS one fly that was somehow missed earlier.
        det = _fit_ellipse_to_arc(c)
        if det is not None and lo <= det.area_px <= hi:
            out.append(det)
        if n < window_min_points * 2:
            continue
        # Slide windows of len max(window_min_points, fraction × n).
        win = max(window_min_points, int(window_fraction * n))
        if win >= n:
            continue
        step = max(5, win // 4)
        for i in range(0, n - win + 1, step):
            sub = c[i:i + win]
            det = _fit_ellipse_to_arc(sub)
            if det is None:
                continue
            if lo <= det.area_px <= hi:
                out.append(det)
    return out


def _fit_ellipse_to_arc(arc: np.ndarray) -> Detection | None:
    if len(arc) < 5:
        return None
    try:
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(arc)
    except cv2.error:
        return None
    major = max(axis_a, axis_b)
    minor = min(axis_a, axis_b)
    if minor <= 0 or major <= 0:
        return None
    # cv2.fitEllipse gives full axes, so area = π/4 · major · minor.
    area = float(np.pi * 0.25 * major * minor)
    perimeter = _ramanujan_perimeter(major * 0.5, minor * 0.5)
    return Detection(
        cx=float(cx), cy=float(cy),
        area_px=area,
        perimeter_px=float(perimeter),
        solidity=1.0,  # an ellipse is convex by construction
        major_axis_px=float(major),
        minor_axis_px=float(minor),
        orientation_deg=float(angle),
        contour=None,
        track_id=-1,
    )


def _ramanujan_perimeter(a: float, b: float) -> float:
    """Ramanujan's first approximation to the ellipse perimeter."""
    h = ((a - b) / (a + b)) ** 2 if (a + b) > 0 else 0.0
    return float(np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h))))


def _hypot(dx: float, dy: float) -> float:
    return float(np.hypot(dx, dy))


__all__ = [
    "DEFAULT_AREA_TOLERANCE",
    "DEFAULT_LEARN_FRAMES",
    "DEFAULT_WINDOW_FRACTION",
    "DEFAULT_WINDOW_MIN_POINTS",
    "HoughRescuer",
    "generate_candidates",
]
