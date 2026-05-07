# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — detect.watershed                                       ║
# ║  « split merged blobs via watershed-on-distance-transform »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The chain-rule splitter (pylace.detect.chain) cuts an oversized ║
# ║  contour perpendicular to its major axis at the centroid. That   ║
# ║  works for two flies touching head-to-tail but cannot resolve    ║
# ║  side-by-side merges (the cut goes the wrong way) or three-way  ║
# ║  contacts (only ever produces two halves). This module does the  ║
# ║  general case: build a binary mask of the merged blob, take its  ║
# ║  distance transform, find local maxima as one foreground seed    ║
# ║  per fly, mark the rest of the blob as "unknown", and run        ║
# ║  cv2.watershed. The basin boundary lands roughly along the       ║
# ║  thinnest neck connecting the flies — exactly where a human      ║
# ║  would cut. Returns one contour per basin.                       ║
# ║                                                                  ║
# ║  References: Beucher & Meyer (ImageJ standard); Soille,          ║
# ║  *Morphological Image Analysis* (2003); the canonical OpenCV     ║
# ║  watershed tutorial. The chain rule is kept as a fallback for    ║
# ║  cases where watershed finds < 2 peaks (which means the blob is  ║
# ║  oversized but visually featureless — typically a chain merge    ║
# ║  with no neck constriction yet, where chain's major-axis cut is  ║
# ║  still a reasonable bet).                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Watershed-on-distance-transform splitter for merged-fly blobs."""

from __future__ import annotations

import cv2
import numpy as np

from pylace.detect.chain import (
    DEFAULT_AREA_RATIO_THRESHOLD,
    DEFAULT_LEARN_FRAMES,
    ELLIPSE_FIT_MIN_POINTS,
    split_along_major_axis,
)

DEFAULT_PEAK_MIN_DISTANCE_PX = 8
"""Local-maximum suppression radius in pixels for the distance-transform
peaks used as watershed seeds. Set well below the inter-fly centroid
distance and above the half-width of a single fly. ~ 8 px is a sensible
default for a typical Drosophila body in a 512–1024 px arena view; raise
this for larger animals or higher-resolution recordings."""

DEFAULT_PEAK_MIN_DEPTH_PX = 2
"""Minimum distance-transform value a candidate peak must reach. Filters
out the per-pixel jitter peaks of an irregular blob boundary."""


class WatershedSplitter:
    """Stateful splitter for oversized merged-fly contours.

    Public surface mirrors :class:`pylace.detect.chain.ChainSplitter`:
    same constructor knobs, same ``maybe_split(contours, fg_mask)``
    method. The ``fg_mask`` argument is required (the watershed needs
    pixel-level support); ``ChainSplitter.maybe_split`` accepts but
    ignores it for signature uniformity, so the calling code does not
    need to branch on which splitter it holds.
    """

    def __init__(
        self,
        *,
        expected_animal_area_px: float | None = None,
        area_ratio_threshold: float = DEFAULT_AREA_RATIO_THRESHOLD,
        learn_frames: int = DEFAULT_LEARN_FRAMES,
        peak_min_distance_px: int = DEFAULT_PEAK_MIN_DISTANCE_PX,
        peak_min_depth_px: float = DEFAULT_PEAK_MIN_DEPTH_PX,
        chain_fallback: bool = True,
    ) -> None:
        if area_ratio_threshold <= 1.0:
            raise ValueError("area_ratio_threshold must be > 1.")
        if learn_frames < 1:
            raise ValueError("learn_frames must be >= 1.")
        if peak_min_distance_px < 1:
            raise ValueError("peak_min_distance_px must be >= 1.")
        if peak_min_depth_px < 0:
            raise ValueError("peak_min_depth_px must be >= 0.")
        if expected_animal_area_px is not None and expected_animal_area_px <= 0:
            raise ValueError("expected_animal_area_px must be > 0.")

        self._area_ratio_threshold = float(area_ratio_threshold)
        self._learn_frames = int(learn_frames)
        self._peak_min_distance_px = int(peak_min_distance_px)
        self._peak_min_depth_px = float(peak_min_depth_px)
        self._chain_fallback = bool(chain_fallback)
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

    def maybe_split(
        self,
        contours: list[np.ndarray],
        fg_mask: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Return contours with each oversized one replaced by its sub-blobs."""
        if self._learning:
            self._observe(contours)
            return contours
        if fg_mask is None:
            raise ValueError(
                "WatershedSplitter requires fg_mask (post-morphology binary "
                "image of the foreground).",
            )
        threshold = self._area_ratio_threshold * float(self.expected_animal_area_px)
        out: list[np.ndarray] = []
        for c in contours:
            if cv2.contourArea(c) <= threshold:
                out.append(c)
                continue
            sub = watershed_split(
                c, fg_mask,
                peak_min_distance_px=self._peak_min_distance_px,
                peak_min_depth_px=self._peak_min_depth_px,
            )
            if len(sub) >= 2:
                out.extend(sub)
            elif self._chain_fallback:
                fallback = split_along_major_axis(c)
                out.extend(fallback if fallback else [c])
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


def watershed_split(
    contour: np.ndarray,
    fg_mask: np.ndarray,
    *,
    peak_min_distance_px: int = DEFAULT_PEAK_MIN_DISTANCE_PX,
    peak_min_depth_px: float = DEFAULT_PEAK_MIN_DEPTH_PX,
) -> list[np.ndarray]:
    """Split one merged contour via watershed-on-distance-transform.

    Args:
        contour: The oversized contour to split.
        fg_mask: Post-morphology binary foreground image (same H × W as
            the source frame). Used only to size the working canvases;
            the splitter rasterises ``contour`` itself and does not
            depend on whether other blobs are present in the mask.
        peak_min_distance_px: Local-maximum suppression radius for the
            distance-transform peaks used as watershed seeds.
        peak_min_depth_px: Minimum distance-transform value a candidate
            peak must reach.

    Returns:
        ``[contour_per_basin]`` with at least two basins, or ``[]`` when
        the splitter cannot identify two distinct peaks (caller should
        fall back to keeping the original contour or applying the
        chain rule).
    """
    h, w = fg_mask.shape[:2]
    blob = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(blob, [contour], -1, 255, thickness=-1)
    if int(blob.sum()) == 0:
        return []

    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)

    # Local-maximum mask: a pixel matches the dilated max in its
    # (2r+1)² neighbourhood AND exceeds the depth threshold.
    k = peak_min_distance_px * 2 + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    dilated = cv2.dilate(dist, kernel)
    is_peak = (dist == dilated) & (dist > peak_min_depth_px)
    n_peaks, peak_labels = cv2.connectedComponents(
        is_peak.astype(np.uint8), connectivity=8,
    )
    # n_peaks counts background as label 0, so n_peaks-1 distinct peaks.
    if n_peaks - 1 < 2:
        return []

    # Build watershed markers:
    #   0  = unknown (the watershed labels these from the seeds)
    #   1  = certain background (outside the slightly-dilated blob)
    #   2+ = one seed per peak
    markers = np.zeros((h, w), dtype=np.int32)
    bg_dilation = cv2.dilate(
        blob, np.ones((3, 3), np.uint8), iterations=2,
    )
    markers[bg_dilation == 0] = 1
    for lbl in range(1, n_peaks):
        markers[peak_labels == lbl] = lbl + 1

    bgr = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
    cv2.watershed(bgr, markers)

    out: list[np.ndarray] = []
    for lbl in range(2, n_peaks + 1):
        basin = np.where(markers == lbl, 255, 0).astype(np.uint8)
        if int(basin.sum()) == 0:
            continue
        contours, _ = cv2.findContours(
            basin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) <= 0:
            continue
        if len(largest) < ELLIPSE_FIT_MIN_POINTS:
            continue
        out.append(largest)
    return out if len(out) >= 2 else []


__all__ = [
    "DEFAULT_PEAK_MIN_DEPTH_PX",
    "DEFAULT_PEAK_MIN_DISTANCE_PX",
    "WatershedSplitter",
    "watershed_split",
]
