# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.appearance                                     ║
# ║  « pose-normalised fingerprints + ToxTrac continuity scoring »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Hand-crafted per-fly appearance features, used by the audit as  ║
# ║  a tiebreaker when the Kalman cost cannot discriminate two       ║
# ║  permutations after a merge.                                     ║
# ║                                                                  ║
# ║  Two ingredients, both pure NumPy + one cv2.warpAffine:          ║
# ║                                                                  ║
# ║  1. Pose-normalised intensity patch (Perez-Escudero et al.,      ║
# ║     Nat Methods 2014, "idTracker" classical version). Each       ║
# ║     detection's bounding region is rotated to canonical          ║
# ║     orientation (major axis horizontal) and resized to a small   ║
# ║     fixed-size patch. A running median per ID over confident     ║
# ║     frames acts as the fly's "fingerprint". The 180-degree       ║
# ║     head/abdomen ambiguity from cv2.fitEllipse is handled at     ║
# ║     match time by comparing against both the patch and its       ║
# ║     180-degree-rotated copy and taking the better score.         ║
# ║                                                                  ║
# ║  2. ToxTrac-style shape-continuity scoring (Rodriguez et al.,    ║
# ║     Methods Ecol Evol 2018). Per-fly running medians of          ║
# ║     ``axis_ratio = major / minor`` and ``area_px`` over          ║
# ║     confident frames. At every merge-split boundary, score each  ║
# ║     permutation by the deviation of post-event values from each  ║
# ║     pre-event ID's running median.                               ║
# ║                                                                  ║
# ║  Both feature streams ride on top of the existing Kalman-        ║
# ║  Mahalanobis cost; the audit combines them via tunable weights.  ║
# ║  This module ships the pure functions; integration lives in      ║
# ║  audit.py.                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Pose-normalised fingerprints and continuity scoring for the audit."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

DEFAULT_PATCH_H = 16
"""Pose-normalised patch height in pixels (perpendicular to the body axis)."""

DEFAULT_PATCH_W = 32
"""Pose-normalised patch width in pixels (along the body axis)."""

DEFAULT_PATCH_PAD_FACTOR = 1.5
"""Source-region size as a multiple of the fitted axes. 1.5 leaves margin
for fitEllipse error and small wing extensions without including too
much background."""

DEFAULT_HISTORY_FRAMES = 250
"""Frames of confident observation that feed each ID's running median.
~ 10 s at 25 fps. Long enough to average out per-frame noise, short
enough that the running median tracks slow appearance drift (lighting,
shadows, fly position relative to the lens)."""

DEFAULT_CONFIDENT_AREA_TOL = 0.30
"""+/- fraction of expected_animal_area_px within which a detection's
area must fall to count as confident. 0.30 keeps fly-on-edge and
mild-overlap frames out of the running medians."""

DEFAULT_CONFIDENT_SEPARATION_FACTOR = 2.0
"""Multiple of the contact threshold below which a frame is *not*
counted as confident. With contact_threshold = 5 mm and factor = 2.0,
flies must be > 10 mm apart for a frame to feed the fingerprint."""


# ─────────────────────────────────────────────────────────────────
#  Pose-normalised patch extraction
# ─────────────────────────────────────────────────────────────────


def extract_patch(
    frame_gray: np.ndarray,
    *,
    cx: float,
    cy: float,
    orientation_deg: float,
    major_axis_px: float,
    minor_axis_px: float,
    patch_h: int = DEFAULT_PATCH_H,
    patch_w: int = DEFAULT_PATCH_W,
    pad_factor: float = DEFAULT_PATCH_PAD_FACTOR,
) -> np.ndarray:
    """Crop a rotated rectangle around the fly and resize to ``(patch_h, patch_w)``.

    The source rectangle is sized ``pad_factor × major_axis_px`` along
    the body axis and ``pad_factor × minor_axis_px`` perpendicular to
    it, centred on ``(cx, cy)``. cv2.fitEllipse's angle convention has
    the unrotated ellipse with its major axis along +y, so we rotate
    the source by ``(90 - orientation_deg)`` degrees CCW to bring the
    major axis onto +x in the output.

    Returns a uint8 array. Pixels outside the source frame are filled
    with 0.
    """
    src_w_px = max(float(major_axis_px) * pad_factor, float(patch_w))
    src_h_px = max(float(minor_axis_px) * pad_factor, float(patch_h))
    angle_rad = np.deg2rad(90.0 - float(orientation_deg))
    cosA = float(np.cos(angle_rad))
    sinA = float(np.sin(angle_rad))
    hw = src_w_px * 0.5
    hh = src_h_px * 0.5
    # Three corners of the source rectangle (in image coordinates),
    # rotated around the centroid: top-left, top-right, bottom-right.
    src_corners = np.array(
        [
            [cx - hw * cosA + hh * sinA, cy - hw * sinA - hh * cosA],
            [cx + hw * cosA + hh * sinA, cy + hw * sinA - hh * cosA],
            [cx + hw * cosA - hh * sinA, cy + hw * sinA + hh * cosA],
        ],
        dtype=np.float32,
    )
    dst_corners = np.array(
        [[0.0, 0.0], [patch_w - 1.0, 0.0], [patch_w - 1.0, patch_h - 1.0]],
        dtype=np.float32,
    )
    M = cv2.getAffineTransform(src_corners, dst_corners)
    patch = cv2.warpAffine(
        frame_gray, M, (patch_w, patch_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if patch.dtype != np.uint8:
        patch = np.clip(patch, 0, 255).astype(np.uint8)
    return patch


def match_patch(query: np.ndarray, reference: np.ndarray) -> float:
    """RMSE between ``query`` and the better-matching of ``reference`` / its 180-deg flip.

    The flip handles the cv2.fitEllipse head/abdomen ambiguity at zero
    cost: we don't need to disambiguate heading first, we just compare
    against both orientations and take the closer one.

    Lower is better. NaN on either input returns +inf.
    """
    if query.size == 0 or reference.size == 0:
        return float("inf")
    if query.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: query {query.shape} vs reference {reference.shape}",
        )
    q = query.astype(np.float32)
    r = reference.astype(np.float32)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(r))):
        return float("inf")
    r_flipped = np.flip(r, axis=(0, 1))
    rmse_a = float(np.sqrt(np.mean((q - r) ** 2)))
    rmse_b = float(np.sqrt(np.mean((q - r_flipped) ** 2)))
    return min(rmse_a, rmse_b)


# ─────────────────────────────────────────────────────────────────
#  Confident-frame detection
# ─────────────────────────────────────────────────────────────────


def is_confident_frame(
    cx: np.ndarray,
    cy: np.ndarray,
    area: np.ndarray,
    *,
    n_animals: int,
    min_pairwise_distance_px: float,
    expected_area_px: float,
    area_tol: float = DEFAULT_CONFIDENT_AREA_TOL,
) -> bool:
    """Return True iff this frame is safe to feed the running medians.

    Three conditions, all required:
      1. Detection count equals ``n_animals``.
      2. All pairwise centroid distances exceed
         ``min_pairwise_distance_px`` (no merges or near-merges).
      3. Every detection's area is within ``area_tol`` of
         ``expected_area_px`` (no oversized merged blobs that
         survived the splitter, no clipped detections).
    """
    n = len(cx)
    if n != n_animals or len(cy) != n or len(area) != n:
        return False
    if min_pairwise_distance_px > 0 and n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.hypot(cx[i] - cx[j], cy[i] - cy[j]))
                if d <= min_pairwise_distance_px:
                    return False
    if expected_area_px > 0:
        lo = expected_area_px * (1.0 - area_tol)
        hi = expected_area_px * (1.0 + area_tol)
        if not np.all((np.asarray(area) >= lo) & (np.asarray(area) <= hi)):
            return False
    return True


# ─────────────────────────────────────────────────────────────────
#  ToxTrac-style continuity scoring
# ─────────────────────────────────────────────────────────────────


def axis_ratio_continuity_score(
    major_axis_px: float, minor_axis_px: float, ref_axis_ratio: float,
) -> float:
    """Normalised absolute difference in axis ratio.

    Returns ``|axis_ratio - ref| / ref``. A value of 0 means perfect
    match; 1.0 means the candidate's axis ratio is twice (or half)
    the reference. Returns 0 when either input is degenerate (caller
    treats this as "no information").
    """
    if minor_axis_px <= 0 or ref_axis_ratio <= 0:
        return 0.0
    axis_ratio = float(major_axis_px) / float(minor_axis_px)
    return float(abs(axis_ratio - ref_axis_ratio) / ref_axis_ratio)


def area_continuity_score(area_px: float, ref_area_px: float) -> float:
    """Normalised absolute difference in area. Same convention as axis_ratio_score."""
    if ref_area_px <= 0:
        return 0.0
    return float(abs(float(area_px) - float(ref_area_px)) / float(ref_area_px))


# ─────────────────────────────────────────────────────────────────
#  Running-median state per ID
# ─────────────────────────────────────────────────────────────────


class RunningMedianFingerprint:
    """Per-track running median of patches + scalar features over confident frames.

    Maintains a fixed-size FIFO of the most recent ``history_frames``
    confident observations per track. Median is recomputed on demand
    rather than incrementally — the history is small (250 entries) so
    np.median is fine.
    """

    def __init__(self, history_frames: int = DEFAULT_HISTORY_FRAMES) -> None:
        if history_frames < 1:
            raise ValueError("history_frames must be >= 1.")
        self._history_frames = int(history_frames)
        self._patches: dict[int, list[np.ndarray]] = {}
        self._areas: dict[int, list[float]] = {}
        self._axis_ratios: dict[int, list[float]] = {}

    def update(
        self, track_id: int, *, patch: np.ndarray, area_px: float,
        major_axis_px: float, minor_axis_px: float,
    ) -> None:
        """Record one confident observation for ``track_id``."""
        if minor_axis_px <= 0:
            return
        axis_ratio = float(major_axis_px) / float(minor_axis_px)
        self._patches.setdefault(track_id, []).append(patch.copy())
        self._areas.setdefault(track_id, []).append(float(area_px))
        self._axis_ratios.setdefault(track_id, []).append(axis_ratio)
        for store in (self._patches, self._areas, self._axis_ratios):
            buf = store.get(track_id)
            if buf is not None and len(buf) > self._history_frames:
                del buf[: len(buf) - self._history_frames]

    def median_patch(self, track_id: int) -> np.ndarray | None:
        buf = self._patches.get(track_id)
        if not buf:
            return None
        stack = np.stack(buf, axis=0).astype(np.float32)
        med = np.median(stack, axis=0)
        return np.clip(med, 0, 255).astype(np.uint8)

    def median_area(self, track_id: int) -> float | None:
        buf = self._areas.get(track_id)
        return float(np.median(buf)) if buf else None

    def median_axis_ratio(self, track_id: int) -> float | None:
        buf = self._axis_ratios.get(track_id)
        return float(np.median(buf)) if buf else None

    def known_ids(self) -> list[int]:
        return sorted(self._patches.keys())


__all__ = [
    "DEFAULT_CONFIDENT_AREA_TOL",
    "DEFAULT_CONFIDENT_SEPARATION_FACTOR",
    "DEFAULT_HISTORY_FRAMES",
    "DEFAULT_PATCH_H",
    "DEFAULT_PATCH_PAD_FACTOR",
    "DEFAULT_PATCH_W",
    "RunningMedianFingerprint",
    "area_continuity_score",
    "axis_ratio_continuity_score",
    "extract_patch",
    "is_confident_frame",
    "match_patch",
]
