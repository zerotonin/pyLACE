# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi.auto_roi                                           ║
# ║  « auto-generate ROIs from the trail-bg vs detection-bg diff »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Where the animal lives most of the time differs from where the  ║
# ║  arena would look if no animal were present. Per-pixel absolute  ║
# ║  difference between the max- and min-projection backgrounds      ║
# ║  surfaces that footprint, and morphological cleanup converts it  ║
# ║  into ROI polygons that buffer the trail by a few pixels —       ║
# ║  enough that detection inside the ROI captures the animal even   ║
# ║  on the rare frame it strays a step.                             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Auto-ROI generation from the (max, min) background pair."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pylace.annotator.geometry import Polygon
from pylace.roi.geometry import ROI


@dataclass
class AutoRoiParams:
    """Tunable knobs for :func:`auto_rois_from_diff`.

    Defaults are chosen so the resulting ROI sits a few pixels outside
    the actual trail — wide enough that a single off-trail step is
    still inside the ROI, narrow enough not to swallow noisy
    artefacts.
    """

    diff_threshold: int = 30
    morph_kernel: int = 5
    erode_iters: int = 1
    dilate_iters: int = 3
    min_area_px: int = 100
    simplify_epsilon: float = 2.0


def auto_rois_from_diff(
    bg_a: np.ndarray,
    bg_b: np.ndarray,
    arena_mask: np.ndarray,
    *,
    params: AutoRoiParams | None = None,
) -> list[ROI]:
    """Return add-ROIs covering pixels that differ between two backgrounds.

    Args:
        bg_a, bg_b: ``uint8`` grayscale backgrounds (typically the
            max- and min-projection sidecars). Order does not matter
            — :func:`cv2.absdiff` is symmetric.
        arena_mask: Boolean ``(H, W)`` arena mask; ROIs are clipped to
            stay inside the arena.
        params: Optional :class:`AutoRoiParams` overrides.

    Returns:
        List of Polygon ROIs (operation=``add``, label=``auto``). Empty
        if no contour exceeds ``min_area_px``.
    """
    cfg = params or AutoRoiParams()
    fg = _trail_mask(bg_a, bg_b, arena_mask, cfg)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [
        roi
        for contour in contours
        if (roi := _polygon_from_contour(contour, cfg)) is not None
    ]


# ─────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────

def _trail_mask(
    bg_a: np.ndarray, bg_b: np.ndarray, arena_mask: np.ndarray,
    cfg: AutoRoiParams,
) -> np.ndarray:
    if bg_a.shape != bg_b.shape:
        raise ValueError(
            f"Shape mismatch: bg_a {bg_a.shape} vs bg_b {bg_b.shape}.",
        )
    if bg_a.shape != arena_mask.shape:
        raise ValueError(
            f"arena_mask shape {arena_mask.shape} differs from bg "
            f"{bg_a.shape}.",
        )
    diff = cv2.absdiff(bg_a, bg_b)
    fg = (diff > cfg.diff_threshold).astype(np.uint8) * 255
    fg = cv2.bitwise_and(fg, arena_mask.astype(np.uint8) * 255)
    if cfg.morph_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel),
        )
        if cfg.erode_iters > 0:
            fg = cv2.erode(fg, kernel, iterations=cfg.erode_iters)
        if cfg.dilate_iters > 0:
            fg = cv2.dilate(fg, kernel, iterations=cfg.dilate_iters)
    return fg


def _polygon_from_contour(
    contour: np.ndarray, cfg: AutoRoiParams,
) -> ROI | None:
    if cv2.contourArea(contour) < cfg.min_area_px:
        return None
    approx = cv2.approxPolyDP(contour, cfg.simplify_epsilon, closed=True)
    if len(approx) < 3:
        return None
    vertices = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
    return ROI(shape=Polygon(vertices=vertices), operation="add", label="auto")


__all__ = ["AutoRoiParams", "auto_rois_from_diff"]
