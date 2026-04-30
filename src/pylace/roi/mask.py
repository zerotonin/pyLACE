# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi.mask                                               ║
# ║  « combine ROIs into a single boolean mask via add/subtract »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Process ROIs in order. The first ROI is implicitly an "add"     ║
# ║  regardless of its declared operation, so a subtract-only set    ║
# ║  is not silently empty. Subsequent ROIs union (add) or remove    ║
# ║  (subtract) from the running mask.                               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Combine ROIs into a single boolean mask."""

from __future__ import annotations

import numpy as np

from pylace.detect.arena_mask import arena_mask
from pylace.roi.geometry import ROISet


def build_combined_mask(
    roi_set: ROISet, frame_size: tuple[int, int],
) -> np.ndarray:
    """Boolean mask that ``True`` inside the combined ROI region (merge mode).

    Args:
        roi_set:    ROIs in order; first is implicitly add.
        frame_size: ``(width, height)`` of the destination mask.

    Returns:
        Boolean ``(height, width)`` mask. All ``True`` if ``roi_set`` is
        empty (the caller may treat that as "no ROI filtering" and rely
        on the arena mask alone).
    """
    if roi_set.mode == "split":
        raise NotImplementedError(
            "build_combined_mask is for merge mode; call build_split_masks for split.",
        )

    width, height = frame_size
    if roi_set.is_empty():
        return np.ones((height, width), dtype=bool)

    mask = np.zeros((height, width), dtype=bool)
    for index, roi in enumerate(roi_set.rois):
        roi_pixels = arena_mask(roi.shape, frame_size)
        if index == 0 or roi.operation == "add":
            mask |= roi_pixels
        else:
            mask &= ~roi_pixels
    return mask


def build_split_masks(
    roi_set: ROISet, frame_size: tuple[int, int],
) -> list[tuple[str, np.ndarray]]:
    """One mask per ROI for split-mode runs; returns ``(label, mask)`` pairs.

    Each ROI becomes its own sub-video. Operation is ignored in split
    mode — every ROI is treated as a positive sub-region; subtract ROIs
    are skipped with a comment-level warning baked into the label
    (subtractive sub-videos are not meaningful).

    Labels default to ``f"roi_{index}"`` if the ROI has no explicit
    label set in the GUI.
    """
    if roi_set.mode != "split":
        raise ValueError("build_split_masks requires ROISet.mode='split'.")

    out: list[tuple[str, np.ndarray]] = []
    for index, roi in enumerate(roi_set.rois):
        if roi.operation == "subtract":
            continue
        label = roi.label.strip() or f"roi_{index}"
        out.append((label, arena_mask(roi.shape, frame_size)))
    return out
