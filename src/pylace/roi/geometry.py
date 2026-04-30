# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi.geometry                                           ║
# ║  « ROI dataclass + ROISet, reusing annotator shape primitives »  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Each ROI wraps an annotator shape (Circle / Rectangle /         ║
# ║  Polygon) with a per-ROI add/subtract op and an optional         ║
# ║  human-readable label. ROISet bundles a list of ROIs with a      ║
# ║  top-level mode (merge | split). Split mode is parsed and        ║
# ║  serialised but rejected by current consumers — full support     ║
# ║  lands in v2 alongside identity tracking.                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""ROI dataclasses with shape + boolean op."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pylace.annotator.geometry import Arena

ROIOperation = Literal["add", "subtract"]
ROIMode = Literal["merge", "split"]


@dataclass
class ROI:
    """One region of interest: shape + boolean op + optional label."""

    shape: Arena
    operation: ROIOperation = "add"
    label: str = ""

    def __post_init__(self) -> None:
        if self.operation not in ("add", "subtract"):
            raise ValueError(
                f"ROI.operation must be 'add' or 'subtract', got {self.operation!r}.",
            )


@dataclass
class ROISet:
    """Ordered collection of ROIs with a top-level merge/split mode.

    ``freehand_mask`` is an optional ``uint8`` raster (0 / 255) the
    same shape as the video frame. It is painted by the brush / eraser
    tools and unions into the combined mask in merge mode. It is also
    surfaced as a single extra ``("freehand", mask)`` pair in
    :func:`pylace.roi.mask.build_split_masks` so split-mode runs can
    pick it up too. Persisted as a sibling PNG sidecar — see
    :mod:`pylace.roi.sidecar`.
    """

    rois: list[ROI] = field(default_factory=list)
    mode: ROIMode = "merge"
    freehand_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("merge", "split"):
            raise ValueError(
                f"ROISet.mode must be 'merge' or 'split', got {self.mode!r}.",
            )

    def is_empty(self) -> bool:
        return len(self.rois) == 0 and not self.has_freehand_mask()

    def has_freehand_mask(self) -> bool:
        return self.freehand_mask is not None and bool(self.freehand_mask.any())

    def add(self, roi: ROI) -> None:
        self.rois.append(roi)

    def remove_at(self, index: int) -> None:
        if 0 <= index < len(self.rois):
            del self.rois[index]
