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
    """Ordered collection of ROIs with a top-level merge/split mode."""

    rois: list[ROI] = field(default_factory=list)
    mode: ROIMode = "merge"

    def __post_init__(self) -> None:
        if self.mode not in ("merge", "split"):
            raise ValueError(
                f"ROISet.mode must be 'merge' or 'split', got {self.mode!r}.",
            )

    def is_empty(self) -> bool:
        return len(self.rois) == 0

    def add(self, roi: ROI) -> None:
        self.rois.append(roi)

    def remove_at(self, index: int) -> None:
        if 0 <= index < len(self.rois):
            del self.rois[index]
