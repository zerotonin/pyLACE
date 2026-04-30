# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi.constants                                          ║
# ║  « shared numbers, colours, paths for the ROI sub-package »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Single source of truth for ROI defaults, sidecar suffix, and    ║
# ║  the BGR colour codes used by the canvas overlay. Wong (2011)    ║
# ║  palette entries are mapped to ROI semantic roles so additive    ║
# ║  and subtractive ROIs stay distinguishable to colour-blind       ║
# ║  reviewers.                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Constants for the ROI sub-package."""

from __future__ import annotations

from typing import Final

# ┌────────────────────────────────────────────────────────────┐
# │ Wong (2011) palette  « colourblind-safe base colours »     │
# └────────────────────────────────────────────────────────────┘
WONG: Final[dict[str, str]] = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermilion":      "#D55E00",
    "reddish_purple": "#CC79A7",
}


def _hex_to_bgr(hex_str: str) -> tuple[int, int, int]:
    """Convert ``#RRGGBB`` to an OpenCV (B, G, R) tuple."""
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# ┌────────────────────────────────────────────────────────────┐
# │ ROI semantic colours « add = sky blue, subtract = orange » │
# └────────────────────────────────────────────────────────────┘
ROI_ADD_COLOR_BGR: Final[tuple[int, int, int]] = _hex_to_bgr(WONG["sky_blue"])
ROI_SUBTRACT_COLOR_BGR: Final[tuple[int, int, int]] = _hex_to_bgr(WONG["vermilion"])
ROI_SELECTED_COLOR_BGR: Final[tuple[int, int, int]] = _hex_to_bgr(WONG["yellow"])
ROI_DRAFT_COLOR_BGR: Final[tuple[int, int, int]] = _hex_to_bgr(WONG["bluish_green"])
ROI_KEPT_FILL_BGR: Final[tuple[int, int, int]] = _hex_to_bgr(WONG["bluish_green"])
ARENA_OUTLINE_COLOR_BGR: Final[tuple[int, int, int]] = _hex_to_bgr(WONG["reddish_purple"])

# ┌────────────────────────────────────────────────────────────┐
# │ Sidecar conventions  « <video>.pylace_rois.json »          │
# └────────────────────────────────────────────────────────────┘
SIDECAR_SUFFIX: Final[str] = ".pylace_rois.json"
SCHEMA_VERSION: Final[int] = 1

# ┌────────────────────────────────────────────────────────────┐
# │ UI defaults  « brush size, snap radius, line widths »      │
# └────────────────────────────────────────────────────────────┘
SNAP_RADIUS_PX: Final[float] = 12.0
HANDLE_SIZE_PX: Final[int] = 8
ROI_OUTLINE_PX: Final[int] = 2
ROI_FILL_ALPHA: Final[float] = 0.15
