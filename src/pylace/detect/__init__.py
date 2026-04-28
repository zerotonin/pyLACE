"""Classical-CV per-frame detection: bg-sub → threshold → contour → ellipse."""

from pylace.detect.arena_mask import arena_mask
from pylace.detect.background import build_max_projection_background
from pylace.detect.frame import Detection, detect_blobs

__all__ = [
    "Detection",
    "arena_mask",
    "build_max_projection_background",
    "detect_blobs",
]
