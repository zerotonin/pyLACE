# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi                                                    ║
# ║  « multi-shape regions of interest with boolean composition »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Per-video ROI sub-package. Each ROI carries a shape (Circle,    ║
# ║  Rectangle, Polygon — reused from the annotator) and a per-ROI   ║
# ║  add/subtract operation. ROIs combine into a single boolean      ║
# ║  mask that the detector intersects with the arena mask.          ║
# ║                                                                  ║
# ║  Split mode (each ROI as its own sub-video) is reserved for v2,  ║
# ║  which lands on top of identity tracking.                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""ROI builder + detector integration."""

from pylace.roi.geometry import ROI, ROIMode, ROIOperation, ROISet
from pylace.roi.mask import build_combined_mask, build_split_masks
from pylace.roi.sidecar import (
    SCHEMA_VERSION,
    ROISidecar,
    ROISidecarSchemaError,
    default_rois_path,
    read_rois,
    write_rois,
)

__all__ = [
    "ROI",
    "ROIMode",
    "ROIOperation",
    "ROISet",
    "ROISidecar",
    "ROISidecarSchemaError",
    "SCHEMA_VERSION",
    "build_combined_mask",
    "build_split_masks",
    "default_rois_path",
    "read_rois",
    "write_rois",
]
