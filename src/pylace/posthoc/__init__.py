# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc                                                ║
# ║  « trajectory cleaner, kinematic readouts, multi-fly metrics »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Consume the per-detection CSV produced by pylace-detect and     ║
# ║  emit a cleaned per-frame trajectory CSV with smoothed positions ║
# ║  velocities, resolved heading and yaw rate. Downstream metrics   ║
# ║  (speed, occupancy, thigmotaxis, walk/stop bouts, nearest-       ║
# ║  neighbour distance) read this cleaned file.                     ║
# ║                                                                  ║
# ║  Pipeline order is fixed: gap-fill → outlier rejection →         ║
# ║  Savitzky-Golay smoothing on (x, y) → body-axis disambiguation → ║
# ║  yaw + yaw-rate. Each stage is individually toggleable from the  ║
# ║  CLI and from clean_trajectory().                                ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Post-hoc trajectory cleaning and analysis."""

from pylace.posthoc.clean import clean_trajectory
from pylace.posthoc.heading import compute_yaw, resolve_headings
from pylace.posthoc.io import (
    CLEANED_EXTRA_COLUMNS,
    DETECTION_COLUMNS,
    read_detections,
    write_trajectory,
)

__all__ = [
    "CLEANED_EXTRA_COLUMNS",
    "DETECTION_COLUMNS",
    "clean_trajectory",
    "compute_yaw",
    "read_detections",
    "resolve_headings",
    "write_trajectory",
]
