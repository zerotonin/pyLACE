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

from pylace.posthoc.analytics import compute_distance_to_wall
from pylace.posthoc.audit import audit_track_identities
from pylace.posthoc.clean import clean_trajectory
from pylace.posthoc.heading import compute_yaw, resolve_headings
from pylace.posthoc.io import (
    CLEANED_EXTRA_COLUMNS,
    DETECTION_COLUMNS,
    read_detections,
    write_trajectory,
)
from pylace.posthoc.metrics import (
    occupancy_heatmap,
    speed_summary,
    summarise_track,
    summarise_tracks,
    thigmotaxis_fraction,
    walk_stop_bouts,
    yaw_rate_summary,
)
from pylace.posthoc.multifly import (
    nearest_neighbour_distance,
    pairwise_distance_matrix,
    polarisation,
)

__all__ = [
    "CLEANED_EXTRA_COLUMNS",
    "DETECTION_COLUMNS",
    "audit_track_identities",
    "clean_trajectory",
    "compute_distance_to_wall",
    "compute_yaw",
    "nearest_neighbour_distance",
    "occupancy_heatmap",
    "pairwise_distance_matrix",
    "polarisation",
    "read_detections",
    "resolve_headings",
    "speed_summary",
    "summarise_track",
    "summarise_tracks",
    "thigmotaxis_fraction",
    "walk_stop_bouts",
    "write_trajectory",
    "yaw_rate_summary",
]
