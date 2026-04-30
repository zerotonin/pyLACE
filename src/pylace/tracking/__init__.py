# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tracking                                               ║
# ║  « Hungarian frame-to-frame identity tracking »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Stateful Tracker that turns per-frame detections into stable    ║
# ║  per-fly track_ids. Pure-logic helper :func:`associate` runs the ║
# ║  optimal centroid pairing via scipy.optimize.linear_sum_assign-  ║
# ║  ment with a max-distance reject; the Tracker manages track      ║
# ║  birth, persistence across short occlusions, and death after     ║
# ║  max_missed_frames consecutive misses.                           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Identity tracking via Hungarian centroid assignment."""

from pylace.tracking.constants import (
    DEFAULT_MAX_DISTANCE_PX,
    DEFAULT_MAX_MISSED_FRAMES,
)
from pylace.tracking.hungarian import associate
from pylace.tracking.tracks import Track, Tracker

__all__ = [
    "DEFAULT_MAX_DISTANCE_PX",
    "DEFAULT_MAX_MISSED_FRAMES",
    "Track",
    "Tracker",
    "associate",
]
