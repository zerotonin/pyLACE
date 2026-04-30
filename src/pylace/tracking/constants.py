# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tracking.constants                                     ║
# ║  « shared defaults for the Hungarian tracker »                   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Constants for the tracking sub-package."""

from __future__ import annotations

from typing import Final

# Maximum centroid jump (px) we are willing to accept as the same fly
# between consecutive frames. Above this, the candidate match is
# rejected and the track is treated as missing for that frame.
DEFAULT_MAX_DISTANCE_PX: Final[float] = 50.0

# Number of consecutive missed frames before a track is retired. Until
# then the track is held in a "missing" state so a fly that is briefly
# occluded recovers its original ID when it reappears.
DEFAULT_MAX_MISSED_FRAMES: Final[int] = 5

# Multi-feature cost weights. The total cost between a track and a
# detection is ``position_cost + area_weight * |Δarea| + perimeter_weight
# * |Δperimeter|``. Defaults are zero so the legacy distance-only
# behaviour is preserved unless a project explicitly opts in.
DEFAULT_AREA_COST_WEIGHT: Final[float] = 0.0
DEFAULT_PERIMETER_COST_WEIGHT: Final[float] = 0.0
