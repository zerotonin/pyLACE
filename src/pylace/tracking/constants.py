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
