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

# ─────────────────────────────────────────────────────────────────
#  Kalman motion model  « Phase 4 — predicted-position cost »
# ─────────────────────────────────────────────────────────────────
# Per-frame position-drift std (px). Small process noise that lets
# the filter accommodate small irregular position errors the
# constant-velocity model does not predict. Larger values make the
# filter more responsive but less smooth.
DEFAULT_KALMAN_Q_POS: Final[float] = 0.5

# Per-frame velocity-jitter std (px/frame). Roughly the rms
# acceleration the filter expects between frames. Drosophila
# walking saccades are well below 1 px/frame² at typical fps; this
# default is conservative.
DEFAULT_KALMAN_Q_VEL: Final[float] = 0.5

# Per-axis measurement noise std (px). The detector's centroid is
# accurate to about 1 px on a 1080p arena view; raise this if your
# detections look noisier (e.g. a low-contrast recording).
DEFAULT_KALMAN_R_POS: Final[float] = 1.0

# Initial velocity std at track birth (px/frame). Generous enough
# to cover the fastest plausible per-frame motion of a fly. The
# filter converges within a few observations regardless of the
# exact value.
DEFAULT_KALMAN_INITIAL_V_STD: Final[float] = 10.0
