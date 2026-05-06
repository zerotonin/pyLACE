# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.constants                                      ║
# ║  « tunable defaults for the trajectory cleaner »                 ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Defaults for the post-hoc trajectory cleaner."""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────
#  Gap fill
# ─────────────────────────────────────────────────────────────────
# Largest detection gap (in frames) we will linearly interpolate.
# Defaults scale with fps to roughly 1/12 second; gaps beyond this are
# left as missing rows because a fly out of view for that long is
# better flagged than fabricated.
DEFAULT_MAX_GAP_FRACTION_OF_FPS: float = 1.0 / 12.0

# ─────────────────────────────────────────────────────────────────
#  Outlier rejection
# ─────────────────────────────────────────────────────────────────
# Velocity ceiling above which a single-frame jump is treated as a
# detection error rather than real fly motion. Walking Drosophila top
# out around 30 mm/s; brief flight bursts reach ~150 mm/s. 200 mm/s
# is a permissive default that catches teleports without rejecting
# real saccades.
DEFAULT_MAX_SPEED_MM_S: float = 200.0

# ─────────────────────────────────────────────────────────────────
#  Savitzky-Golay smoothing  « Geurten 2022 LACE paper convention »
# ─────────────────────────────────────────────────────────────────
DEFAULT_SG_WINDOW: int = 11
DEFAULT_SG_POLYORDER: int = 3

# ─────────────────────────────────────────────────────────────────
#  Body-axis (heading) disambiguator
# ─────────────────────────────────────────────────────────────────
# Speed below which the velocity vector is considered too small to
# anchor head/tail. The DP step still chooses a candidate, but its
# unary cost is set to zero so transition costs dominate (i.e. the
# previous frame's heading is preserved).
DEFAULT_HEADING_SPEED_FLOOR_MM_S: float = 1.0

# Cost (in degrees-equivalent) of flipping the heading by ~180°
# between adjacent frames. Multiplied into the Viterbi transition
# cost; higher values bias the DP toward smoother heading sequences.
DEFAULT_HEADING_FLIP_PENALTY: float = 90.0

# Smoothing window for the cosine / sine of the resolved heading
# before computing yaw rate. Smaller than the position SG because we
# want to preserve fast turns without de-jittering them away.
DEFAULT_YAW_SG_WINDOW: int = 7
DEFAULT_YAW_SG_POLYORDER: int = 2

# ─────────────────────────────────────────────────────────────────
#  Kinematic readouts (Phase 2)  « Branson / Robie / Anderson »
# ─────────────────────────────────────────────────────────────────
# Schmitt-trigger walk / stop classifier. The hysteresis between on
# and off thresholds prevents bout fragmentation from speed jitter at
# the boundary; minimum duration suppresses single-frame spurious
# walks. Robie et al., J Exp Biol 2010.
DEFAULT_BOUT_ON_MM_S: float = 2.5
DEFAULT_BOUT_OFF_MM_S: float = 1.0
DEFAULT_BOUT_MIN_DURATION_S: float = 0.1

# 2D occupancy histogram resolution. 64 × 64 over the whole frame is
# the convention for visualising arena-scale heatmaps.
DEFAULT_OCCUPANCY_BINS: int = 64

# Outer band fraction defining "wall zone" for thigmotaxis
# computation. Soibam et al., PLoS ONE 2012, used 25%; 20% is the
# slightly stricter Branson convention.
DEFAULT_THIGMOTAXIS_OUTER_FRAC: float = 0.20
