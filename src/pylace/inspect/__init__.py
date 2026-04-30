# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — inspect                                                ║
# ║  « visual inspection of tracked detections »                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Loads a video plus its companion .pylace_detections.csv and     ║
# ║  renders two complementary views: a live frame scrubber with a   ║
# ║  fading n-second trail behind each animal, and a static-          ║
# ║  background overview that draws every track's full trajectory.   ║
# ║  Per-track colours come from the Wong (2011) colourblind-safe    ║
# ║  palette; if a recording has more than eight animals the palette ║
# ║  extends with golden-angle HSV samples.                          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Visual inspection of tracked-detections CSVs."""

from pylace.inspect.palette import palette_bgr
from pylace.inspect.traces import (
    TrackTrajectory,
    read_traces,
    render_current_markers,
    render_full_trajectories,
    render_trail,
)

__all__ = [
    "TrackTrajectory",
    "palette_bgr",
    "read_traces",
    "render_current_markers",
    "render_full_trajectories",
    "render_trail",
]
