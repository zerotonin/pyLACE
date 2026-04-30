# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tracking.hungarian                                     ║
# ║  « one-shot Hungarian centroid assignment with distance reject » ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Pure-logic frame-to-frame association via the Hungarian algorithm."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def associate(
    track_positions: np.ndarray,
    detection_positions: np.ndarray,
    max_distance_px: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimal centroid pairing between tracks and detections.

    Args:
        track_positions: ``(T, 2)`` float array of last-known track
            centroids in pixel coordinates.
        detection_positions: ``(D, 2)`` float array of detection
            centroids in pixel coordinates.
        max_distance_px: Pairings with cost above this are rejected;
            the corresponding track and detection end up unmatched.

    Returns:
        ``(matches, unmatched_tracks, unmatched_detections)``.
        ``matches`` is a list of ``(track_index, detection_index)``
        pairs whose distance is at most ``max_distance_px``. The two
        index lists cover whatever is left.
    """
    if max_distance_px < 0:
        raise ValueError("max_distance_px must be >= 0.")
    n_tracks = len(track_positions)
    n_detections = len(detection_positions)
    if n_tracks == 0 or n_detections == 0:
        return [], list(range(n_tracks)), list(range(n_detections))

    track_positions = np.asarray(track_positions, dtype=np.float64).reshape(-1, 2)
    detection_positions = np.asarray(
        detection_positions, dtype=np.float64,
    ).reshape(-1, 2)

    diff = track_positions[:, None, :] - detection_positions[None, :, :]
    cost = np.linalg.norm(diff, axis=-1)

    track_idxs, det_idxs = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for ti, di in zip(track_idxs, det_idxs, strict=False):
        if cost[ti, di] <= max_distance_px:
            matches.append((int(ti), int(di)))
            matched_tracks.add(int(ti))
            matched_dets.add(int(di))

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_detections = [
        i for i in range(n_detections) if i not in matched_dets
    ]
    return matches, unmatched_tracks, unmatched_detections


__all__ = ["associate"]
