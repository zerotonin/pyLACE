# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tracking.tracks                                        ║
# ║  « stateful Hungarian tracker with track birth / death »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  One Tracker per ROI run. Per-frame: build the cost matrix       ║
# ║  between active tracks and new detections, run the optimal       ║
# ║  pairing via :func:`associate`, accept matches under             ║
# ║  ``max_distance_px``, increment the missed-frame counter for     ║
# ║  unmatched tracks, retire any track unmatched for more than      ║
# ║  ``max_missed_frames`` consecutive frames, and birth a new       ║
# ║  track for each unmatched detection.                             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Stateful frame-to-frame identity tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pylace.detect.frame import Detection
from pylace.tracking.constants import (
    DEFAULT_MAX_DISTANCE_PX,
    DEFAULT_MAX_MISSED_FRAMES,
)
from pylace.tracking.hungarian import associate


@dataclass
class Track:
    """Per-track state held by the :class:`Tracker`."""

    track_id: int
    last_position: tuple[float, float]
    last_frame_idx: int
    age: int = 1
    missed_frames: int = 0


class Tracker:
    """Hungarian centroid tracker, dynamic-N or fixed-N.

    Two modes share the same ``step`` API:

    - **Dynamic-N** (``n_animals=None``, the default). Tracks are born
      for unmatched detections; tracks unmatched for more than
      ``max_missed_frames`` consecutive frames are retired. ``max_distance_px``
      rejects implausible matches.
    - **Fixed-N** (``n_animals=N``). Tracks are born for the first N
      detections that the tracker sees, then never replaced. Unmatched
      tracks stay alive forever; ``max_missed_frames`` is ignored. The
      Hungarian runs unconstrained (the user has guaranteed there are
      exactly N animals, so the tracker trusts the optimal pairing
      regardless of distance). Unmatched detections beyond N are
      dropped — they are spurious by the user's own count.

      Fixed-N mode is what the LACE paper assumes: the user knows N, so
      the tracker never silently births a phantom track when two flies
      merge into one blob.
    """

    def __init__(
        self,
        *,
        max_distance_px: float = DEFAULT_MAX_DISTANCE_PX,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        n_animals: int | None = None,
    ) -> None:
        if max_distance_px < 0:
            raise ValueError("max_distance_px must be >= 0.")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be >= 0.")
        if n_animals is not None and n_animals < 1:
            raise ValueError("n_animals must be >= 1 (or None for dynamic-N).")
        self._max_distance_px = float(max_distance_px)
        self._max_missed_frames = int(max_missed_frames)
        self._n_animals = int(n_animals) if n_animals is not None else None
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 0

    @property
    def active_tracks(self) -> list[Track]:
        """Tracks currently in the live + grace pool, in insertion order."""
        return list(self._tracks.values())

    @property
    def is_fixed_n(self) -> bool:
        return self._n_animals is not None

    def step(
        self, frame_idx: int, detections: list[Detection],
    ) -> list[Detection]:
        """Update tracker state and return the detections actually kept.

        Stamps ``Detection.track_id`` in place. In dynamic-N mode this
        returns the same list it was given. In fixed-N mode it drops
        any detection beyond the Nth that could not be matched to a
        track, since those are spurious under the user's known count.
        """
        track_ids, track_positions = self._snapshot_tracks()
        detection_positions = self._detection_positions(detections)

        match_threshold = (
            float("inf") if self.is_fixed_n else self._max_distance_px
        )
        matches, unmatched_t, unmatched_d = associate(
            track_positions, detection_positions, match_threshold,
        )

        self._apply_matches(matches, track_ids, detections, frame_idx)
        self._increment_missed(unmatched_t, track_ids)

        if self.is_fixed_n:
            return self._step_fixed_n(unmatched_d, detections, frame_idx)
        self._retire_dead_tracks()
        self._birth_new_tracks(unmatched_d, detections, frame_idx)
        return detections

    def _step_fixed_n(
        self,
        unmatched_d: list[int],
        detections: list[Detection],
        frame_idx: int,
    ) -> list[Detection]:
        """Birth tracks until N exist; drop any further unmatched detections."""
        assert self._n_animals is not None
        slots = self._n_animals - len(self._tracks)
        for di in unmatched_d[:slots]:
            self._birth(detections[di], frame_idx)
        # Keep only detections that ended up with a track_id.
        return [d for d in detections if d.track_id >= 0]

    # ── Internal ───────────────────────────────────────────────────────

    def _snapshot_tracks(self) -> tuple[list[int], np.ndarray]:
        track_ids = list(self._tracks.keys())
        if not track_ids:
            return track_ids, np.zeros((0, 2), dtype=np.float64)
        positions = np.array(
            [self._tracks[tid].last_position for tid in track_ids],
            dtype=np.float64,
        )
        return track_ids, positions

    @staticmethod
    def _detection_positions(detections: list[Detection]) -> np.ndarray:
        if not detections:
            return np.zeros((0, 2), dtype=np.float64)
        return np.array([(d.cx, d.cy) for d in detections], dtype=np.float64)

    def _apply_matches(
        self,
        matches: list[tuple[int, int]],
        track_ids: list[int],
        detections: list[Detection],
        frame_idx: int,
    ) -> None:
        for ti, di in matches:
            tid = track_ids[ti]
            track = self._tracks[tid]
            d = detections[di]
            track.last_position = (d.cx, d.cy)
            track.last_frame_idx = frame_idx
            track.age += 1
            track.missed_frames = 0
            d.track_id = tid

    def _increment_missed(
        self, unmatched_t: list[int], track_ids: list[int],
    ) -> None:
        for ti in unmatched_t:
            tid = track_ids[ti]
            self._tracks[tid].missed_frames += 1

    def _retire_dead_tracks(self) -> None:
        dead = [
            tid for tid, t in self._tracks.items()
            if t.missed_frames > self._max_missed_frames
        ]
        for tid in dead:
            del self._tracks[tid]

    def _birth_new_tracks(
        self,
        unmatched_d: list[int],
        detections: list[Detection],
        frame_idx: int,
    ) -> None:
        for di in unmatched_d:
            self._birth(detections[di], frame_idx)

    def _birth(self, detection: Detection, frame_idx: int) -> None:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = Track(
            track_id=tid,
            last_position=(detection.cx, detection.cy),
            last_frame_idx=frame_idx,
        )
        detection.track_id = tid


__all__ = ["Track", "Tracker"]
