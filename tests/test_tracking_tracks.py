# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tests.test_tracking_tracks                             ║
# ║  « stateful Tracker: birth, persistence, occlusion, death »      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for :class:`pylace.tracking.tracks.Tracker`."""

from __future__ import annotations

import numpy as np
import pytest

from pylace.detect.frame import Detection
from pylace.tracking.tracks import Tracker


def _det(cx: float, cy: float) -> Detection:
    return Detection(
        cx=cx, cy=cy, area_px=100.0,
        major_axis_px=10.0, minor_axis_px=5.0,
        orientation_deg=0.0, contour=None,
    )


def test_first_frame_births_one_track_per_detection():
    tracker = Tracker(max_distance_px=50.0, max_missed_frames=5)
    detections = [_det(10, 10), _det(50, 50)]
    tracker.step(0, detections)
    assert sorted(d.track_id for d in detections) == [0, 1]
    assert len(tracker.active_tracks) == 2


def test_second_frame_preserves_ids_under_small_motion():
    tracker = Tracker(max_distance_px=50.0, max_missed_frames=5)
    f0 = [_det(10, 10), _det(50, 50)]
    tracker.step(0, f0)
    id_a, id_b = f0[0].track_id, f0[1].track_id

    f1 = [_det(11, 12), _det(52, 49)]
    tracker.step(1, f1)
    assert f1[0].track_id == id_a
    assert f1[1].track_id == id_b


def test_swapped_detection_order_is_resolved_by_hungarian():
    tracker = Tracker(max_distance_px=50.0, max_missed_frames=5)
    f0 = [_det(10, 10), _det(50, 50)]
    tracker.step(0, f0)
    id_left, id_right = f0[0].track_id, f0[1].track_id

    # Detector returned the right blob first this frame.
    f1 = [_det(51, 51), _det(11, 11)]
    tracker.step(1, f1)
    assert f1[0].track_id == id_right
    assert f1[1].track_id == id_left


def test_brief_occlusion_keeps_id_within_grace_period():
    tracker = Tracker(max_distance_px=50.0, max_missed_frames=3)
    f0 = [_det(10, 10)]
    tracker.step(0, f0)
    tid = f0[0].track_id

    tracker.step(1, [])
    tracker.step(2, [])
    f3 = [_det(11, 11)]
    tracker.step(3, f3)
    assert f3[0].track_id == tid


def test_long_occlusion_retires_track_and_births_new_one():
    tracker = Tracker(max_distance_px=50.0, max_missed_frames=2)
    f0 = [_det(10, 10)]
    tracker.step(0, f0)
    tid_old = f0[0].track_id

    for missing in range(1, 6):
        tracker.step(missing, [])
    f6 = [_det(10, 10)]
    tracker.step(6, f6)
    assert f6[0].track_id != tid_old


def test_far_jump_does_not_steal_an_existing_track():
    tracker = Tracker(max_distance_px=20.0, max_missed_frames=5)
    f0 = [_det(10, 10)]
    tracker.step(0, f0)
    tid_old = f0[0].track_id

    f1 = [_det(200, 200)]
    tracker.step(1, f1)
    assert f1[0].track_id != tid_old
    # Both old (now missing) and new track exist.
    assert len(tracker.active_tracks) == 2


def test_unmatched_detection_births_new_track():
    tracker = Tracker(max_distance_px=50.0, max_missed_frames=5)
    tracker.step(0, [_det(10, 10)])
    f1 = [_det(10, 10), _det(200, 200)]
    tracker.step(1, f1)
    assert {d.track_id for d in f1} == {0, 1}


def test_negative_init_args_raise():
    with pytest.raises(ValueError):
        Tracker(max_distance_px=-1.0)
    with pytest.raises(ValueError):
        Tracker(max_missed_frames=-1)
    with pytest.raises(ValueError):
        Tracker(n_animals=0)
    with pytest.raises(ValueError):
        Tracker(n_animals=-3)


# ─────────────────────────────────────────────────────────────────
#  Fixed-N mode (LACE-paper assumption)
# ─────────────────────────────────────────────────────────────────


def test_fixed_n_births_up_to_n_then_stops():
    """In fixed-N mode tracks are born for the first N detections only."""
    tracker = Tracker(n_animals=2)
    f0 = [_det(10, 10), _det(20, 20)]
    tracker.step(0, f0)
    assert len(tracker.active_tracks) == 2

    # A third blob in the next frame is spurious — must be dropped.
    f1 = [_det(11, 11), _det(21, 21), _det(80, 80)]
    kept = tracker.step(1, f1)
    assert len(tracker.active_tracks) == 2
    assert len(kept) == 2
    assert all(d.track_id >= 0 for d in kept)


def test_fixed_n_keeps_track_alive_through_long_occlusion():
    """A track unmatched for many frames is preserved, not retired."""
    tracker = Tracker(n_animals=2, max_missed_frames=3)
    tracker.step(0, [_det(10, 10), _det(50, 50)])
    ids_before = sorted(t.track_id for t in tracker.active_tracks)

    # Long stretch with only one detection (the other fly is in a chain).
    for fr in range(1, 20):
        tracker.step(fr, [_det(10 + fr * 0.1, 10)])

    assert len(tracker.active_tracks) == 2
    ids_after = sorted(t.track_id for t in tracker.active_tracks)
    assert ids_after == ids_before


def test_fixed_n_recovers_id_after_chain_breaks():
    """The lost fly reattaches to its original track when it reappears."""
    tracker = Tracker(n_animals=2)
    tracker.step(0, [_det(10, 10), _det(50, 50)])
    id_left, id_right = tracker.active_tracks[0].track_id, tracker.active_tracks[1].track_id

    # 30 frames of chain — only one detection.
    for fr in range(1, 31):
        tracker.step(fr, [_det(30, 30)])

    # Chain breaks, two detections reappear.
    f31 = [_det(12, 11), _det(52, 49)]
    kept = tracker.step(31, f31)
    assert len(kept) == 2
    matched_ids = {d.track_id for d in kept}
    assert matched_ids == {id_left, id_right}


def test_fixed_n_ignores_max_distance_for_recovery():
    """A long-jump match is accepted because there are exactly N animals."""
    tracker = Tracker(n_animals=1, max_distance_px=20.0)
    tracker.step(0, [_det(10, 10)])
    tid = tracker.active_tracks[0].track_id

    # The fly teleports 200 px — far above max_distance_px. Fixed-N must
    # still accept the match because there is only one animal.
    kept = tracker.step(1, [_det(210, 10)])
    assert kept[0].track_id == tid


def test_fixed_n_birth_only_when_detections_appear():
    """If the first frames have no detections, no tracks are born."""
    tracker = Tracker(n_animals=2)
    for fr in range(0, 5):
        tracker.step(fr, [])
    assert len(tracker.active_tracks) == 0
    tracker.step(5, [_det(10, 10)])
    assert len(tracker.active_tracks) == 1
    tracker.step(6, [_det(11, 11), _det(50, 50)])
    assert len(tracker.active_tracks) == 2


def test_fixed_n_does_not_mark_unassigned_detections_with_track_id():
    """Detections beyond the Nth slot keep ``track_id == -1`` and are dropped."""
    tracker = Tracker(n_animals=1)
    tracker.step(0, [_det(10, 10)])

    # Two detections, but only one slot — the spurious one is dropped.
    extras = [_det(11, 11), _det(80, 80)]
    kept = tracker.step(1, extras)
    assert len(kept) == 1
    assert kept[0].cx == pytest.approx(11.0)
    # The dropped detection still has the sentinel.
    assert extras[1].track_id == -1
