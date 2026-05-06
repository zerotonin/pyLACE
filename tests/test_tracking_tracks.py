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


def _det(
    cx: float, cy: float,
    *, area_px: float = 100.0, perimeter_px: float = 40.0,
    solidity: float = 0.95,
) -> Detection:
    return Detection(
        cx=cx, cy=cy, area_px=area_px, perimeter_px=perimeter_px,
        solidity=solidity,
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


def test_area_cost_resolves_two_close_detections_with_different_sizes():
    """Position-only would mis-pair; area weight tips the assignment correctly."""
    # Track A is at (30, 30) with area 1000.
    # Track B is at (32, 30) with area 5000.
    # New detections: one at (31, 30) area 1100, one at (31, 30) area 4800.
    # Position-only Hungarian sees the same distance for both pairings;
    # adding area_cost_weight breaks the tie toward the right partner.
    tracker = Tracker(n_animals=2, area_cost_weight=0.05)
    tracker.step(0, [
        _det(30, 30, area_px=1000),
        _det(32, 30, area_px=5000),
    ])
    track_a = tracker.active_tracks[0]
    track_b = tracker.active_tracks[1]

    detections = [
        _det(31, 30, area_px=1100),
        _det(31, 30, area_px=4800),
    ]
    tracker.step(1, detections)
    # The 1100-area detection should have stayed with track A (started 1000),
    # and the 4800-area detection with track B (started 5000).
    by_area = {round(d.area_px): d.track_id for d in detections}
    assert by_area[1100] == track_a.track_id
    assert by_area[4800] == track_b.track_id


def test_perimeter_cost_resolves_when_areas_match_but_contours_differ():
    tracker = Tracker(n_animals=2, perimeter_cost_weight=0.5)
    tracker.step(0, [
        _det(30, 30, perimeter_px=80),
        _det(32, 30, perimeter_px=200),
    ])
    track_short = tracker.active_tracks[0]
    track_long = tracker.active_tracks[1]

    detections = [
        _det(31, 30, perimeter_px=85),
        _det(31, 30, perimeter_px=195),
    ]
    tracker.step(1, detections)
    by_per = {round(d.perimeter_px): d.track_id for d in detections}
    assert by_per[85] == track_short.track_id
    assert by_per[195] == track_long.track_id


def test_negative_cost_weights_raise():
    with pytest.raises(ValueError):
        Tracker(area_cost_weight=-0.1)
    with pytest.raises(ValueError):
        Tracker(perimeter_cost_weight=-0.1)


# ─────────────────────────────────────────────────────────────────
#  Kalman motion model (Phase 4)
# ─────────────────────────────────────────────────────────────────


def test_kalman_state_acquires_velocity_after_a_few_observations():
    """A track that consistently moves should pick up a non-zero velocity."""
    tracker = Tracker(max_distance_px=50.0)
    for fr in range(6):
        # 4 px per frame in +x.
        tracker.step(fr, [_det(10 + 4 * fr, 10)])
    track = tracker.active_tracks[0]
    vx, vy = track.velocity
    assert 2.0 < vx < 6.0
    assert abs(vy) < 1.0


def test_kalman_predicts_through_a_short_gap_at_a_constant_velocity():
    """A fly moving steadily should be re-acquired after a missed frame."""
    # Velocity of 4 px / frame; max_distance_px=10 (tight enough that a
    # raw last-position cost would reject the post-gap detection).
    tracker = Tracker(max_distance_px=10.0, max_missed_frames=3)
    for fr in range(6):
        tracker.step(fr, [_det(10 + 4 * fr, 10)])
    tid = tracker.active_tracks[0].track_id
    # Frame 6 missed. Frame 7 has a detection 8 px from the predicted
    # position (raw distance from frame-5's last_position = 12 px,
    # which would be rejected by max_distance_px=10).
    tracker.step(6, [])
    f7 = [_det(10 + 4 * 7, 10)]  # 38, 10
    kept = tracker.step(7, f7)
    assert kept[0].track_id == tid


def test_kalman_velocity_vanishes_for_a_stationary_track():
    tracker = Tracker(max_distance_px=10.0)
    for fr in range(20):
        tracker.step(fr, [_det(20, 20)])
    track = tracker.active_tracks[0]
    vx, vy = track.velocity
    assert abs(vx) < 0.5
    assert abs(vy) < 0.5


def test_kalman_does_not_break_fixed_n_chain_recovery():
    """The Phase-4 upgrade must not regress the existing chain-recovery test."""
    tracker = Tracker(n_animals=2)
    tracker.step(0, [_det(10, 10), _det(50, 50)])
    id_left = tracker.active_tracks[0].track_id
    id_right = tracker.active_tracks[1].track_id
    for fr in range(1, 31):
        tracker.step(fr, [_det(30, 30)])
    f31 = [_det(12, 11), _det(52, 49)]
    kept = tracker.step(31, f31)
    matched = {d.track_id for d in kept}
    assert matched == {id_left, id_right}


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
