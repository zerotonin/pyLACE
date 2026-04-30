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
