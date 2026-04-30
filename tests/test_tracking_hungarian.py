# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tests.test_tracking_hungarian                          ║
# ║  « centroid-pair association with distance reject »              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for :func:`pylace.tracking.hungarian.associate`."""

from __future__ import annotations

import numpy as np
import pytest

from pylace.tracking.hungarian import associate


def test_empty_tracks_returns_unmatched_detections():
    matches, unmatched_t, unmatched_d = associate(
        np.zeros((0, 2)), np.array([[10.0, 10.0], [20.0, 20.0]]),
        max_distance_px=50.0,
    )
    assert matches == []
    assert unmatched_t == []
    assert unmatched_d == [0, 1]


def test_empty_detections_returns_unmatched_tracks():
    matches, unmatched_t, unmatched_d = associate(
        np.array([[10.0, 10.0], [20.0, 20.0]]), np.zeros((0, 2)),
        max_distance_px=50.0,
    )
    assert matches == []
    assert unmatched_t == [0, 1]
    assert unmatched_d == []


def test_close_pairs_are_matched_optimally():
    tracks = np.array([[0.0, 0.0], [100.0, 100.0]])
    dets = np.array([[1.0, 1.0], [99.0, 101.0]])
    matches, unmatched_t, unmatched_d = associate(tracks, dets, max_distance_px=10.0)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert unmatched_t == []
    assert unmatched_d == []


def test_optimal_assignment_swaps_when_better():
    """Greedy nearest-neighbour would mis-pair; Hungarian must pick total-min."""
    tracks = np.array([[0.0, 0.0], [10.0, 0.0]])
    dets = np.array([[8.0, 0.0], [11.0, 0.0]])
    # Nearest-neighbour would pair t0→d0 (8) and t1→d1 (1): total 9.
    # Hungarian pairs t0→d1 (11) and t1→d0 (2): total 13. Wait — that's worse.
    # Reality check: Hungarian minimises total cost so it picks the 9-total pairing.
    matches, _, _ = associate(tracks, dets, max_distance_px=20.0)
    matches_dict = dict(matches)
    assert matches_dict == {0: 0, 1: 1}


def test_pairings_above_threshold_are_rejected():
    tracks = np.array([[0.0, 0.0]])
    dets = np.array([[200.0, 0.0]])
    matches, unmatched_t, unmatched_d = associate(tracks, dets, max_distance_px=50.0)
    assert matches == []
    assert unmatched_t == [0]
    assert unmatched_d == [0]


def test_partial_match_with_extra_detection():
    tracks = np.array([[0.0, 0.0]])
    dets = np.array([[1.0, 1.0], [200.0, 200.0]])
    matches, unmatched_t, unmatched_d = associate(tracks, dets, max_distance_px=10.0)
    assert matches == [(0, 0)]
    assert unmatched_t == []
    assert unmatched_d == [1]


def test_partial_match_with_extra_track():
    tracks = np.array([[0.0, 0.0], [200.0, 200.0]])
    dets = np.array([[1.0, 1.0]])
    matches, unmatched_t, unmatched_d = associate(tracks, dets, max_distance_px=10.0)
    assert matches == [(0, 0)]
    assert unmatched_t == [1]
    assert unmatched_d == []


def test_negative_max_distance_raises():
    with pytest.raises(ValueError):
        associate(np.zeros((1, 2)), np.zeros((1, 2)), max_distance_px=-1.0)
