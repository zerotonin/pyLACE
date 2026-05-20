"""Tests for ``pylace.review.candidates``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylace.review.candidates import (
    CANDIDATES_SUFFIX,
    Candidate,
    SOURCE_CONTACT,
    SOURCE_JUMP,
    default_candidates_path,
    detect_candidates,
    read_candidates,
    write_candidates,
)


def _two_track_df(
    cx0: np.ndarray, cy0: np.ndarray,
    cx1: np.ndarray, cy1: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i in range(cx0.size):
        rows.append(dict(frame_idx=i, track_id=0,
                         cx_smooth_px=cx0[i], cy_smooth_px=cy0[i]))
        rows.append(dict(frame_idx=i, track_id=1,
                         cx_smooth_px=cx1[i], cy_smooth_px=cy1[i]))
    return pd.DataFrame(rows)


def _three_track_df(arrs: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    n = arrs[0][0].size
    rows = []
    for i in range(n):
        for tid, (cx, cy) in enumerate(arrs):
            rows.append(dict(frame_idx=i, track_id=tid,
                             cx_smooth_px=cx[i], cy_smooth_px=cy[i]))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
#  Detector
# ─────────────────────────────────────────────────────────────────


def test_no_candidates_for_well_separated_static_tracks():
    n = 50
    df = _two_track_df(
        np.zeros(n), np.zeros(n),
        np.full(n, 500.0), np.full(n, 500.0),
    )
    c = detect_candidates(df, fps=10.0, pix_per_mm=1.0,
                          contact_threshold_mm=5.0,
                          jump_threshold_mm_per_frame=10.0)
    assert c == []


def test_contact_event_is_detected():
    n = 50
    cx0 = np.arange(n, dtype=float); cy0 = np.zeros(n)
    cx1 = np.arange(n, dtype=float); cy1 = np.full(n, 50.0)
    cy1[25] = 1.0  # brief proximity at frame 25
    df = _two_track_df(cx0, cy0, cx1, cy1)
    c = detect_candidates(df, fps=10.0, pix_per_mm=1.0,
                          contact_threshold_mm=5.0,
                          jump_threshold_mm_per_frame=100.0)
    assert len(c) == 1
    assert c[0].frame_start == 25
    assert c[0].frame_end == 25
    assert c[0].animal_a == 0 and c[0].animal_b == 1
    assert SOURCE_CONTACT in c[0].source
    assert SOURCE_JUMP not in c[0].source


def test_jump_event_is_detected():
    n = 30
    cx0 = np.arange(n, dtype=float); cy0 = np.zeros(n)
    cx1 = np.arange(n, dtype=float); cy1 = np.full(n, 100.0)
    cx0[15] = 100.0  # track 0 teleports by ~100 px at frame 15
    df = _two_track_df(cx0, cy0, cx1, cy1)
    c = detect_candidates(df, fps=10.0, pix_per_mm=1.0,
                          contact_threshold_mm=5.0,
                          jump_threshold_mm_per_frame=10.0)
    assert any(SOURCE_JUMP in cand.source for cand in c)


def test_contact_and_jump_combined_source():
    n = 40
    cx0 = np.full(n, 50.0); cy0 = np.full(n, 50.0)
    cx1 = np.full(n, 50.0); cy1 = np.full(n, 150.0)
    # At frame 20, both tracks teleport to the same point — gives a
    # zero-distance contact AND a large per-track jump.
    cx0[20] = 200.0; cy0[20] = 100.0
    cx1[20] = 200.0; cy1[20] = 100.0
    df = _two_track_df(cx0, cy0, cx1, cy1)
    c = detect_candidates(df, fps=10.0, pix_per_mm=1.0,
                          contact_threshold_mm=5.0,
                          jump_threshold_mm_per_frame=10.0)
    assert len(c) >= 1
    parts = c[0].source.split(",")
    assert SOURCE_CONTACT in parts
    assert SOURCE_JUMP in parts


def test_adjacent_event_frames_coalesce_into_one_block():
    n = 40
    cx0 = np.arange(n, dtype=float); cy0 = np.zeros(n)
    cx1 = np.arange(n, dtype=float); cy1 = np.full(n, 50.0)
    # 5-frame contact 25..29.
    cy1[25:30] = 1.0
    df = _two_track_df(cx0, cy0, cx1, cy1)
    c = detect_candidates(df, fps=10.0, pix_per_mm=1.0,
                          contact_threshold_mm=5.0,
                          jump_threshold_mm_per_frame=100.0,
                          coalesce_window_frames=2)
    assert len(c) == 1
    assert c[0].frame_start == 25
    assert c[0].frame_end == 29
    assert c[0].n_frames == 5


def test_closest_pair_is_reported_for_three_tracks():
    n = 30
    t0 = (np.full(n, 50.0), np.full(n, 50.0))
    t1 = (np.full(n, 50.0), np.full(n, 51.0))   # always touching track 0
    t2 = (np.full(n, 500.0), np.full(n, 500.0)) # far away
    df = _three_track_df([t0, t1, t2])
    c = detect_candidates(df, fps=10.0, pix_per_mm=1.0,
                          contact_threshold_mm=5.0,
                          jump_threshold_mm_per_frame=100.0)
    assert len(c) >= 1
    # The pair touching is (0, 1), not (0, 2) or (1, 2).
    assert {c[0].animal_a, c[0].animal_b} == {0, 1}


def test_single_track_returns_no_candidates():
    n = 30
    df = pd.DataFrame({
        "frame_idx": np.arange(n),
        "track_id": [0] * n,
        "cx_smooth_px": np.arange(n, dtype=float),
        "cy_smooth_px": np.zeros(n),
    })
    assert detect_candidates(df, fps=10.0, pix_per_mm=1.0) == []


def test_rejects_invalid_params():
    df = _two_track_df(
        np.zeros(10), np.zeros(10),
        np.full(10, 100.0), np.full(10, 100.0),
    )
    with pytest.raises(ValueError, match="pix_per_mm"):
        detect_candidates(df, fps=10.0, pix_per_mm=0.0)
    with pytest.raises(ValueError, match="fps"):
        detect_candidates(df, fps=0.0, pix_per_mm=1.0)
    with pytest.raises(ValueError, match="jump"):
        detect_candidates(df, fps=10.0, pix_per_mm=1.0,
                          jump_threshold_mm_per_frame=0.0)


def test_event_id_matches_audit_make_event_id():
    """Round-trip: the candidate's event_id reproduces make_event_id(frame_start, a, b)."""
    from pylace.review.verdicts import make_event_id
    n = 40
    cx0 = np.arange(n, dtype=float); cy0 = np.zeros(n)
    cx1 = np.arange(n, dtype=float); cy1 = np.full(n, 50.0)
    cy1[20] = 1.0
    df = _two_track_df(cx0, cy0, cx1, cy1)
    c = detect_candidates(df, fps=10.0, pix_per_mm=1.0,
                          contact_threshold_mm=5.0,
                          jump_threshold_mm_per_frame=100.0)
    assert c[0].event_id == make_event_id(20, 0, 1)


# ─────────────────────────────────────────────────────────────────
#  Round-trip
# ─────────────────────────────────────────────────────────────────


def test_write_and_read_round_trip(tmp_path: Path):
    candidates = [
        Candidate(
            event_id="100-0-1", frame_start=100, frame_end=105,
            animal_a=0, animal_b=1,
            source="contact,jump",
            min_distance_px=4.2, max_jump_px=42.0, n_frames=6,
        ),
        Candidate(
            event_id="200-0-1", frame_start=200, frame_end=200,
            animal_a=0, animal_b=1,
            source="contact",
            min_distance_px=1.1, max_jump_px=0.5, n_frames=1,
        ),
    ]
    p = tmp_path / "vid.pylace_candidates.csv"
    write_candidates(candidates, p)
    loaded = read_candidates(p)
    assert len(loaded) == 2
    # File-order is sorted by frame_start.
    assert loaded[0].frame_start == 100
    assert loaded[1].frame_start == 200
    assert loaded[0].source == "contact,jump"
    assert loaded[0].min_distance_px == pytest.approx(4.2)


def test_read_missing_file_returns_empty_list(tmp_path: Path):
    assert read_candidates(tmp_path / "missing.csv") == []


def test_default_candidates_path_strips_audited_suffix(tmp_path: Path):
    audited = tmp_path / "vid.mp4.pylace_audited.csv"
    out = default_candidates_path(audited)
    assert out.name == "vid.mp4" + CANDIDATES_SUFFIX
