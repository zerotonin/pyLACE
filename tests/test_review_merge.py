"""Tests for ``pylace.review.merge``."""

from __future__ import annotations

import math

import pandas as pd

from pylace.review.candidates import Candidate
from pylace.review.merge import (
    SOURCE_AUDIT,
    ReviewEvent,
    merge_review_events,
)
from pylace.review.verdicts import Verdict, VerdictRecord, make_event_id, now_iso


def _make_candidate(
    frame_start: int = 100, animal_a: int = 0, animal_b: int = 1,
    source: str = "contact",
) -> Candidate:
    return Candidate(
        event_id=make_event_id(frame_start, animal_a, animal_b),
        frame_start=frame_start,
        frame_end=frame_start + 2,
        animal_a=animal_a, animal_b=animal_b,
        source=source,
        min_distance_px=3.0, max_jump_px=4.0, n_frames=3,
    )


def _make_audit_row(
    event_id: str, frame_start: int = 100, frame_end: int = 102,
    verdict: str = "", cost_before: float = 10.0, cost_after: float = 4.0,
    permutation=None,
) -> dict:
    return {
        "frame_idx": frame_end, "event_id": event_id,
        "frame_start": frame_start, "frame_end": frame_end,
        "verdict": verdict,
        "permutation": list(permutation) if permutation else [],
        "cost_before": cost_before, "cost_after": cost_after,
        "improvement_ratio": 1.0 - cost_after / cost_before if cost_before > 0 else 0.0,
        "cost_kalman_before": cost_before, "cost_kalman_after": cost_after,
        "cost_appearance_before": 0.0, "cost_appearance_after": 0.0,
        "cost_axis_before": 0.0, "cost_axis_after": 0.0,
        "cost_area_before": 0.0, "cost_area_after": 0.0,
    }


def _make_verdict(frame_start: int = 100, animal_a: int = 0, animal_b: int = 1,
                  verdict: Verdict = Verdict.MOUNT) -> VerdictRecord:
    eid = make_event_id(frame_start, animal_a, animal_b)
    return VerdictRecord(
        event_id=eid,
        frame_start=frame_start, frame_end=frame_start + 2,
        animal_a=animal_a, animal_b=animal_b,
        verdict=verdict,
        source="audit,contact", reviewer="bart",
        timestamp_iso=now_iso(), note="",
    )


# ─────────────────────────────────────────────────────────────────
#  Single-source paths
# ─────────────────────────────────────────────────────────────────


def test_candidates_only_passes_through():
    cands = [_make_candidate(frame_start=10), _make_candidate(frame_start=100)]
    events = merge_review_events(candidates=cands)
    assert len(events) == 2
    assert events[0].frame_start == 10
    assert events[1].frame_start == 100
    assert events[0].status == "pending"
    assert events[0].audit_committed is False


def test_audit_only_creates_event_when_no_candidate_matches():
    eid = make_event_id(50, 0, 1)
    log = pd.DataFrame([_make_audit_row(eid, frame_start=50, frame_end=50,
                                        permutation=(1, 0))])
    events = merge_review_events(audit_log=log)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_id == eid
    assert SOURCE_AUDIT in ev.sources
    assert ev.audit_committed is True
    assert ev.audit_permutation == (1, 0)
    assert ev.audit_cost_before == 10.0
    assert ev.audit_cost_after == 4.0
    assert ev.status == "auto_swap"


def test_verdict_only_creates_orphan_event():
    """If audit + candidates are gone but a verdict exists, it still surfaces."""
    v = _make_verdict(verdict=Verdict.REJECT_SWAP)
    events = merge_review_events(verdicts={v.event_id: v})
    assert len(events) == 1
    assert events[0].verdict is Verdict.REJECT_SWAP
    assert events[0].status == "reject_swap"


# ─────────────────────────────────────────────────────────────────
#  Joining
# ─────────────────────────────────────────────────────────────────


def test_candidate_and_audit_merge_on_event_id():
    c = _make_candidate(frame_start=100, source="contact")
    log = pd.DataFrame([_make_audit_row(c.event_id, frame_start=100, frame_end=102,
                                        permutation=(1, 0))])
    events = merge_review_events(candidates=[c], audit_log=log)
    assert len(events) == 1
    ev = events[0]
    assert "contact" in ev.sources
    assert SOURCE_AUDIT in ev.sources
    assert ev.audit_committed is True
    assert ev.min_distance_px == 3.0  # candidate field preserved


def test_verdict_attaches_to_existing_event():
    c = _make_candidate(frame_start=100)
    v = _make_verdict(frame_start=100, verdict=Verdict.MOUNT)
    events = merge_review_events(candidates=[c], verdicts={v.event_id: v})
    assert len(events) == 1
    assert events[0].verdict is Verdict.MOUNT
    assert events[0].status == "mount"


def test_mount_audit_verdict_marks_event_not_committed():
    """An audit row with verdict='mount' didn't commit a swap; status is 'mount'."""
    eid = make_event_id(75, 0, 1)
    log = pd.DataFrame([_make_audit_row(eid, frame_start=75, frame_end=80,
                                        verdict="mount", cost_before=float("nan"),
                                        cost_after=float("nan"))])
    events = merge_review_events(audit_log=log)
    assert len(events) == 1
    assert events[0].audit_committed is False
    assert events[0].audit_verdict == "mount"


def test_three_sources_full_merge():
    c = _make_candidate(frame_start=100, source="contact,jump")
    log = pd.DataFrame([_make_audit_row(c.event_id, permutation=(1, 0))])
    v = _make_verdict(frame_start=100, verdict=Verdict.ACCEPT_SWAP)
    events = merge_review_events(
        candidates=[c], audit_log=log, verdicts={v.event_id: v},
    )
    assert len(events) == 1
    ev = events[0]
    assert set(ev.sources) >= {"contact", "jump", SOURCE_AUDIT}
    assert ev.audit_committed is True
    assert ev.verdict is Verdict.ACCEPT_SWAP
    # verdict trumps audit in the status badge
    assert ev.status == "accept_swap"


# ─────────────────────────────────────────────────────────────────
#  Ordering + edge cases
# ─────────────────────────────────────────────────────────────────


def test_events_sorted_by_frame_start():
    cands = [
        _make_candidate(frame_start=200),
        _make_candidate(frame_start=50),
        _make_candidate(frame_start=125),
    ]
    events = merge_review_events(candidates=cands)
    assert [e.frame_start for e in events] == [50, 125, 200]


def test_empty_inputs_return_empty_list():
    assert merge_review_events() == []
    assert merge_review_events(candidates=[]) == []
    assert merge_review_events(audit_log=pd.DataFrame()) == []
    assert merge_review_events(verdicts={}) == []


def test_audit_row_without_event_id_is_skipped():
    """Old audit logs without the event_id column should not blow up."""
    log = pd.DataFrame([{
        "frame_idx": 100, "event_id": "", "frame_start": 100, "frame_end": 100,
        "verdict": "", "permutation": [1, 0],
        "cost_before": 10.0, "cost_after": 4.0, "improvement_ratio": 0.6,
        "cost_kalman_before": 10.0, "cost_kalman_after": 4.0,
        "cost_appearance_before": 0.0, "cost_appearance_after": 0.0,
        "cost_axis_before": 0.0, "cost_axis_after": 0.0,
        "cost_area_before": 0.0, "cost_area_after": 0.0,
    }])
    assert merge_review_events(audit_log=log) == []


def test_nan_costs_become_none():
    eid = make_event_id(50, 0, 1)
    log = pd.DataFrame([_make_audit_row(eid, cost_before=float("nan"),
                                        cost_after=float("nan"))])
    events = merge_review_events(audit_log=log)
    assert events[0].audit_cost_before is None
    assert events[0].audit_cost_after is None
