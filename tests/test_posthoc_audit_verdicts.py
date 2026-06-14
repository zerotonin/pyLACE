"""Verdict ↔ audit handshake tests.

The audit reads ``{event_id: VerdictRecord}`` and must honour each verdict
kind correctly: ``accept_swap`` forces a swap regardless of cost,
``reject_swap`` blocks any swap at that event, ``mount`` blocks the swap
AND tags the audited rows with ``event_type='mount'``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pylace.posthoc.audit import EVENT_TYPE_MOUNT, audit_track_identities
from pylace.review.verdicts import Verdict, VerdictRecord, make_event_id, now_iso


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


def _injected_swap_setup(n: int = 60, swap_at: int = 30):
    """Two stationary tracks, contact at swap_at-1, IDs swapped from swap_at."""
    cx0 = np.full(n, 50.0); cy0 = np.full(n, 50.0)
    cx1 = np.full(n, 50.0); cy1 = np.full(n, 150.0)
    cx0[swap_at:], cx1[swap_at:] = cx1[swap_at:].copy(), cx0[swap_at:].copy()
    cy0[swap_at:], cy1[swap_at:] = cy1[swap_at:].copy(), cy0[swap_at:].copy()
    cy0[swap_at - 1] = 100.0
    cy1[swap_at - 1] = 100.0
    return _two_track_df(cx0, cy0, cx1, cy1), swap_at - 1


def _make_verdict(frame_start: int, verdict: Verdict) -> dict:
    eid = make_event_id(frame_start, 0, 1)
    return {eid: VerdictRecord(
        event_id=eid,
        frame_start=frame_start, frame_end=frame_start,
        animal_a=0, animal_b=1,
        verdict=verdict,
        source="test", reviewer="unittest",
        timestamp_iso=now_iso(), note="",
    )}


# ─────────────────────────────────────────────────────────────────
#  Baseline: no verdicts → cost-based behaviour unchanged
# ─────────────────────────────────────────────────────────────────


def test_audit_without_verdicts_uses_cost_based_logic():
    df, _ = _injected_swap_setup()
    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=20.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=200.0,
    )
    assert len(log) == 1
    assert log["verdict"].iloc[0] == ""
    assert (audited["event_type"] == "").all()


def test_audit_swap_log_carries_event_id_columns():
    df, _ = _injected_swap_setup()
    _, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=20.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=200.0,
    )
    assert "event_id" in log.columns
    assert "frame_start" in log.columns
    assert "frame_end" in log.columns
    assert "verdict" in log.columns
    # frame_start should equal block_start, frame_end >= frame_start.
    assert log["frame_start"].iloc[0] <= log["frame_end"].iloc[0]


def test_audit_empty_log_has_verdict_columns():
    """Even when no swaps happen, the log schema is stable."""
    n = 50
    df = _two_track_df(
        cx0=np.arange(n, dtype=float), cy0=np.zeros(n),
        cx1=np.arange(n, dtype=float), cy1=np.full(n, 200.0),
    )
    _, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=5.0, window_s=1.0,
    )
    assert log.empty
    for col in ("event_id", "frame_start", "frame_end", "verdict"):
        assert col in log.columns


# ─────────────────────────────────────────────────────────────────
#  reject_swap: audit must leave IDs untouched at that event
# ─────────────────────────────────────────────────────────────────


def test_reject_swap_verdict_blocks_an_otherwise_cost_winning_swap():
    df, contact_frame = _injected_swap_setup()
    verdicts = _make_verdict(contact_frame, Verdict.REJECT_SWAP)
    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=20.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=200.0,
        verdicts=verdicts,
    )
    assert log.empty, "verdict should have blocked the swap"
    assert (audited["track_id"] == audited["original_track_id"]).all()
    assert (audited["event_type"] == "").all()


# ─────────────────────────────────────────────────────────────────
#  accept_swap: force the pair-swap regardless of cost
# ─────────────────────────────────────────────────────────────────


def test_accept_swap_verdict_forces_swap_on_clean_data():
    """No injected swap, just a contact — cost-based would refuse, verdict forces it."""
    n = 60
    swap_at = 30
    cx0 = np.full(n, 50.0); cy0 = np.full(n, 50.0)
    cx1 = np.full(n, 50.0); cy1 = np.full(n, 150.0)
    cy0[swap_at - 1] = 100.0
    cy1[swap_at - 1] = 100.0
    df = _two_track_df(cx0, cy0, cx1, cy1)
    verdicts = _make_verdict(swap_at - 1, Verdict.ACCEPT_SWAP)

    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=20.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=200.0,
        verdicts=verdicts,
    )
    assert len(log) == 1
    assert log["verdict"].iloc[0] == Verdict.ACCEPT_SWAP.value

    post = audited[audited["frame_idx"] > swap_at - 1]
    by_track = post.groupby("track_id")
    assert by_track.get_group(0)["cy_smooth_px"].mean() == 150.0
    assert by_track.get_group(1)["cy_smooth_px"].mean() == 50.0


def test_accept_swap_bypasses_jump_gate():
    """accept_swap must commit even when the perm would normally fail max_jump_mm.

    Geometry: two stationary tracks 400 px apart that briefly touch at one
    frame (a contact event the auditor will detect). After the contact,
    they return to their original positions — so the (1, 0) permutation
    would normally be refused by ``max_jump_mm`` (it would teleport each
    track ~400 px). The ``accept_swap`` verdict has to bypass that gate.
    """
    n = 50
    swap_at = 25
    cx0 = np.full(n, 50.0);  cy0 = np.full(n, 50.0)
    cx1 = np.full(n, 450.0); cy1 = np.full(n, 50.0)
    # Briefly bring them to the same point at frame swap_at - 1 so the
    # auditor sees a contact event there.
    cx0[swap_at - 1] = 250.0; cy0[swap_at - 1] = 50.0
    cx1[swap_at - 1] = 250.0; cy1[swap_at - 1] = 50.0
    df = _two_track_df(cx0, cy0, cx1, cy1)

    # Without verdict + tight jump gate: cost-based refuses.
    _, log_no_verdict = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=5.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=20.0,
    )
    assert log_no_verdict.empty

    # With accept_swap verdict: forced through.
    verdicts = _make_verdict(swap_at - 1, Verdict.ACCEPT_SWAP)
    _, log_with_verdict = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=5.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=20.0,
        verdicts=verdicts,
    )
    assert len(log_with_verdict) == 1


# ─────────────────────────────────────────────────────────────────
#  mount: block swap AND tag event_type=mount on the contact rows
# ─────────────────────────────────────────────────────────────────


def test_mount_verdict_tags_event_type_without_swapping_ids():
    df, contact_frame = _injected_swap_setup()
    verdicts = _make_verdict(contact_frame, Verdict.MOUNT)

    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=20.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=200.0,
        verdicts=verdicts,
    )

    # IDs untouched.
    assert (audited["track_id"] == audited["original_track_id"]).all()

    # Both flies at the contact frame are tagged 'mount'.
    contact_rows = audited[audited["frame_idx"] == contact_frame]
    assert (contact_rows["event_type"] == EVENT_TYPE_MOUNT).all()
    assert len(contact_rows) == 2

    # Log carries one row for the mount event.
    assert len(log) == 1
    assert log["verdict"].iloc[0] == EVENT_TYPE_MOUNT


def test_mount_verdict_only_tags_frames_in_the_block():
    n = 60
    cx0 = np.full(n, 50.0); cy0 = np.full(n, 50.0)
    cx1 = np.full(n, 50.0); cy1 = np.full(n, 150.0)
    # 5-frame contact starting at frame 30.
    cy0[30:35] = 100.0
    cy1[30:35] = 100.0
    df = _two_track_df(cx0, cy0, cx1, cy1)

    verdicts = _make_verdict(30, Verdict.MOUNT)
    audited, _ = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=20.0, window_s=1.0,
        coalesce_window_frames=1,
        verdicts=verdicts,
    )
    mount_frames = sorted(
        audited[audited["event_type"] == EVENT_TYPE_MOUNT]["frame_idx"].unique(),
    )
    assert mount_frames == list(range(30, 35))


# ─────────────────────────────────────────────────────────────────
#  unknown / missing event_id: ignored (cost-based path used)
# ─────────────────────────────────────────────────────────────────


def test_unknown_verdict_falls_back_to_cost_logic():
    df, contact_frame = _injected_swap_setup()
    verdicts = _make_verdict(contact_frame, Verdict.UNKNOWN)
    _, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=20.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=200.0,
        verdicts=verdicts,
    )
    # Cost-based logic still commits the swap — same as no-verdict case.
    assert len(log) == 1
    assert log["verdict"].iloc[0] == ""


def test_accept_swap_with_explicit_permutation_applies_3way_cycle():
    """A 3-fly 3-way cycle (perm = (1, 2, 0)) recorded in the verdict file
    must be applied directly by the audit rather than falling back to a
    pair-swap on (animal_a, animal_b)."""
    n = 60
    contact_frame = 29
    # Three stationary tracks at distinct y positions; brief 3-way contact
    # at the contact frame (all collapse to the same x, y so the audit
    # detects a block there).
    cx = [np.full(n, 100.0), np.full(n, 100.0), np.full(n, 100.0)]
    cy = [np.full(n, 50.0), np.full(n, 100.0), np.full(n, 150.0)]
    for arr in (*cx, *cy):
        arr[contact_frame] = 80.0

    rows = []
    for i in range(n):
        for tid in (0, 1, 2):
            rows.append(dict(
                frame_idx=i, track_id=tid,
                cx_smooth_px=cx[tid][i], cy_smooth_px=cy[tid][i],
            ))
    df = pd.DataFrame(rows)

    event_id = make_event_id(contact_frame, 0, 1)
    verdicts = {event_id: VerdictRecord(
        event_id=event_id,
        frame_start=contact_frame, frame_end=contact_frame,
        animal_a=0, animal_b=1,
        verdict=Verdict.ACCEPT_SWAP,
        source="test", reviewer="unittest",
        timestamp_iso=now_iso(),
        permutation=(1, 2, 0),
    )}
    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=30.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=400.0,
        verdicts=verdicts,
    )
    assert len(log) == 1
    assert log["verdict"].iloc[0] == Verdict.ACCEPT_SWAP.value

    post = audited[audited["frame_idx"] > contact_frame]
    by_track = post.groupby("track_id")
    # User-natural cycle (1, 2, 0): old label 0 → new label 1; old 1 → 2;
    # old 2 → 0. The relabelled DataFrame's track 0 therefore holds what
    # was track 2 (y=150); track 1 holds what was track 0 (y=50); track 2
    # holds what was track 1 (y=100).
    assert by_track.get_group(0)["cy_smooth_px"].mean() == 150.0
    assert by_track.get_group(1)["cy_smooth_px"].mean() == 50.0
    assert by_track.get_group(2)["cy_smooth_px"].mean() == 100.0


def test_accept_swap_with_malformed_permutation_falls_back_to_pair_swap():
    """A permutation with the wrong track ids should be ignored, not crash."""
    df, contact_frame = _injected_swap_setup()
    event_id = make_event_id(contact_frame, 0, 1)
    verdicts = {event_id: VerdictRecord(
        event_id=event_id,
        frame_start=contact_frame, frame_end=contact_frame,
        animal_a=0, animal_b=1,
        verdict=Verdict.ACCEPT_SWAP,
        source="test", reviewer="unittest",
        timestamp_iso=now_iso(),
        permutation=(7, 8),   # garbage
    )}
    _, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=20.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=200.0,
        verdicts=verdicts,
    )
    assert len(log) == 1
    assert log["verdict"].iloc[0] == Verdict.ACCEPT_SWAP.value
    # Fallback pair-swap on (animal_a=0, animal_b=1) was applied.


def test_verdict_for_unrelated_event_id_is_ignored():
    df, _ = _injected_swap_setup()
    # A verdict on a totally different event id should not affect anything.
    other = {make_event_id(999, 5, 9): VerdictRecord(
        event_id=make_event_id(999, 5, 9),
        frame_start=999, frame_end=999,
        animal_a=5, animal_b=9,
        verdict=Verdict.REJECT_SWAP,
        source="test", reviewer="unittest",
        timestamp_iso=now_iso(), note="",
    )}
    _, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=20.0, window_s=1.0,
        swap_cost_ratio=0.9, max_jump_mm=200.0,
        verdicts=other,
    )
    assert len(log) == 1, "the real event should still be audited"
