"""Identity audit: detect and undo post-merge ID swaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pylace.posthoc.audit import (
    _block_cost,
    _find_event_blocks,
    audit_track_identities,
)


def _build_two_track_df(
    cx0: np.ndarray, cy0: np.ndarray,
    cx1: np.ndarray, cy1: np.ndarray,
) -> pd.DataFrame:
    n = cx0.size
    rows = []
    for i in range(n):
        rows.append(dict(
            frame_idx=i, track_id=0,
            cx_smooth_px=cx0[i], cy_smooth_px=cy0[i],
        ))
        rows.append(dict(
            frame_idx=i, track_id=1,
            cx_smooth_px=cx1[i], cy_smooth_px=cy1[i],
        ))
    return pd.DataFrame(rows)


def _build_three_track_df(arrs: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    n = arrs[0][0].size
    rows = []
    for i in range(n):
        for tid, (cx, cy) in enumerate(arrs):
            rows.append(dict(
                frame_idx=i, track_id=tid,
                cx_smooth_px=cx[i], cy_smooth_px=cy[i],
            ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
#  Pure-helper tests
# ─────────────────────────────────────────────────────────────────


def test_find_event_blocks_with_proximity():
    n = 20
    cx = np.zeros((n, 2))
    cy = np.zeros((n, 2))
    cx[:, 0] = np.arange(n)
    cx[:, 1] = np.arange(n) + 50.0
    cx[10, 1] = cx[10, 0] + 1.0  # contact at frame 10
    blocks = _find_event_blocks(cx, cy,
                                contact_threshold_px=2.0,
                                coalesce_window_frames=2)
    assert blocks == [(10, 10)]


def test_find_event_blocks_coalesces_neighbours():
    n = 20
    cx = np.zeros((n, 2))
    cy = np.zeros((n, 2))
    cx[:, 0] = np.arange(n)
    cx[:, 1] = np.arange(n) + 50.0
    cx[10:13, 1] = cx[10:13, 0] + 1.0  # contact frames 10–12
    blocks = _find_event_blocks(cx, cy,
                                contact_threshold_px=2.0,
                                coalesce_window_frames=2)
    assert blocks == [(10, 12)]


def test_find_event_blocks_treats_nan_as_event():
    n = 10
    cx = np.zeros((n, 2))
    cy = np.zeros((n, 2))
    cx[:, 0] = np.arange(n) * 1.0
    cx[:, 1] = np.arange(n) * 1.0 + 50.0
    cx[5, 0] = np.nan
    blocks = _find_event_blocks(cx, cy,
                                contact_threshold_px=1.0,
                                coalesce_window_frames=2)
    assert blocks == [(5, 5)]


def test_block_cost_zero_for_consistent_motion():
    n = 30
    cx = np.zeros((n, 2))
    cy = np.zeros((n, 2))
    cx[:, 0] = np.arange(n) * 1.0
    cy[:, 0] = 0.0
    cx[:, 1] = np.arange(n) * 1.0
    cy[:, 1] = 100.0
    # Identity should have ~zero cost; it's the consistent assignment.
    c = _block_cost(cx, cy, pre_lo=0, pre_hi=10, post_lo=15, post_hi=25,
                   perm=(0, 1))
    assert c < 0.5
    # Swapped should have a large cost.
    c_swap = _block_cost(cx, cy, pre_lo=0, pre_hi=10, post_lo=15, post_hi=25,
                         perm=(1, 0))
    assert c_swap > c * 10


# ─────────────────────────────────────────────────────────────────
#  End-to-end audit
# ─────────────────────────────────────────────────────────────────


def test_audit_undoes_a_clean_two_track_swap():
    """Two flies pass each other; an injected ID swap is detected and undone."""
    n = 50
    t = np.arange(n, dtype=float)
    # Track 0 truly moves +x at y=0; track 1 truly moves +x at y=100.
    # The CSV has them correct for frames 0..29, then swapped from frame 30.
    swap_at = 30
    cx0_truth = t.copy()
    cy0_truth = np.zeros(n)
    cx1_truth = t.copy()
    cy1_truth = np.full(n, 100.0)
    # Inject swap.
    cx0 = cx0_truth.copy(); cy0 = cy0_truth.copy()
    cx1 = cx1_truth.copy(); cy1 = cy1_truth.copy()
    cx0[swap_at:], cx1[swap_at:] = cx1[swap_at:].copy(), cx0[swap_at:].copy()
    cy0[swap_at:], cy1[swap_at:] = cy1[swap_at:].copy(), cy0[swap_at:].copy()
    # Force a contact event near the swap frame so the auditor visits it.
    cy0[swap_at - 1] = 50.0
    cy1[swap_at - 1] = 50.0
    df = _build_two_track_df(cx0, cy0, cx1, cy1)

    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=60.0, window_s=1.0,
        swap_cost_ratio=0.9,
    )
    assert not log.empty, "expected at least one swap commit"

    # After audit, track 0 should sit at y≈0 throughout, track 1 at y≈100.
    by_track = audited.groupby("track_id")
    cy0_audited = by_track.get_group(0)["cy_smooth_px"].to_numpy()
    cy1_audited = by_track.get_group(1)["cy_smooth_px"].to_numpy()
    # The contact frame is intentionally noisy; check the rest.
    far = np.arange(n) != (swap_at - 1)
    assert np.allclose(cy0_audited[far], 0.0, atol=1e-6)
    assert np.allclose(cy1_audited[far], 100.0, atol=1e-6)


def test_audit_no_op_on_well_separated_tracks():
    """When tracks never come close, the audit should not invent swaps."""
    n = 50
    t = np.arange(n, dtype=float)
    df = _build_two_track_df(
        cx0=t,            cy0=np.zeros(n),
        cx1=t,            cy1=np.full(n, 200.0),
    )
    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=5.0,  # 200 >> 5 so no events
        window_s=1.0,
    )
    assert log.empty
    # Identity of track_id values preserved.
    assert (audited["track_id"] == audited["original_track_id"]).all()


def test_audit_preserves_stationary_track_through_a_two_fly_swap():
    """Three tracks: one stays still, two cross. The still one keeps its ID."""
    n = 60
    t = np.arange(n, dtype=float)
    # Track 0: stationary at (10, 100).
    cx0 = np.full(n, 10.0); cy0 = np.full(n, 100.0)
    # Track 1: moves +x at y = 0 → 60.
    cx1 = t.copy(); cy1 = t.copy()
    # Track 2: moves +x at y = 60 → 0 (crossing track 1).
    cx2 = t.copy(); cy2 = 60.0 - t.copy()
    # Inject swap of 1 ↔ 2 from the crossing point.
    swap_at = 30
    cx1[swap_at:], cx2[swap_at:] = cx2[swap_at:].copy(), cx1[swap_at:].copy()
    cy1[swap_at:], cy2[swap_at:] = cy2[swap_at:].copy(), cy1[swap_at:].copy()
    df = _build_three_track_df([(cx0, cy0), (cx1, cy1), (cx2, cy2)])

    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=10.0, window_s=1.0,
        swap_cost_ratio=0.9,
    )
    # Track 0 should stay at (10, 100) under the same id throughout.
    audited_0 = audited[audited["track_id"] == 0]
    assert np.allclose(audited_0["cx_smooth_px"].to_numpy(), 10.0)
    assert np.allclose(audited_0["cy_smooth_px"].to_numpy(), 100.0)


def test_audit_returns_trivial_for_single_track():
    n = 10
    df = pd.DataFrame({
        "frame_idx": np.arange(n),
        "track_id": [0] * n,
        "cx_smooth_px": np.arange(n, dtype=float),
        "cy_smooth_px": np.zeros(n),
    })
    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
    )
    assert log.empty
    assert (audited["track_id"] == audited["original_track_id"]).all()


def test_audit_rejects_invalid_pix_per_mm():
    df = _build_two_track_df(
        cx0=np.arange(10, dtype=float), cy0=np.zeros(10),
        cx1=np.arange(10, dtype=float), cy1=np.full(10, 100.0),
    )
    with pytest.raises(ValueError, match="pix_per_mm"):
        audit_track_identities(df, fps=10.0, pix_per_mm=0.0)


def test_audit_rejects_invalid_swap_cost_ratio():
    df = _build_two_track_df(
        cx0=np.arange(10, dtype=float), cy0=np.zeros(10),
        cx1=np.arange(10, dtype=float), cy1=np.full(10, 100.0),
    )
    with pytest.raises(ValueError, match="swap_cost_ratio"):
        audit_track_identities(df, fps=10.0, pix_per_mm=1.0,
                                swap_cost_ratio=0.0)
    with pytest.raises(ValueError, match="swap_cost_ratio"):
        audit_track_identities(df, fps=10.0, pix_per_mm=1.0,
                                swap_cost_ratio=1.5)


def test_audit_kalman_catches_a_swap_between_two_stationary_tracks():
    """Two near-stationary flies trade IDs — the lite median-velocity
    predictor produced near-zero velocity for both pre-event so the
    cost was indifferent to the swap. The Kalman version uses
    Mahalanobis² which depends on the spatial covariance, not motion,
    so the swap is sharply detected.
    """
    n = 40
    swap_at = 20
    # Track 0 sits near (5, 5). Track 1 sits near (5, 80).
    cx0 = np.full(n, 5.0); cy0 = np.full(n, 5.0)
    cx1 = np.full(n, 5.0); cy1 = np.full(n, 80.0)
    # Inject a swap.
    cx0[swap_at:], cx1[swap_at:] = cx1[swap_at:].copy(), cx0[swap_at:].copy()
    cy0[swap_at:], cy1[swap_at:] = cy1[swap_at:].copy(), cy0[swap_at:].copy()
    # Force a contact event near the swap so the auditor visits it.
    cy0[swap_at - 1] = 40.0
    cy1[swap_at - 1] = 40.0
    df = _build_two_track_df(cx0, cy0, cx1, cy1)
    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=50.0, window_s=1.0,
        swap_cost_ratio=0.9,
    )
    assert not log.empty, "Kalman audit should have caught the swap"
    # Track 0 should sit at y ≈ 5 throughout (except the contact frame).
    by_track = audited.groupby("track_id")
    cy0_audited = by_track.get_group(0)["cy_smooth_px"].to_numpy()
    far = np.arange(n) != (swap_at - 1)
    assert np.allclose(cy0_audited[far], 5.0, atol=1e-6)
