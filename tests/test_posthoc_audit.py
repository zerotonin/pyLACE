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


def test_audit_skips_giant_event_blocks():
    """Event blocks longer than max_event_block_s are left untouched."""
    n = 200
    swap_at = 100
    # Two truly stationary tracks far apart, with a CSV-injected swap
    # in the middle. We then make the entire span around the swap a
    # contact event by shoving the y-coordinates close together for a
    # very long stretch — far longer than max_event_block_s — so the
    # block coalesces and gets skipped.
    cx0 = np.full(n, 5.0); cy0 = np.full(n, 5.0)
    cx1 = np.full(n, 5.0); cy1 = np.full(n, 80.0)
    cx0[swap_at:], cx1[swap_at:] = cx1[swap_at:].copy(), cx0[swap_at:].copy()
    cy0[swap_at:], cy1[swap_at:] = cy1[swap_at:].copy(), cy0[swap_at:].copy()
    # Long contact event spanning frames 20..180 (160 frames at 10 fps = 16 s,
    # well over the 5 × window_s = 5 s default cap).
    cy0[20:180] = 40.0
    cy1[20:180] = 40.0
    df = _build_two_track_df(cx0, cy0, cx1, cy1)

    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=50.0, window_s=1.0,
        swap_cost_ratio=0.9,
    )
    # Block was skipped → no commit, IDs unchanged from the (swapped) input.
    assert log.empty, "the giant block should have been skipped"
    assert (audited["track_id"] == audited["original_track_id"]).all()


def test_audit_position_jump_gate_refuses_teleport_perms():
    """A perm that would teleport a track is refused regardless of cost."""
    n = 50
    swap_at = 25
    # Two stationary tracks 400 px apart. CSV is honest (no injected
    # swap). A 1-frame contact event triggers the audit. The position-
    # jump gate must refuse the (1, 0) permutation because committing
    # it would teleport track 0 by ~400 px.
    cx0 = np.full(n, 50.0);  cy0 = np.full(n, 50.0)
    cx1 = np.full(n, 450.0); cy1 = np.full(n, 50.0)
    # Inject a single-frame "contact" by squashing the y-coords.
    cy0[swap_at] = 100.0
    cy1[swap_at] = 100.0
    df = _build_two_track_df(cx0, cy0, cx1, cy1)

    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=80.0, window_s=1.0,
        swap_cost_ratio=0.9,
        max_jump_mm=20.0,    # 20 mm × 1 px/mm = 20 px — well below 400 px.
    )
    assert log.empty, "the 400 px teleport should fail the position-jump gate"


def test_audit_position_jump_gate_does_not_block_legitimate_local_swap():
    """A real swap with both tracks staying in place still passes the gate."""
    n = 50
    swap_at = 25
    # Two near-stationary tracks 100 px apart, swapped at frame 25.
    cx0 = np.full(n, 50.0); cy0 = np.full(n, 50.0)
    cx1 = np.full(n, 50.0); cy1 = np.full(n, 150.0)
    cx0[swap_at:], cx1[swap_at:] = cx1[swap_at:].copy(), cx0[swap_at:].copy()
    cy0[swap_at:], cy1[swap_at:] = cy1[swap_at:].copy(), cy0[swap_at:].copy()
    cy0[swap_at - 1] = 100.0  # single-frame contact only
    cy1[swap_at - 1] = 100.0
    df = _build_two_track_df(cx0, cy0, cx1, cy1)

    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=20.0, window_s=1.0,
        swap_cost_ratio=0.9,
        max_jump_mm=200.0,   # > the 100 px swap distance, so swap passes.
    )
    assert not log.empty, "the local stationary swap should still be caught"


def test_audit_rejects_invalid_max_event_block_s():
    df = _build_two_track_df(
        cx0=np.arange(10, dtype=float), cy0=np.zeros(10),
        cx1=np.arange(10, dtype=float), cy1=np.full(10, 100.0),
    )
    with pytest.raises(ValueError, match="max_event_block_s"):
        audit_track_identities(df, fps=10.0, pix_per_mm=1.0,
                                max_event_block_s=0.0)


def test_audit_rejects_invalid_max_jump_mm():
    df = _build_two_track_df(
        cx0=np.arange(10, dtype=float), cy0=np.zeros(10),
        cx1=np.arange(10, dtype=float), cy1=np.full(10, 100.0),
    )
    with pytest.raises(ValueError, match="max_jump_mm"):
        audit_track_identities(df, fps=10.0, pix_per_mm=1.0,
                                max_jump_mm=-1.0)


def test_audit_appearance_breaks_kalman_ties(tmp_path):
    """Two identically-positioned identical-area flies, distinguishable only by appearance.

    Constructs a CSV where positions and areas are *symmetric* (so
    Kalman-Mahalanobis alone cannot favour either permutation), then
    injects an appearance fingerprint sidecar that's only consistent
    with the swap perm. The audit must commit the swap.
    """
    n = 60
    swap_at = 30
    # Two stationary tracks at (50, 50) and (150, 50). Identical areas.
    cx0 = np.full(n, 50.0); cy0 = np.full(n, 50.0)
    cx1 = np.full(n, 150.0); cy1 = np.full(n, 50.0)
    # Inject CSV swap at frame 30.
    cx0[swap_at:], cx1[swap_at:] = cx1[swap_at:].copy(), cx0[swap_at:].copy()
    cy0[swap_at:], cy1[swap_at:] = cy1[swap_at:].copy(), cy0[swap_at:].copy()
    # Force a contact event at frame 29 so the auditor visits.
    cx0[swap_at - 1] = 100.0
    cx1[swap_at - 1] = 100.0
    df = _build_two_track_df(cx0, cy0, cx1, cy1)

    # Build a fingerprint sidecar.
    h, w = 8, 16
    n_dets = 2 * n
    frame_idx_arr = np.repeat(np.arange(n), 2).astype(np.int32)
    track_id_arr = np.tile([0, 1], n).astype(np.int32)
    is_conf = np.ones(n_dets, dtype=bool)
    is_conf[2 * (swap_at - 1):2 * (swap_at - 1) + 2] = False  # contact frame not confident
    # Patches: track 0's TRUE fly is bright, track 1's TRUE fly is dark.
    # Because the CSV swaps the labels at frame swap_at, track_id 0 in
    # the CSV is at the bright fly's position pre-swap, then at the
    # dark fly's position post-swap. The fingerprint patches follow
    # the TRUE fly (bright vs dark), since the fingerprint extractor
    # would have seen each fly's actual intensity. So:
    #   pre-swap track_id 0 row → bright patch
    #   post-swap track_id 0 row → dark patch (because the CSV's track_id 0
    #     post-swap is at the dark fly's position).
    bright = np.full((h, w), 200, dtype=np.uint8)
    dark = np.full((h, w), 50, dtype=np.uint8)
    patches = np.zeros((n_dets, h, w), dtype=np.uint8)
    for k in range(n_dets):
        f = frame_idx_arr[k]
        t = track_id_arr[k]
        # Determine which TRUE fly this is.
        if f < swap_at:
            true_fly = t  # CSV honest pre-swap.
        else:
            true_fly = 1 - t  # CSV swapped post-swap.
        patches[k] = bright if true_fly == 0 else dark
    npz_path = tmp_path / "fp.npz"
    np.savez(
        npz_path,
        frame_idx=frame_idx_arr, track_id=track_id_arr, patch=patches,
        is_confident=is_conf,
        cx_px=np.zeros(n_dets, dtype=np.float32),
        cy_px=np.zeros(n_dets, dtype=np.float32),
        area_px=np.full(n_dets, 1000.0, dtype=np.float32),
        major_axis_px=np.full(n_dets, 40.0, dtype=np.float32),
        minor_axis_px=np.full(n_dets, 20.0, dtype=np.float32),
        orientation_deg=np.full(n_dets, 90.0, dtype=np.float32),
        patch_h=np.array(h, dtype=np.int32),
        patch_w=np.array(w, dtype=np.int32),
    )

    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=80.0, window_s=2.0,
        swap_cost_ratio=0.9,
        max_jump_mm=200.0,
        fingerprint_path=npz_path,
        appearance_weight=1.0,
        axis_ratio_weight=0.0,
        area_weight=0.0,
    )
    assert not log.empty, "appearance term should have caught the swap"
    # Track 0 should consistently sit at one true fly's intensity post-audit.
    # We don't check positions (they're symmetric) — we check the swap log.
    assert tuple(log.iloc[0]["permutation"]) == (1, 0)
    assert log.iloc[0]["cost_appearance_after"] < log.iloc[0]["cost_appearance_before"]


def test_audit_falls_back_to_kalman_when_no_fingerprint(tmp_path):
    """Without a fingerprint file, the audit behaves exactly as before."""
    n = 50
    swap_at = 25
    cx0 = np.full(n, 5.0); cy0 = np.full(n, 5.0)
    cx1 = np.full(n, 5.0); cy1 = np.full(n, 80.0)
    cx0[swap_at:], cx1[swap_at:] = cx1[swap_at:].copy(), cx0[swap_at:].copy()
    cy0[swap_at:], cy1[swap_at:] = cy1[swap_at:].copy(), cy0[swap_at:].copy()
    cy0[swap_at - 1] = 40.0
    cy1[swap_at - 1] = 40.0
    df = _build_two_track_df(cx0, cy0, cx1, cy1)

    audited, log = audit_track_identities(
        df, fps=10.0, pix_per_mm=1.0,
        contact_threshold_mm=50.0, window_s=1.0,
        swap_cost_ratio=0.9,
        # Path that doesn't exist: should silently fall back.
        fingerprint_path=tmp_path / "missing.npz",
    )
    assert not log.empty
    # Cost columns for appearance should all be 0 (no fingerprint).
    assert log.iloc[0]["cost_appearance_before"] == 0.0
    assert log.iloc[0]["cost_appearance_after"] == 0.0


def test_audit_rejects_negative_appearance_weight():
    df = _build_two_track_df(
        cx0=np.arange(10, dtype=float), cy0=np.zeros(10),
        cx1=np.arange(10, dtype=float), cy1=np.full(10, 100.0),
    )
    with pytest.raises(ValueError, match="appearance_weight"):
        audit_track_identities(df, fps=10.0, pix_per_mm=1.0,
                                appearance_weight=-1.0)


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
