"""Pose-normalised fingerprint and ToxTrac continuity scoring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pylace.posthoc.appearance import (
    RunningMedianFingerprint,
    area_continuity_score,
    axis_ratio_continuity_score,
    canonicalize_patch_by_intensity,
    extract_patch,
    is_confident_frame,
    match_patch,
)


# ─────────────────────────────────────────────────────────────────
#  extract_patch
# ─────────────────────────────────────────────────────────────────


def test_extract_patch_centres_dark_blob_on_bright_background():
    """The dark fly silhouette ends up in the patch centre."""
    frame = np.full((200, 300), 200, dtype=np.uint8)
    cv2.ellipse(frame, (150, 100), (40, 15), 0, 0, 360, 60, thickness=-1)
    patch = extract_patch(
        frame, cx=150, cy=100, orientation_deg=90.0,
        major_axis_px=80.0, minor_axis_px=30.0, patch_h=16, patch_w=32,
    )
    assert patch.shape == (16, 32)
    assert patch.dtype == np.uint8
    centre = patch[6:10, 14:18].astype(np.float32).mean()
    corners = patch[[0, 0, -1, -1], [0, -1, 0, -1]].astype(np.float32).mean()
    assert centre < 100, "fly body should be dark in the patch centre"
    assert corners > 150, "background should be bright at the corners"


def test_extract_patch_aligns_major_axis_horizontally():
    """A vertical-major-axis ellipse and a horizontal one produce similar patches."""
    frame_h = np.full((200, 200), 200, dtype=np.uint8)
    frame_v = np.full((200, 200), 200, dtype=np.uint8)
    # Horizontal major axis: ellipse axes (40, 15) at angle 0 → fitEllipse angle ~ 90.
    cv2.ellipse(frame_h, (100, 100), (40, 15), 0, 0, 360, 60, thickness=-1)
    # Vertical major axis: ellipse axes (15, 40) at angle 0 → fitEllipse angle ~ 0.
    cv2.ellipse(frame_v, (100, 100), (15, 40), 0, 0, 360, 60, thickness=-1)
    patch_h = extract_patch(
        frame_h, cx=100, cy=100, orientation_deg=90.0,
        major_axis_px=80.0, minor_axis_px=30.0, patch_h=16, patch_w=32,
    )
    patch_v = extract_patch(
        frame_v, cx=100, cy=100, orientation_deg=0.0,
        major_axis_px=80.0, minor_axis_px=30.0, patch_h=16, patch_w=32,
    )
    rmse = match_patch(patch_h, patch_v)
    assert rmse < 10.0, f"pose-normalisation should align both, got rmse={rmse}"


def test_canonicalize_patch_puts_dark_half_on_right():
    """Left half darker than right → flipped 180; otherwise unchanged."""
    # Dark on right (already canonical):
    p = np.full((8, 16), 200, dtype=np.uint8)
    p[:, 8:] = 50
    out = canonicalize_patch_by_intensity(p)
    assert (out == p).all(), "patch with darker-right should not be flipped"
    # Dark on left (needs flip):
    p2 = np.full((8, 16), 200, dtype=np.uint8)
    p2[:, :8] = 50
    out2 = canonicalize_patch_by_intensity(p2)
    # After 180 flip, the dark half (originally left) is now on the right.
    assert float(out2[:, 8:].mean()) < float(out2[:, :8].mean())


def test_extract_patch_canonicalizes_head_abdomen():
    """Two flies painted at +x and -x orientations produce the same patch."""
    # Build a frame with an asymmetric body: dark "abdomen" on the right side,
    # lighter "head" on the left, all on a bright background.
    def paint(angle):
        frame = np.full((200, 200), 220, dtype=np.uint8)
        # Two halves of an ellipse: light + dark, joined.
        cv2.ellipse(frame, (100, 100), (40, 14), angle, 0, 360, 100, thickness=-1)
        # Add a darker spot on the +x side (head-pointing direction depends on
        # angle): place the spot at angle "behind" the centre.
        rad = np.deg2rad(angle)
        # cv2.ellipse(angle): rotates the ellipse so its +x end points
        # in image direction (cos(angle), sin(angle)). We want a darker
        # blob on that end.
        head_x = int(100 + 30 * np.cos(rad))
        head_y = int(100 + 30 * np.sin(rad))
        cv2.circle(frame, (head_x, head_y), 6, 30, thickness=-1)
        return frame

    # Paint at angle 0 deg:
    f0 = paint(0)
    contours, _ = cv2.findContours(
        (255 - f0 > 80).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    (cx0, cy0), (a, b), fit0 = cv2.fitEllipse(contours[0])
    p0 = extract_patch(
        f0, cx=cx0, cy=cy0, orientation_deg=fit0,
        major_axis_px=max(a, b), minor_axis_px=min(a, b),
    )

    # Paint at angle 180 deg (same fly, same shape, just rotated 180).
    f180 = paint(180)
    contours, _ = cv2.findContours(
        (255 - f180 > 80).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    (cx2, cy2), (a, b), fit2 = cv2.fitEllipse(contours[0])
    p180 = extract_patch(
        f180, cx=cx2, cy=cy2, orientation_deg=fit2,
        major_axis_px=max(a, b), minor_axis_px=min(a, b),
    )

    # After canonicalization both should look the same, regardless of
    # which 180-degree branch fitEllipse landed on.
    rmse = float(np.sqrt(((p0.astype(np.float32) - p180.astype(np.float32))**2).mean()))
    assert rmse < 15.0, (
        f"canonicalized patches from 0-deg and 180-deg orientations "
        f"should match closely, got rmse={rmse}"
    )


def test_extract_patch_can_disable_canonicalization():
    """Disabling canonicalization keeps the raw warpAffine output."""
    frame = np.full((200, 200), 220, dtype=np.uint8)
    cv2.ellipse(frame, (100, 100), (40, 14), 0, 0, 360, 100, thickness=-1)
    cv2.circle(frame, (130, 100), 6, 30, thickness=-1)
    p_canon = extract_patch(
        frame, cx=100, cy=100, orientation_deg=90.0,
        major_axis_px=80.0, minor_axis_px=28.0,
        canonicalize_head_abdomen=True,
    )
    p_raw = extract_patch(
        frame, cx=100, cy=100, orientation_deg=90.0,
        major_axis_px=80.0, minor_axis_px=28.0,
        canonicalize_head_abdomen=False,
    )
    # Both should be horizontal (rotation works either way), but canon
    # may have been flipped to put the dark spot on the +x side.
    canon_dark_right = float(p_canon[:, 16:].mean()) < float(p_canon[:, :16].mean())
    assert canon_dark_right, "canonicalized patch should have darker right half"
    # Raw output may or may not be canonical depending on fitEllipse's
    # 180-degree branch — we just check we can disable the step.
    if (p_canon != p_raw).any():
        # If they differ, p_raw should be the 180-degree flip of p_canon.
        assert (np.flip(p_raw, axis=(0, 1)) == p_canon).all()


@pytest.mark.parametrize("paint_angle", [0, 15, 30, 45, 60, 75, 90, 135])
def test_extract_patch_horizontal_for_all_paint_angles(paint_angle):
    """Patches are horizontal regardless of the original fly orientation.

    Earlier the rotation used (90 - orientation_deg) which has the
    wrong sign and produced 60-90 degree misalignment for in-between
    angles. The visual symptom: each track's confident-frame patches
    were oriented in random directions, so the per-pixel median
    washed out the head/abdomen asymmetry and the appearance term
    barely discriminated between flies.
    """
    frame = np.full((200, 200), 200, dtype=np.uint8)
    cv2.ellipse(frame, (100, 100), (50, 15), paint_angle, 0, 360, 50, thickness=-1)
    contours, _ = cv2.findContours(
        (255 - frame > 100).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    (cx, cy), (a, b), fit_angle = cv2.fitEllipse(contours[0])
    major, minor = max(a, b), min(a, b)
    patch = extract_patch(
        frame, cx=cx, cy=cy, orientation_deg=fit_angle,
        major_axis_px=major, minor_axis_px=minor, patch_h=16, patch_w=32,
    )
    # The fly should fill the central rows and leave the edge rows blank.
    centre_rows_dark = (patch[6:10] < 150).sum()
    edge_rows_dark = (patch[[0, 1, -2, -1]] < 150).sum()
    assert centre_rows_dark > 60, (
        f"paint_angle={paint_angle}: centre rows have {centre_rows_dark} "
        f"dark pixels, expected > 60 (fly should be horizontal)"
    )
    assert edge_rows_dark < 5, (
        f"paint_angle={paint_angle}: edge rows have {edge_rows_dark} "
        f"dark pixels, expected < 5 (fly should not bleed to the edges)"
    )


# ─────────────────────────────────────────────────────────────────
#  match_patch
# ─────────────────────────────────────────────────────────────────


def test_match_patch_self_match_is_zero():
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 255, size=(16, 32), dtype=np.uint8)
    assert match_patch(patch, patch) == pytest.approx(0.0)


def test_match_patch_handles_180_degree_flip():
    """A patch and its 180-deg flip should match perfectly via the flip branch."""
    rng = np.random.default_rng(1)
    patch = rng.integers(0, 255, size=(16, 32), dtype=np.uint8)
    flipped = np.flip(patch, axis=(0, 1))
    assert match_patch(patch, flipped) == pytest.approx(0.0)


def test_match_patch_distinguishes_genuinely_different_patches():
    a = np.full((16, 32), 50, dtype=np.uint8)
    b = np.full((16, 32), 200, dtype=np.uint8)
    assert match_patch(a, b) == pytest.approx(150.0, abs=1e-3)


def test_match_patch_shape_mismatch_raises():
    a = np.zeros((16, 32), dtype=np.uint8)
    b = np.zeros((8, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        match_patch(a, b)


# ─────────────────────────────────────────────────────────────────
#  Continuity scores
# ─────────────────────────────────────────────────────────────────


def test_axis_ratio_continuity_score_zero_on_match():
    assert axis_ratio_continuity_score(80.0, 30.0, 80.0 / 30.0) == pytest.approx(0.0)


def test_axis_ratio_continuity_score_normalises_by_reference():
    # Candidate axis_ratio = 4.0, ref = 2.0 → diff = 2.0, normalised = 1.0.
    assert axis_ratio_continuity_score(80.0, 20.0, 2.0) == pytest.approx(1.0)


def test_area_continuity_score_zero_on_match():
    assert area_continuity_score(1000.0, 1000.0) == pytest.approx(0.0)


def test_area_continuity_score_normalises_by_reference():
    assert area_continuity_score(1500.0, 1000.0) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────
#  Confidence detection
# ─────────────────────────────────────────────────────────────────


def test_is_confident_frame_accepts_clean_three_fly_frame():
    cx = np.array([10.0, 200.0, 50.0])
    cy = np.array([10.0, 100.0, 150.0])
    area = np.array([1000.0, 1050.0, 1020.0])
    assert is_confident_frame(
        cx, cy, area, n_animals=3, min_pairwise_distance_px=20,
        expected_area_px=1000.0, area_tol=0.3,
    )


def test_is_confident_frame_rejects_near_merge():
    cx = np.array([10.0, 12.0, 50.0])
    cy = np.array([10.0, 12.0, 150.0])
    area = np.array([1000.0, 1050.0, 1020.0])
    assert not is_confident_frame(
        cx, cy, area, n_animals=3, min_pairwise_distance_px=20,
        expected_area_px=1000.0, area_tol=0.3,
    )


def test_is_confident_frame_rejects_oversized_blob():
    cx = np.array([10.0, 200.0, 50.0])
    cy = np.array([10.0, 100.0, 150.0])
    area = np.array([1000.0, 2500.0, 1020.0])
    assert not is_confident_frame(
        cx, cy, area, n_animals=3, min_pairwise_distance_px=20,
        expected_area_px=1000.0, area_tol=0.3,
    )


def test_is_confident_frame_rejects_wrong_count():
    cx = np.array([10.0, 200.0])
    cy = np.array([10.0, 100.0])
    area = np.array([1000.0, 1050.0])
    assert not is_confident_frame(
        cx, cy, area, n_animals=3, min_pairwise_distance_px=20,
        expected_area_px=1000.0, area_tol=0.3,
    )


# ─────────────────────────────────────────────────────────────────
#  RunningMedianFingerprint
# ─────────────────────────────────────────────────────────────────


def test_running_median_returns_none_for_unknown_id():
    rmf = RunningMedianFingerprint()
    assert rmf.median_patch(99) is None
    assert rmf.median_area(99) is None
    assert rmf.median_axis_ratio(99) is None


def test_running_median_caps_history_at_history_frames():
    rmf = RunningMedianFingerprint(history_frames=5)
    patch = np.full((4, 8), 100, dtype=np.uint8)
    for area in range(20):
        rmf.update(0, patch=patch, area_px=float(area), major_axis_px=20.0,
                   minor_axis_px=10.0)
    # Only the last 5 areas (15..19) should remain → median = 17.0.
    assert rmf.median_area(0) == pytest.approx(17.0)


def test_running_median_patch_returns_pixelwise_median():
    rmf = RunningMedianFingerprint(history_frames=5)
    a = np.full((4, 8), 50, dtype=np.uint8)
    b = np.full((4, 8), 200, dtype=np.uint8)
    rmf.update(0, patch=a, area_px=1000.0, major_axis_px=20.0, minor_axis_px=10.0)
    rmf.update(0, patch=a, area_px=1000.0, major_axis_px=20.0, minor_axis_px=10.0)
    rmf.update(0, patch=b, area_px=1000.0, major_axis_px=20.0, minor_axis_px=10.0)
    med = rmf.median_patch(0)
    # Median of [50, 50, 200] = 50.
    assert med is not None
    assert (med == 50).all()


def test_running_median_validates_history_frames():
    with pytest.raises(ValueError, match="history_frames"):
        RunningMedianFingerprint(history_frames=0)
