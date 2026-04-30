"""Pure-logic tests for the background editor's inpaint helper."""

from __future__ import annotations

import numpy as np
import pytest

from pylace.tune.bg_editor import apply_inpaint


def test_inpaint_fills_dark_patch_with_bright_surround():
    bg = np.full((40, 40), 200, dtype=np.uint8)
    bg[15:25, 15:25] = 30  # the "stationary fly"
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[15:25, 15:25] = 255

    filled = apply_inpaint(bg, mask, radius=3)

    patch = filled[15:25, 15:25]
    assert patch.mean() > 150  # was 30
    assert patch.max() <= 255 and patch.min() >= 0
    # Outside the mask is byte-for-byte unchanged.
    assert np.array_equal(filled[0:5, 0:5], bg[0:5, 0:5])


def test_inpaint_returns_uint8_same_shape():
    bg = np.random.randint(150, 220, size=(20, 30), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:10, 10:15] = 255

    out = apply_inpaint(bg, mask, radius=3)

    assert out.shape == bg.shape
    assert out.dtype == np.uint8


def test_inpaint_with_empty_mask_is_lossless():
    bg = np.random.randint(0, 255, size=(20, 20), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    out = apply_inpaint(bg, mask, radius=3)
    assert np.array_equal(out, bg)


def test_inpaint_rejects_shape_mismatch():
    bg = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_inpaint(bg, mask, radius=3)


def test_inpaint_rejects_wrong_dtype():
    bg = np.zeros((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_inpaint(bg, mask, radius=3)


def test_inpaint_rejects_zero_radius():
    bg = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_inpaint(bg, mask, radius=0)


# ── copy_circle ────────────────────────────────────────────────────────


def test_copy_circle_writes_destination_from_source():
    from pylace.tune.bg_editor import copy_circle

    bg = np.full((40, 40), 50, dtype=np.uint8)
    bg[8:12, 8:12] = 220  # bright source patch
    copy_circle(bg, src_x=10, src_y=10, dst_x=30, dst_y=30, radius=3)

    # Destination centre is bright now (copied from source).
    assert bg[30, 30] == 220
    # Source unchanged.
    assert bg[10, 10] == 220


def test_copy_circle_only_writes_inside_circle():
    from pylace.tune.bg_editor import copy_circle

    bg = np.full((40, 40), 50, dtype=np.uint8)
    bg[5:15, 5:15] = 200
    copy_circle(bg, src_x=10, src_y=10, dst_x=30, dst_y=30, radius=2)

    # Outside the radius=2 disc, dest is unchanged.
    assert bg[33, 33] == 50
    assert bg[27, 27] == 50


def test_copy_circle_skips_when_source_out_of_bounds():
    from pylace.tune.bg_editor import copy_circle

    bg = np.full((20, 20), 50, dtype=np.uint8)
    # Source far outside the image.
    copy_circle(bg, src_x=200, src_y=200, dst_x=10, dst_y=10, radius=3)
    assert (bg == 50).all()


def test_copy_circle_rejects_zero_radius():
    from pylace.tune.bg_editor import copy_circle

    bg = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        copy_circle(bg, src_x=5, src_y=5, dst_x=2, dst_y=2, radius=0)


# ── heal_circle ────────────────────────────────────────────────────────


def test_heal_circle_modifies_destination_region():
    from pylace.tune.bg_editor import heal_circle

    rng = np.random.default_rng(0)
    bg = rng.integers(120, 140, size=(50, 50), dtype=np.uint8)
    bg[5:15, 5:15] = 200
    before_dst = bg[35:45, 35:45].copy()

    heal_circle(bg, src_x=10, src_y=10, dst_x=40, dst_y=40, radius=4)

    assert not np.array_equal(bg[35:45, 35:45], before_dst)


def test_heal_circle_silently_skips_at_image_edge():
    from pylace.tune.bg_editor import heal_circle

    bg = np.full((20, 20), 100, dtype=np.uint8)
    before = bg.copy()
    # Destination too close to the right edge for seamlessClone.
    heal_circle(bg, src_x=5, src_y=5, dst_x=18, dst_y=18, radius=3)
    assert np.array_equal(bg, before)
