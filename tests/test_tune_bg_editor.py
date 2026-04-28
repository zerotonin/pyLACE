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
