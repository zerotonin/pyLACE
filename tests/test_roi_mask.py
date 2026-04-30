# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tests.test_roi_mask                                    ║
# ║  « combined ROI mask: union, subtract, first-implicit-add »      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for ``build_combined_mask``."""

from __future__ import annotations

import numpy as np
import pytest

from pylace.annotator.geometry import Circle, Rectangle
from pylace.roi.geometry import ROI, ROISet
from pylace.roi.mask import build_combined_mask


def test_empty_roi_set_returns_all_true():
    mask = build_combined_mask(ROISet(), frame_size=(40, 30))
    assert mask.shape == (30, 40)
    assert mask.dtype == bool
    assert mask.all()


def test_single_add_roi_matches_arena_mask():
    rs = ROISet(rois=[ROI(shape=Circle(20.0, 15.0, 10.0))])
    mask = build_combined_mask(rs, frame_size=(40, 30))
    assert mask[15, 20]
    assert not mask[0, 0]


def test_two_disjoint_adds_union():
    rs = ROISet(rois=[
        ROI(shape=Rectangle.from_two_points((0, 0), (10, 10))),
        ROI(shape=Rectangle.from_two_points((30, 20), (40, 30)),
            operation="add"),
    ])
    mask = build_combined_mask(rs, frame_size=(40, 30))
    assert mask[5, 5]
    assert mask[25, 35]
    assert not mask[15, 20]


def test_subtract_removes_inner_circle():
    rs = ROISet(rois=[
        ROI(shape=Rectangle.from_two_points((0, 0), (40, 30)), operation="add"),
        ROI(shape=Circle(20.0, 15.0, 5.0), operation="subtract"),
    ])
    mask = build_combined_mask(rs, frame_size=(40, 30))
    assert mask[0, 0]
    assert mask[29, 39]
    assert not mask[15, 20]


def test_first_subtract_is_treated_as_add():
    """First ROI is implicitly add regardless of declared operation."""
    rs = ROISet(rois=[
        ROI(shape=Rectangle.from_two_points((0, 0), (10, 10)),
            operation="subtract"),
    ])
    mask = build_combined_mask(rs, frame_size=(40, 30))
    assert mask[5, 5]


def test_split_mode_raises_not_implemented_for_combined():
    rs = ROISet(rois=[ROI(shape=Circle(0, 0, 1))], mode="split")
    with pytest.raises(NotImplementedError):
        build_combined_mask(rs, frame_size=(40, 30))


def test_split_mode_returns_one_mask_per_add_roi():
    from pylace.roi.mask import build_split_masks

    rs = ROISet(
        rois=[
            ROI(shape=Circle(20.0, 15.0, 5.0), label="left"),
            ROI(shape=Rectangle.from_two_points((25, 10), (35, 20)), label="right"),
        ],
        mode="split",
    )
    pairs = build_split_masks(rs, frame_size=(40, 30))
    labels = [label for label, _ in pairs]
    assert labels == ["left", "right"]
    assert pairs[0][1].shape == (30, 40)
    assert pairs[0][1][15, 20]
    assert pairs[1][1][12, 30]


def test_split_mode_skips_subtract_rois():
    from pylace.roi.mask import build_split_masks

    rs = ROISet(
        rois=[
            ROI(shape=Circle(20.0, 15.0, 5.0), label="keep"),
            ROI(
                shape=Rectangle.from_two_points((0, 0), (10, 10)),
                operation="subtract", label="ignore_me",
            ),
        ],
        mode="split",
    )
    pairs = build_split_masks(rs, frame_size=(40, 30))
    assert [label for label, _ in pairs] == ["keep"]


def test_split_mode_default_label_when_blank():
    from pylace.roi.mask import build_split_masks

    rs = ROISet(
        rois=[
            ROI(shape=Circle(20.0, 15.0, 5.0)),
            ROI(shape=Circle(10.0, 10.0, 3.0)),
        ],
        mode="split",
    )
    pairs = build_split_masks(rs, frame_size=(40, 30))
    assert [label for label, _ in pairs] == ["roi_0", "roi_1"]


def test_split_mode_helper_rejects_merge_mode():
    from pylace.roi.mask import build_split_masks

    rs = ROISet(rois=[ROI(shape=Circle(0, 0, 1))], mode="merge")
    with pytest.raises(ValueError):
        build_split_masks(rs, frame_size=(40, 30))


def test_mask_shape_matches_frame_size():
    rs = ROISet(rois=[ROI(shape=Circle(50.0, 50.0, 10.0))])
    mask = build_combined_mask(rs, frame_size=(640, 480))
    assert mask.shape == (480, 640)
    assert mask.dtype == bool
