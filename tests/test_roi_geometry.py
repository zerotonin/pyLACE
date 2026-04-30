# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tests.test_roi_geometry                                ║
# ║  « ROI / ROISet validation and basic accessors »                 ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for ROI / ROISet dataclasses."""

from __future__ import annotations

import pytest

from pylace.annotator.geometry import Circle, Polygon, Rectangle
from pylace.roi.geometry import ROI, ROISet


def test_roi_defaults_to_add_op():
    roi = ROI(shape=Circle(10.0, 10.0, 5.0))
    assert roi.operation == "add"
    assert roi.label == ""


def test_roi_rejects_unknown_operation():
    with pytest.raises(ValueError):
        ROI(shape=Circle(0, 0, 1), operation="invert")


def test_roi_set_defaults_to_merge_mode():
    rs = ROISet()
    assert rs.mode == "merge"
    assert rs.is_empty()


def test_roi_set_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ROISet(mode="overlay")


def test_roi_set_add_and_remove_at():
    rs = ROISet()
    rs.add(ROI(shape=Circle(0, 0, 1), label="a"))
    rs.add(ROI(shape=Rectangle.from_two_points((0, 0), (1, 1)), label="b"))
    rs.add(ROI(shape=Polygon([(0, 0), (1, 0), (1, 1)]), label="c"))
    assert [r.label for r in rs.rois] == ["a", "b", "c"]

    rs.remove_at(1)
    assert [r.label for r in rs.rois] == ["a", "c"]

    rs.remove_at(99)  # out-of-range is a no-op
    assert len(rs.rois) == 2


def test_roi_set_with_only_freehand_is_not_empty():
    import numpy as np

    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    rs = ROISet(freehand_mask=mask)
    assert not rs.is_empty()
    assert rs.has_freehand_mask()


def test_roi_set_with_blank_freehand_mask_is_empty():
    import numpy as np

    rs = ROISet(freehand_mask=np.zeros((20, 20), dtype=np.uint8))
    assert rs.is_empty()
    assert not rs.has_freehand_mask()
