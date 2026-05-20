"""Smoke tests for ``pylace.review.review_panel``.

These tests exercise the panel's data-binding logic (table population,
verdict application, autosave) without actually showing the window.
Run with QT_QPA_PLATFORM=offscreen so they work on CI without a display.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PyQt6 = pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from pylace.review.merge import ReviewEvent  # noqa: E402
from pylace.review.review_panel import ReviewPanel  # noqa: E402
from pylace.review.verdicts import Verdict, read_verdicts  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    """Create one QApplication for the whole test module."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    yield app


def _two_events() -> list[ReviewEvent]:
    return [
        ReviewEvent(
            event_id="100-0-1", frame_start=100, frame_end=102,
            animal_a=0, animal_b=1,
            sources=("contact", "audit"),
            audit_cost_kalman_before=10.0, audit_cost_kalman_after=4.0,
            audit_committed=True, audit_permutation=(1, 0),
        ),
        ReviewEvent(
            event_id="200-0-1", frame_start=200, frame_end=200,
            animal_a=0, animal_b=1,
            sources=("contact",),
        ),
    ]


def test_panel_populates_one_row_per_event(qapp, tmp_path: Path):
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
    )
    assert panel._table.rowCount() == 2
    assert panel._table.item(0, 0).text() == "100"
    assert panel._table.item(1, 0).text() == "200"
    assert panel._table.item(0, 1).text() == "0,1"


def test_apply_verdict_writes_to_disk(qapp, tmp_path: Path):
    events = _two_events()
    out = tmp_path / "v.csv"
    panel = ReviewPanel(events, verdicts_path=out, reviewer="unittest")
    panel._table.selectRow(0)
    panel._apply_verdict(Verdict.MOUNT)

    assert out.exists()
    loaded = read_verdicts(out)
    assert "100-0-1" in loaded
    assert loaded["100-0-1"].verdict is Verdict.MOUNT
    assert loaded["100-0-1"].reviewer == "unittest"


def test_apply_verdict_advances_selection(qapp, tmp_path: Path):
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
    )
    panel._table.selectRow(0)
    panel._apply_verdict(Verdict.REJECT_SWAP)
    assert panel._current_row() == 1


def test_apply_verdict_emits_frameJumpRequested_on_select(qapp, tmp_path: Path):
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
    )
    seen: list[int] = []
    panel.frameJumpRequested.connect(seen.append)
    panel._table.selectRow(1)
    assert seen == [200]


def test_existing_verdicts_round_trip_through_panel(qapp, tmp_path: Path):
    """A verdict already attached to a ReviewEvent should appear in the queue."""
    events = _two_events()
    events[0].verdict = Verdict.MOUNT
    events[0].reviewer = "older-bart"
    events[0].note = "two flies in copula"
    panel = ReviewPanel(
        events=events,
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
    )
    # Status badge shows 'mount'.
    assert panel._table.item(0, 3).text() == "mount"
    # And re-saving still preserves the existing record.
    panel._table.selectRow(0)
    panel._apply_verdict(Verdict.ACCEPT_SWAP)
    loaded = read_verdicts(tmp_path / "v.csv")
    assert loaded["100-0-1"].verdict is Verdict.ACCEPT_SWAP
    assert loaded["100-0-1"].reviewer == "unittest"


def test_note_field_is_persisted_with_verdict(qapp, tmp_path: Path):
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
    )
    panel._table.selectRow(0)
    panel._ed_note.setText("looked like a mount")
    panel._apply_verdict(Verdict.MOUNT)
    loaded = read_verdicts(tmp_path / "v.csv")
    assert loaded["100-0-1"].note == "looked like a mount"
