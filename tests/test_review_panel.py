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


# ─────────────────────────────────────────────────────────────────
#  Editable spinboxes + zoom-window signal
# ─────────────────────────────────────────────────────────────────


def test_spinboxes_seed_from_selected_event(qapp, tmp_path: Path):
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
        track_ids=[0, 1],
    )
    panel._table.selectRow(0)
    assert panel._sb_frame_start.value() == 100
    assert panel._sb_frame_end.value() == 102
    # Permutation table is seeded with the pair-swap (0 ↔ 1).
    assert panel._read_permutation() == (1, 0)


def test_editing_frame_and_pair_changes_event_id_on_save(qapp, tmp_path: Path):
    """Editing frame_start or the involved pair must rekey the verdict."""
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
        track_ids=[0, 1, 2],
    )
    panel._table.selectRow(0)
    panel._sb_frame_start.setValue(95)
    panel._sb_frame_end.setValue(110)
    # Express a (0 ↔ 2) swap by editing the permutation spinboxes.
    panel._perm_spinboxes[0].setValue(2)
    panel._perm_spinboxes[1].setValue(1)
    panel._perm_spinboxes[2].setValue(0)
    panel._apply_verdict(Verdict.ACCEPT_SWAP)

    loaded = read_verdicts(tmp_path / "v.csv")
    assert "95-0-2" in loaded
    assert "100-0-1" not in loaded
    rec = loaded["95-0-2"]
    assert rec.frame_start == 95
    assert rec.frame_end == 110
    assert rec.animal_a == 0
    assert rec.animal_b == 2
    assert rec.permutation == (2, 1, 0)


def test_zoom_window_signal_emitted_on_row_select(qapp, tmp_path: Path):
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
    )
    seen: list[tuple[int, int]] = []
    panel.zoomWindowRequested.connect(lambda lo, hi: seen.append((lo, hi)))
    # The panel starts with row 0 selected; move to row 1 to fire a signal.
    panel._table.selectRow(1)
    assert len(seen) >= 1
    lo, hi = seen[-1]
    # Event 1 frame_start=200, frame_end=200 → centre=200, half-window=100.
    assert hi - lo == 200
    assert lo == 100
    assert hi == 300


def test_zoom_window_spinbox_re_emits(qapp, tmp_path: Path):
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
    )
    panel._table.selectRow(0)
    seen: list[tuple[int, int]] = []
    panel.zoomWindowRequested.connect(lambda lo, hi: seen.append((lo, hi)))
    panel._sb_zoom.setValue(400)
    assert seen, "changing the zoom-window spinbox should emit"
    lo, hi = seen[-1]
    assert hi - lo == 400


# ─────────────────────────────────────────────────────────────────
#  Permutation table (3+ animal support)
# ─────────────────────────────────────────────────────────────────


def test_permutation_table_has_one_spinbox_per_track(qapp, tmp_path: Path):
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
        track_ids=[0, 1, 2],
    )
    assert len(panel._perm_spinboxes) == 3
    panel._table.selectRow(0)
    # Both test events have animal_a=0, animal_b=1 → seeded with the
    # pair-swap (1, 0) on the first two columns, identity on the third.
    assert panel._read_permutation() == (1, 0, 2)


def test_reset_button_restores_identity(qapp, tmp_path: Path):
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
        track_ids=[0, 1, 2],
    )
    panel._table.selectRow(0)
    panel._perm_spinboxes[0].setValue(2)
    panel._perm_spinboxes[2].setValue(0)
    panel._reset_perm_to_identity()
    assert panel._read_permutation() == (0, 1, 2)


def test_accept_swap_rejects_non_permutation(qapp, tmp_path: Path, monkeypatch):
    """Duplicate values in the spinboxes mean it is not a real permutation."""
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
        track_ids=[0, 1, 2],
    )
    panel._table.selectRow(0)
    panel._perm_spinboxes[0].setValue(1)
    panel._perm_spinboxes[1].setValue(1)   # duplicate!
    panel._perm_spinboxes[2].setValue(2)
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QMessageBox.warning", lambda *a, **kw: None,
    )
    panel._apply_verdict(Verdict.ACCEPT_SWAP)
    assert not (tmp_path / "v.csv").exists()


def test_accept_swap_rejects_identity_permutation(qapp, tmp_path: Path, monkeypatch):
    """Identity is meaningless for accept_swap; user should use Reject instead."""
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
        track_ids=[0, 1, 2],
    )
    panel._table.selectRow(0)
    panel._reset_perm_to_identity()  # force identity
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QMessageBox.warning", lambda *a, **kw: None,
    )
    panel._apply_verdict(Verdict.ACCEPT_SWAP)
    assert not (tmp_path / "v.csv").exists()


def test_three_way_cycle_round_trips_through_csv(qapp, tmp_path: Path):
    """Bart's 3-fly scenario: enter a 3-cycle and re-read it."""
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
        track_ids=[0, 1, 2],
    )
    panel._table.selectRow(0)
    panel._perm_spinboxes[0].setValue(1)
    panel._perm_spinboxes[1].setValue(2)
    panel._perm_spinboxes[2].setValue(0)
    panel._apply_verdict(Verdict.ACCEPT_SWAP)

    loaded = read_verdicts(tmp_path / "v.csv")
    # The event_id keys off the first two changed tracks: 0 and 1.
    assert "100-0-1" in loaded
    rec = loaded["100-0-1"]
    assert rec.verdict is Verdict.ACCEPT_SWAP
    assert rec.permutation == (1, 2, 0)


def test_mount_verdict_does_not_record_permutation(qapp, tmp_path: Path):
    """mount uses the spinboxes' changed tracks for animal_a/b, but never
    records a permutation field — that's only for accept_swap."""
    panel = ReviewPanel(
        events=_two_events(),
        verdicts_path=tmp_path / "v.csv",
        reviewer="unittest",
        track_ids=[0, 1, 2],
    )
    panel._table.selectRow(0)
    panel._reset_perm_to_identity()
    # Mark tracks 0 and 2 as the involved pair by editing those spinboxes.
    panel._perm_spinboxes[0].setValue(2)
    panel._perm_spinboxes[2].setValue(0)
    panel._apply_verdict(Verdict.MOUNT)
    loaded = read_verdicts(tmp_path / "v.csv")
    assert "100-0-2" in loaded
    assert loaded["100-0-2"].permutation is None
    assert loaded["100-0-2"].animal_a == 0
    assert loaded["100-0-2"].animal_b == 2
