# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — review.review_panel                                    ║
# ║  « swap-review dock: queue + details + verdict buttons »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Lives as a QDockWidget on the side of pylace-explore. Three     ║
# ║  zones, top to bottom:                                           ║
# ║    1. Event queue table (frame, animals, source, status, costs)  ║
# ║    2. Details — audit cost breakdown, source tokens, note field  ║
# ║    3. Verdict buttons + a / r / m / u keyboard shortcuts         ║
# ║                                                                  ║
# ║  Verdicts autosave on every button press: the in-memory dict is  ║
# ║  upserted and the verdicts.csv is atomically rewritten via       ║
# ║  pylace.review.verdicts.write_verdicts. The panel emits          ║
# ║  ``frameJumpRequested(int)`` so the host window can drive its    ║
# ║  scrubber without this widget needing a direct reference.        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Qt dock widget for human-in-the-loop swap review."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from pylace.review.merge import ReviewEvent
from pylace.review.verdicts import (
    Verdict,
    VerdictRecord,
    make_event_id,
    now_iso,
    upsert_verdict,
    write_verdicts,
)


_STATUS_COLOURS: dict[str, str] = {
    "pending":      "#888888",
    "auto_swap":    "#0072B2",   # blue — Wong
    Verdict.ACCEPT_SWAP.value: "#009E73",   # bluish green
    Verdict.REJECT_SWAP.value: "#D55E00",   # vermilion
    Verdict.MOUNT.value:       "#E69F00",   # orange
    Verdict.UNKNOWN.value:     "#CC79A7",   # reddish purple
}

DEFAULT_ZOOM_FRAMES = 200


class ReviewPanel(QtWidgets.QDockWidget):
    """Sidebar panel for swap-candidate review."""

    frameJumpRequested = QtCore.pyqtSignal(int)
    zoomWindowRequested = QtCore.pyqtSignal(int, int)   # (frame_lo, frame_hi)
    verdictWritten = QtCore.pyqtSignal()

    def __init__(
        self,
        events: list[ReviewEvent],
        verdicts_path: Path,
        reviewer: str | None = None,
        track_ids: list[int] | None = None,
        max_track_id: int | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("Swap review", parent)
        self.setObjectName("ReviewPanel")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self._events: list[ReviewEvent] = list(events)
        self._verdicts_path = Path(verdicts_path)
        self._reviewer = (
            reviewer
            if reviewer
            else os.environ.get("USER", "") or "anonymous"
        )
        seen_ids = sorted({
            int(a) for ev in events for a in (ev.animal_a, ev.animal_b)
        })
        if track_ids is not None:
            self._track_ids: list[int] = sorted(int(t) for t in track_ids)
        elif seen_ids:
            self._track_ids = seen_ids
        else:
            self._track_ids = [0, 1]
        if max_track_id is None:
            max_track_id = (
                max(self._track_ids) if self._track_ids else 9
            )
        self._track_id_max = max(int(max_track_id), 1)
        self._zoom_window_frames = DEFAULT_ZOOM_FRAMES
        self._perm_spinboxes: list[QtWidgets.QSpinBox] = []

        self._verdict_dict: dict[str, VerdictRecord] = {
            ev.event_id: VerdictRecord(
                event_id=ev.event_id,
                frame_start=ev.frame_start,
                frame_end=ev.frame_end,
                animal_a=ev.animal_a,
                animal_b=ev.animal_b,
                verdict=ev.verdict,
                source=",".join(ev.sources),
                reviewer=ev.reviewer,
                timestamp_iso=ev.timestamp_iso,
                note=ev.note,
            )
            for ev in self._events
            if ev.verdict is not None
        }

        self._build_ui()
        self._populate_table()
        self._wire_shortcuts()

    # ── Layout ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(4, 4, 4, 4)

        top_row = QtWidgets.QHBoxLayout()
        self._lbl_summary = QtWidgets.QLabel("", wrap)
        top_row.addWidget(self._lbl_summary, stretch=1)
        top_row.addWidget(QtWidgets.QLabel("Zoom window (frames):", wrap))
        self._sb_zoom = QtWidgets.QSpinBox(wrap)
        self._sb_zoom.setRange(20, 20000)
        self._sb_zoom.setSingleStep(20)
        self._sb_zoom.setValue(self._zoom_window_frames)
        self._sb_zoom.valueChanged.connect(self._on_zoom_changed)
        top_row.addWidget(self._sb_zoom)
        v.addLayout(top_row)

        self._table = QtWidgets.QTableWidget(wrap)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["frame", "animals", "source", "status", "ΔK"],
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows,
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection,
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        )
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        v.addWidget(self._table, stretch=2)

        details_box = QtWidgets.QGroupBox("Details (verdict applies to these values)", wrap)
        form = QtWidgets.QFormLayout(details_box)

        self._lbl_costs = QtWidgets.QLabel("(select an event)", details_box)
        self._lbl_costs.setTextFormat(Qt.TextFormat.PlainText)
        self._lbl_costs.setStyleSheet(
            "QLabel { font-family: monospace; font-size: 10pt; }",
        )
        form.addRow("Costs", self._lbl_costs)
        self._lbl_source = QtWidgets.QLabel("", details_box)
        form.addRow("Sources", self._lbl_source)

        frame_row = QtWidgets.QHBoxLayout()
        self._sb_frame_start = QtWidgets.QSpinBox(details_box)
        self._sb_frame_start.setRange(0, 100_000_000)
        self._sb_frame_start.setToolTip(
            "Start frame of the event. Edit before pressing a verdict "
            "to override the auto-detected block start.",
        )
        self._sb_frame_end = QtWidgets.QSpinBox(details_box)
        self._sb_frame_end.setRange(0, 100_000_000)
        self._sb_frame_end.setToolTip("End frame of the event.")
        frame_row.addWidget(QtWidgets.QLabel("start:", details_box))
        frame_row.addWidget(self._sb_frame_start)
        frame_row.addSpacing(8)
        frame_row.addWidget(QtWidgets.QLabel("end:", details_box))
        frame_row.addWidget(self._sb_frame_end)
        frame_row.addStretch(1)
        form.addRow("Frames", frame_row)

        perm_row = QtWidgets.QHBoxLayout()
        perm_row.setSpacing(8)
        perm_row.addWidget(QtWidgets.QLabel("track → new id:", details_box))
        self._perm_spinboxes = []
        for tid in self._track_ids:
            sb = QtWidgets.QSpinBox(details_box)
            sb.setRange(0, max(99, self._track_id_max + 5))
            sb.setValue(tid)
            sb.setPrefix(f"{tid}→")
            sb.setToolTip(
                f"What track {tid} should be relabelled to under "
                "'Accept swap'. Default = identity (no relabel). "
                "Edit several at once to express a 3-way cycle.",
            )
            perm_row.addWidget(sb)
            self._perm_spinboxes.append(sb)
        btn_reset = QtWidgets.QPushButton("Reset", details_box)
        btn_reset.setToolTip("Restore identity (no relabel).")
        btn_reset.clicked.connect(self._reset_perm_to_identity)
        perm_row.addWidget(btn_reset)
        perm_row.addStretch(1)
        form.addRow("Permutation", perm_row)

        self._ed_note = QtWidgets.QLineEdit(details_box)
        self._ed_note.setPlaceholderText(
            "Optional note — saved with the verdict",
        )
        form.addRow("Note", self._ed_note)
        v.addWidget(details_box, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_accept = self._mk_btn(
            "Accept swap [a]", Verdict.ACCEPT_SWAP, "#009E73",
        )
        self._btn_reject = self._mk_btn(
            "Reject [r]", Verdict.REJECT_SWAP, "#D55E00",
        )
        self._btn_mount = self._mk_btn(
            "Mount [m]", Verdict.MOUNT, "#E69F00",
        )
        self._btn_unknown = self._mk_btn(
            "Unknown [u]", Verdict.UNKNOWN, "#CC79A7",
        )
        for b in (self._btn_accept, self._btn_reject,
                  self._btn_mount, self._btn_unknown):
            btn_row.addWidget(b)
        v.addLayout(btn_row)

        nav_row = QtWidgets.QHBoxLayout()
        self._btn_prev = QtWidgets.QPushButton("◀ Prev [p]", wrap)
        self._btn_next = QtWidgets.QPushButton("Next ▶ [n]", wrap)
        self._btn_prev.clicked.connect(self._select_prev)
        self._btn_next.clicked.connect(self._select_next)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        nav_row.addStretch(1)
        v.addLayout(nav_row)

        self.setWidget(wrap)

    def _mk_btn(
        self, label: str, kind: Verdict, colour: str,
    ) -> QtWidgets.QPushButton:
        b = QtWidgets.QPushButton(label, self)
        b.setStyleSheet(
            f"QPushButton {{ background-color: {colour}; color: white; "
            f"font-weight: bold; padding: 6px; }}",
        )
        b.clicked.connect(lambda _checked=False, v=kind: self._apply_verdict(v))
        return b

    def _wire_shortcuts(self) -> None:
        for key, kind in (
            ("a", Verdict.ACCEPT_SWAP),
            ("r", Verdict.REJECT_SWAP),
            ("m", Verdict.MOUNT),
            ("u", Verdict.UNKNOWN),
        ):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self.widget())
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda v=kind: self._apply_verdict(v))
        for key, fn in (
            ("n", self._select_next),
            ("p", self._select_prev),
        ):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self.widget())
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # ── Population ────────────────────────────────────────────────────

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._events))
        for row, ev in enumerate(self._events):
            self._set_row(row, ev)
        self._refresh_summary()
        if self._events:
            self._table.selectRow(0)

    def _set_row(self, row: int, ev: ReviewEvent) -> None:
        self._table.setItem(row, 0, _ro_item(f"{ev.frame_start}"))
        self._table.setItem(
            row, 1, _ro_item(f"{ev.animal_a},{ev.animal_b}"),
        )
        self._table.setItem(row, 2, _ro_item(",".join(ev.sources) or "—"))
        status = ev.status
        status_item = _ro_item(status)
        status_item.setForeground(QtGui.QBrush(
            QtGui.QColor(_STATUS_COLOURS.get(status, "#000000")),
        ))
        self._table.setItem(row, 3, status_item)
        delta = ev.audit_cost_kalman_after
        kalman_before = ev.audit_cost_kalman_before
        if delta is not None and kalman_before is not None:
            self._table.setItem(
                row, 4,
                _ro_item(f"{kalman_before:.2f} → {delta:.2f}"),
            )
        else:
            self._table.setItem(row, 4, _ro_item("—"))

    def _refresh_summary(self) -> None:
        n = len(self._events)
        n_pending = sum(1 for e in self._events if e.verdict is None)
        n_done = n - n_pending
        self._lbl_summary.setText(
            f"<b>{n}</b> events  ·  <b>{n_done}</b> reviewed  "
            f"·  <b>{n_pending}</b> pending  ·  "
            f"reviewer: <i>{self._reviewer}</i>",
        )

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_row_selected(self) -> None:
        row = self._current_row()
        if row is None:
            return
        ev = self._events[row]
        self._populate_details(ev)
        self.frameJumpRequested.emit(ev.frame_start)
        lo, hi = self._zoom_window(ev.frame_start, ev.frame_end)
        self.zoomWindowRequested.emit(lo, hi)

    def _populate_details(self, ev: ReviewEvent) -> None:
        costs_lines = []
        costs_lines.append(
            f"kalman   ident {_fmt(ev.audit_cost_kalman_before):>8}  "
            f"swap {_fmt(ev.audit_cost_kalman_after):>8}",
        )
        costs_lines.append(
            f"appear   ident {_fmt(ev.audit_cost_appearance_before):>8}  "
            f"swap {_fmt(ev.audit_cost_appearance_after):>8}",
        )
        if ev.audit_cost_before is not None:
            costs_lines.append(
                f"total    ident {_fmt(ev.audit_cost_before):>8}  "
                f"swap {_fmt(ev.audit_cost_after):>8}",
            )
        self._lbl_costs.setText("\n".join(costs_lines) or "—")
        self._lbl_source.setText(",".join(ev.sources) or "—")
        for sb, value in (
            (self._sb_frame_start, ev.frame_start),
            (self._sb_frame_end, ev.frame_end),
        ):
            sb.blockSignals(True)
            sb.setValue(int(value))
            sb.blockSignals(False)
        self._seed_perm_from_event(ev)
        self._ed_note.setText(ev.note or "")

    def _seed_perm_from_event(self, ev: ReviewEvent) -> None:
        """Set the permutation spinboxes to a sensible default for ``ev``.

        Priority:
          1. explicit ev.audit_permutation (e.g. from a re-loaded verdict)
          2. pair-swap on the detected (animal_a, animal_b)
          3. identity
        """
        for sb, tid in zip(self._perm_spinboxes, self._track_ids, strict=True):
            sb.blockSignals(True)
            sb.setValue(tid)
            sb.blockSignals(False)
        perm = ev.audit_permutation
        if (
            perm is not None
            and len(perm) == len(self._track_ids)
            and sorted(perm) == sorted(self._track_ids)
        ):
            for sb, val in zip(self._perm_spinboxes, perm, strict=True):
                sb.blockSignals(True)
                sb.setValue(int(val))
                sb.blockSignals(False)
            return
        if ev.animal_a in self._track_ids and ev.animal_b in self._track_ids:
            ia = self._track_ids.index(ev.animal_a)
            ib = self._track_ids.index(ev.animal_b)
            self._perm_spinboxes[ia].blockSignals(True)
            self._perm_spinboxes[ia].setValue(ev.animal_b)
            self._perm_spinboxes[ia].blockSignals(False)
            self._perm_spinboxes[ib].blockSignals(True)
            self._perm_spinboxes[ib].setValue(ev.animal_a)
            self._perm_spinboxes[ib].blockSignals(False)

    def _reset_perm_to_identity(self) -> None:
        for sb, tid in zip(self._perm_spinboxes, self._track_ids, strict=True):
            sb.blockSignals(True)
            sb.setValue(tid)
            sb.blockSignals(False)

    def _read_permutation(self) -> tuple[int, ...]:
        return tuple(int(sb.value()) for sb in self._perm_spinboxes)

    def _zoom_window(self, frame_start: int, frame_end: int) -> tuple[int, int]:
        """Return ``(lo, hi)`` frames covering the event ± half the window."""
        centre = (int(frame_start) + int(frame_end)) // 2
        half = max(1, self._zoom_window_frames // 2)
        return centre - half, centre + half

    def _on_zoom_changed(self, value: int) -> None:
        self._zoom_window_frames = int(value)
        row = self._current_row()
        if row is None:
            return
        ev = self._events[row]
        lo, hi = self._zoom_window(ev.frame_start, ev.frame_end)
        self.zoomWindowRequested.emit(lo, hi)

    def _apply_verdict(self, kind: Verdict) -> None:
        row = self._current_row()
        if row is None:
            return
        ev = self._events[row]

        frame_start = int(self._sb_frame_start.value())
        frame_end = int(self._sb_frame_end.value())
        if frame_end < frame_start:
            frame_start, frame_end = frame_end, frame_start

        perm = self._read_permutation()
        permutation_field: tuple[int, ...] | None = None
        animal_a, animal_b = ev.animal_a, ev.animal_b

        if kind is Verdict.ACCEPT_SWAP:
            # accept_swap only makes sense if the user changed something.
            # Validate the permutation is a true permutation of the track ids.
            if sorted(perm) != sorted(self._track_ids):
                QtWidgets.QMessageBox.warning(
                    self.widget(), "Invalid permutation",
                    "The 'track → new id' spinboxes must be a permutation "
                    "of the existing track ids (no duplicates, no new ids). "
                    "Hit Reset to start over.",
                )
                return
            changed = [
                (tid, new) for tid, new in zip(
                    self._track_ids, perm, strict=True,
                )
                if int(tid) != int(new)
            ]
            if not changed:
                QtWidgets.QMessageBox.warning(
                    self.widget(), "Identity permutation",
                    "The permutation is identity — no relabel would happen. "
                    "Use 'Reject' if the event is not a swap.",
                )
                return
            permutation_field = tuple(int(x) for x in perm)
            # Derive a representative pair for the event_id from the
            # first two tracks whose label actually changed.
            animal_a = int(changed[0][0])
            animal_b = (
                int(changed[1][0]) if len(changed) >= 2 else int(changed[0][1])
            )
        else:
            # mount / reject / unknown: animal_a, animal_b come from the
            # spinboxes' first two differing entries if the user edited
            # them, else fall back to the auto-detected pair.
            diffs = [
                tid for tid, new in zip(self._track_ids, perm, strict=True)
                if int(tid) != int(new)
            ]
            if len(diffs) >= 2:
                animal_a = int(diffs[0])
                animal_b = int(diffs[1])

        if animal_a == animal_b:
            QtWidgets.QMessageBox.warning(
                self.widget(), "Invalid animals",
                "animal_a and animal_b must differ — fix the permutation "
                "spinboxes before pressing a verdict.",
            )
            return

        event_id = make_event_id(frame_start, animal_a, animal_b)
        if event_id != ev.event_id and ev.event_id in self._verdict_dict:
            del self._verdict_dict[ev.event_id]

        record = VerdictRecord(
            event_id=event_id,
            frame_start=frame_start,
            frame_end=frame_end,
            animal_a=animal_a,
            animal_b=animal_b,
            verdict=kind,
            source=",".join(ev.sources),
            reviewer=self._reviewer,
            timestamp_iso=now_iso(),
            note=self._ed_note.text().strip(),
            permutation=permutation_field,
        )
        upsert_verdict(self._verdict_dict, record)
        write_verdicts(self._verdict_dict, self._verdicts_path)

        ev.event_id = event_id
        ev.frame_start = frame_start
        ev.frame_end = frame_end
        ev.animal_a = animal_a
        ev.animal_b = animal_b
        ev.verdict = kind
        ev.reviewer = record.reviewer
        ev.timestamp_iso = record.timestamp_iso
        ev.note = record.note
        if permutation_field is not None:
            ev.audit_permutation = permutation_field
        self._set_row(row, ev)
        self._refresh_summary()
        self.verdictWritten.emit()
        self._select_next()

    def _select_prev(self) -> None:
        row = self._current_row()
        if row is None:
            return
        new = max(0, row - 1)
        if new != row:
            self._table.selectRow(new)

    def _select_next(self) -> None:
        row = self._current_row()
        if row is None:
            return
        new = min(self._table.rowCount() - 1, row + 1)
        if new != row:
            self._table.selectRow(new)

    def _current_row(self) -> int | None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        return int(rows[0].row())


def _ro_item(text: str) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


def _fmt(value: float | None) -> str:
    if value is None:
        return "  n/a"
    return f"{value:7.2f}"


__all__ = ["ReviewPanel"]
