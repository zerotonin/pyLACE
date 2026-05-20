# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — review.merge                                           ║
# ║  « merge candidates + audit log + verdicts → review queue »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The GUI shows one row per swap-candidate event. Three sources   ║
# ║  contribute to that list:                                        ║
# ║    1. candidates.csv  — cheap detector (contact / jump)          ║
# ║    2. audit_swaps.csv — the audit's per-event log, with costs    ║
# ║    3. verdicts.csv    — already-made human calls                 ║
# ║  This module joins them by ``event_id`` and returns a list of    ║
# ║  :class:`ReviewEvent` ordered by ``frame_start``. Each event     ║
# ║  carries the union of detector signals plus, if present, the    ║
# ║  audit's cost breakdown and the existing human verdict.          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Merge candidates ∪ audit log ∪ verdicts into a unified review queue."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pylace.review.candidates import Candidate, read_candidates
from pylace.review.verdicts import Verdict, VerdictRecord, read_verdicts

SOURCE_AUDIT = "audit"


@dataclass
class ReviewEvent:
    """One event in the human-review queue.

    Constructed by :func:`merge_review_events`; mutable so the GUI can
    update fields (e.g. attach a fresh verdict) without rebuilding.
    """

    event_id: str
    frame_start: int
    frame_end: int
    animal_a: int
    animal_b: int
    sources: tuple[str, ...] = field(default_factory=tuple)

    # Candidate-detector signals
    min_distance_px: float | None = None
    max_jump_px: float | None = None
    n_frames: int = 0

    # Audit signals (None if the audit did not visit this event)
    audit_cost_before: float | None = None
    audit_cost_after: float | None = None
    audit_cost_kalman_before: float | None = None
    audit_cost_kalman_after: float | None = None
    audit_cost_appearance_before: float | None = None
    audit_cost_appearance_after: float | None = None
    audit_committed: bool = False           # did the audit decide to swap?
    audit_verdict: str = ""                 # '', 'accept_swap', 'mount', etc.
    audit_permutation: tuple[int, ...] | None = None

    # Human verdict (None = no row in verdicts.csv yet)
    verdict: Verdict | None = None
    reviewer: str = ""
    note: str = ""
    timestamp_iso: str = ""

    @property
    def status(self) -> str:
        """One-word status for the GUI queue badge."""
        if self.verdict is not None:
            return self.verdict.value
        if self.audit_committed:
            return "auto_swap"
        return "pending"


def merge_review_events(
    *,
    candidates: list[Candidate] | None = None,
    audit_log: pd.DataFrame | None = None,
    verdicts: dict[str, VerdictRecord] | None = None,
) -> list[ReviewEvent]:
    """Join candidate, audit, and verdict rows by ``event_id``.

    Sources are unioned per event (e.g. ``"contact,jump,audit"``). Any
    audit event without a matching candidate creates a candidate-less
    review event so the GUI can still surface it; same for orphaned
    verdicts. Output is sorted by ``(frame_start, event_id)``.
    """
    by_id: dict[str, ReviewEvent] = {}

    for c in candidates or []:
        if c.event_id in by_id:
            ev = by_id[c.event_id]
            ev.sources = _union_sources(ev.sources, c.source)
            ev.min_distance_px = _safe_min(ev.min_distance_px, c.min_distance_px)
            ev.max_jump_px = _safe_max(ev.max_jump_px, c.max_jump_px)
            ev.n_frames = max(ev.n_frames, c.n_frames)
        else:
            by_id[c.event_id] = ReviewEvent(
                event_id=c.event_id,
                frame_start=c.frame_start,
                frame_end=c.frame_end,
                animal_a=c.animal_a,
                animal_b=c.animal_b,
                sources=_split_sources(c.source),
                min_distance_px=_finite_or_none(c.min_distance_px),
                max_jump_px=_finite_or_none(c.max_jump_px),
                n_frames=c.n_frames,
            )

    if audit_log is not None and not audit_log.empty:
        for _, row in audit_log.iterrows():
            event_id = _str_or_empty(row.get("event_id", ""))
            if not event_id:
                continue
            frame_start = _int_or_default(row.get("frame_start"), -1)
            frame_end = _int_or_default(row.get("frame_end"), frame_start)
            audit_verdict = _str_or_empty(row.get("verdict", ""))
            perm = row.get("permutation")
            if isinstance(perm, str):
                perm = _parse_perm(perm)
            elif isinstance(perm, list):
                perm = tuple(int(x) for x in perm)
            else:
                perm = None
            audit_committed = audit_verdict != Verdict.MOUNT.value

            ev = by_id.get(event_id)
            if ev is None:
                a, b = _extract_pair_from_event_id(event_id)
                ev = ReviewEvent(
                    event_id=event_id,
                    frame_start=frame_start if frame_start >= 0 else 0,
                    frame_end=frame_end if frame_end >= 0 else frame_start,
                    animal_a=a, animal_b=b,
                )
                by_id[event_id] = ev
            ev.sources = _union_sources(ev.sources, SOURCE_AUDIT)
            ev.audit_cost_before = _float_or_none(row.get("cost_before"))
            ev.audit_cost_after = _float_or_none(row.get("cost_after"))
            ev.audit_cost_kalman_before = _float_or_none(row.get("cost_kalman_before"))
            ev.audit_cost_kalman_after = _float_or_none(row.get("cost_kalman_after"))
            ev.audit_cost_appearance_before = _float_or_none(row.get("cost_appearance_before"))
            ev.audit_cost_appearance_after = _float_or_none(row.get("cost_appearance_after"))
            ev.audit_committed = audit_committed
            ev.audit_verdict = audit_verdict
            ev.audit_permutation = perm

    for vr in (verdicts or {}).values():
        ev = by_id.get(vr.event_id)
        if ev is None:
            ev = ReviewEvent(
                event_id=vr.event_id,
                frame_start=vr.frame_start,
                frame_end=vr.frame_end,
                animal_a=vr.animal_a,
                animal_b=vr.animal_b,
            )
            by_id[vr.event_id] = ev
        ev.verdict = vr.verdict
        ev.reviewer = vr.reviewer
        ev.note = vr.note
        ev.timestamp_iso = vr.timestamp_iso

    out = list(by_id.values())
    out.sort(key=lambda e: (e.frame_start, e.event_id))
    return out


def load_review_events(
    *,
    candidates_path: Path | None = None,
    audit_log_path: Path | None = None,
    verdicts_path: Path | None = None,
) -> list[ReviewEvent]:
    """Convenience: read all three files (any may be missing) and merge."""
    candidates = read_candidates(candidates_path) if candidates_path else []
    audit_log = (
        pd.read_csv(audit_log_path)
        if audit_log_path and Path(audit_log_path).exists()
        else None
    )
    verdicts = read_verdicts(verdicts_path) if verdicts_path else {}
    return merge_review_events(
        candidates=candidates,
        audit_log=audit_log,
        verdicts=verdicts,
    )


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────


def _union_sources(existing: tuple[str, ...], new: str) -> tuple[str, ...]:
    out = list(existing)
    for tok in _split_sources(new):
        if tok and tok not in out:
            out.append(tok)
    return tuple(out)


def _split_sources(s: str) -> tuple[str, ...]:
    return tuple(tok for tok in str(s).split(",") if tok)


def _str_or_empty(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int_or_default(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _safe_min(a: float | None, b: float) -> float | None:
    if not math.isfinite(b):
        return a
    if a is None:
        return float(b)
    return float(min(a, b))


def _safe_max(a: float | None, b: float) -> float | None:
    if not math.isfinite(b):
        return a
    if a is None:
        return float(b)
    return float(max(a, b))


def _parse_perm(s: str) -> tuple[int, ...] | None:
    """Parse '[0, 1, 2]' or '(0, 1, 2)' into a tuple of ints; '' → None."""
    s = s.strip()
    if not s:
        return None
    s = s.strip("[](){} ")
    if not s:
        return None
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        return None


def _extract_pair_from_event_id(event_id: str) -> tuple[int, int]:
    """``'100-0-1'`` → ``(0, 1)``; on malformed input returns ``(0, 1)``."""
    parts = event_id.split("-")
    if len(parts) >= 3:
        try:
            return int(parts[-2]), int(parts[-1])
        except ValueError:
            pass
    return 0, 1


__all__ = [
    "ReviewEvent",
    "SOURCE_AUDIT",
    "load_review_events",
    "merge_review_events",
]
