# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — review.verdicts                                        ║
# ║  « human verdicts on swap-candidate events, on disk »            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Each verdict is one human call on one candidate swap event:     ║
# ║    accept_swap — yes, do swap these IDs                          ║
# ║    reject_swap — no, leave IDs alone (auto-detector was wrong)   ║
# ║    mount       — the two animals are physically on top of each   ║
# ║                  other; leave IDs alone AND label the audited    ║
# ║                  rows with event_type='mount' so downstream      ║
# ║                  kinematics can drop the window                  ║
# ║    unknown     — looked at it, not sure (treated like no row)    ║
# ║                                                                  ║
# ║  Verdicts live next to the audited CSV in                        ║
# ║  ``<video>.pylace_verdicts.csv``. Round-trip is via pandas;      ║
# ║  writes are atomic (temp file + rename) so an autosave during a  ║
# ║  GUI keystroke never leaves a half-written file on disk.         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Verdict dataclass + on-disk format for swap-review verdicts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable

import pandas as pd

VERDICTS_SUFFIX = ".pylace_verdicts.csv"


class Verdict(str, Enum):
    """The four allowed verdict labels."""

    ACCEPT_SWAP = "accept_swap"
    REJECT_SWAP = "reject_swap"
    MOUNT = "mount"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, value: str) -> "Verdict":
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"Unknown verdict {value!r}; "
                f"expected one of {[v.value for v in cls]}",
            ) from exc


@dataclass(frozen=True)
class VerdictRecord:
    """One reviewer call on one candidate event.

    ``permutation`` is optional and only meaningful for ``accept_swap``
    verdicts on N>2 events: it stores the full per-track relabelling as
    a tuple where ``permutation[i]`` is the new label for the i-th
    sorted track id. ``None`` falls back to a pair-swap on
    ``(animal_a, animal_b)``.
    """

    event_id: str
    frame_start: int
    frame_end: int
    animal_a: int
    animal_b: int
    verdict: Verdict
    source: str = ""
    reviewer: str = ""
    timestamp_iso: str = ""
    note: str = ""
    permutation: tuple[int, ...] | None = None


_COLUMNS: tuple[str, ...] = (
    "event_id", "frame_start", "frame_end",
    "animal_a", "animal_b",
    "verdict", "source", "reviewer", "timestamp_iso", "note",
    "permutation",
)


def encode_permutation(perm: tuple[int, ...] | None) -> str:
    """Encode ``perm`` as a CSV-safe comma-separated string. ``None`` → ``''``."""
    if perm is None:
        return ""
    return ",".join(str(int(x)) for x in perm)


def decode_permutation(value: object) -> tuple[int, ...] | None:
    """Inverse of :func:`encode_permutation`; tolerant of NaN / empty cells."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if not s:
        return None
    try:
        return tuple(int(tok.strip()) for tok in s.split(",") if tok.strip())
    except ValueError:
        return None


def make_event_id(frame_start: int, animal_a: int, animal_b: int) -> str:
    """Deterministic id; pair order is canonicalised so (a,b) == (b,a)."""
    a, b = sorted((int(animal_a), int(animal_b)))
    return f"{int(frame_start)}-{a}-{b}"


def now_iso() -> str:
    """UTC timestamp at 1 s resolution, suitable for the verdicts CSV."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def default_verdicts_path(audited_csv: Path) -> Path:
    """Conventional verdicts CSV next to the audited / trajectory CSV."""
    from pylace.posthoc.io import trajectory_stem  # lazy: avoid circular import via posthoc.__init__
    audited_csv = Path(audited_csv)
    return audited_csv.with_name(trajectory_stem(audited_csv) + VERDICTS_SUFFIX)


def read_verdicts(path: Path) -> dict[str, VerdictRecord]:
    """Load verdicts keyed by event_id. Missing file → empty dict.

    Files written before the ``permutation`` column existed are still
    accepted: the column defaults to ``None`` (pair-swap fallback).
    """
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = [c for c in _COLUMNS if c != "permutation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Verdicts CSV {path} missing columns: {missing}",
        )
    has_perm = "permutation" in df.columns
    records: dict[str, VerdictRecord] = {}
    for _, row in df.iterrows():
        rec = VerdictRecord(
            event_id=str(row["event_id"]),
            frame_start=int(row["frame_start"]),
            frame_end=int(row["frame_end"]),
            animal_a=int(row["animal_a"]),
            animal_b=int(row["animal_b"]),
            verdict=Verdict.from_str(str(row["verdict"])),
            source=_str_or_empty(row["source"]),
            reviewer=_str_or_empty(row["reviewer"]),
            timestamp_iso=_str_or_empty(row["timestamp_iso"]),
            note=_str_or_empty(row["note"]),
            permutation=decode_permutation(row["permutation"]) if has_perm else None,
        )
        records[rec.event_id] = rec
    return records


def write_verdicts(
    records: Iterable[VerdictRecord] | dict[str, VerdictRecord],
    path: Path,
) -> None:
    """Atomically replace the verdicts CSV at ``path``.

    Writes to ``<path>.tmp`` then ``os.replace`` so a crash mid-write
    cannot corrupt an existing file. Sort order is ``frame_start`` then
    ``event_id`` so diffs stay readable.
    """
    if isinstance(records, dict):
        records = list(records.values())
    else:
        records = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "event_id": r.event_id,
                "frame_start": int(r.frame_start),
                "frame_end": int(r.frame_end),
                "animal_a": int(r.animal_a),
                "animal_b": int(r.animal_b),
                "verdict": r.verdict.value,
                "source": r.source,
                "reviewer": r.reviewer,
                "timestamp_iso": r.timestamp_iso,
                "note": r.note,
                "permutation": encode_permutation(r.permutation),
            }
            for r in records
        ],
        columns=list(_COLUMNS),
    )
    if not df.empty:
        df = df.sort_values(["frame_start", "event_id"]).reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def upsert_verdict(
    records: dict[str, VerdictRecord],
    new: VerdictRecord,
) -> dict[str, VerdictRecord]:
    """Insert / replace ``new`` by ``event_id``; returns the same dict for chaining."""
    records[new.event_id] = new
    return records


def _str_or_empty(value: object) -> str:
    """pandas turns blank cells into NaN; collapse those back to ``''``."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


__all__ = [
    "VERDICTS_SUFFIX",
    "Verdict",
    "VerdictRecord",
    "decode_permutation",
    "default_verdicts_path",
    "encode_permutation",
    "make_event_id",
    "now_iso",
    "read_verdicts",
    "upsert_verdict",
    "write_verdicts",
]
