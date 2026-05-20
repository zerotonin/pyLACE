"""Round-trip + helper tests for ``pylace.review.verdicts``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pylace.review.verdicts import (
    VERDICTS_SUFFIX,
    Verdict,
    VerdictRecord,
    default_verdicts_path,
    make_event_id,
    now_iso,
    read_verdicts,
    upsert_verdict,
    write_verdicts,
)


def _make_record(
    frame_start: int = 100,
    animal_a: int = 0,
    animal_b: int = 1,
    verdict: Verdict = Verdict.MOUNT,
) -> VerdictRecord:
    return VerdictRecord(
        event_id=make_event_id(frame_start, animal_a, animal_b),
        frame_start=frame_start,
        frame_end=frame_start + 5,
        animal_a=animal_a,
        animal_b=animal_b,
        verdict=verdict,
        source="audit,contact",
        reviewer="bart",
        timestamp_iso=now_iso(),
        note="manual review",
    )


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────


def test_make_event_id_canonicalises_pair_order():
    assert make_event_id(100, 0, 1) == make_event_id(100, 1, 0)


def test_make_event_id_distinct_for_different_frames():
    assert make_event_id(100, 0, 1) != make_event_id(101, 0, 1)


def test_default_verdicts_path_strips_audited_suffix(tmp_path: Path):
    audited = tmp_path / "vid.mp4.pylace_audited.csv"
    out = default_verdicts_path(audited)
    assert out.name == "vid.mp4" + VERDICTS_SUFFIX


def test_default_verdicts_path_strips_trajectory_suffix(tmp_path: Path):
    traj = tmp_path / "vid.mp4.pylace_trajectory.csv"
    out = default_verdicts_path(traj)
    assert out.name == "vid.mp4" + VERDICTS_SUFFIX


def test_verdict_from_str_rejects_garbage():
    with pytest.raises(ValueError, match="Unknown verdict"):
        Verdict.from_str("nope")


# ─────────────────────────────────────────────────────────────────
#  Round-trip
# ─────────────────────────────────────────────────────────────────


def test_read_missing_file_returns_empty_dict(tmp_path: Path):
    assert read_verdicts(tmp_path / "nope.csv") == {}


def test_write_and_read_round_trips_all_fields(tmp_path: Path):
    r = _make_record()
    p = tmp_path / "v.csv"
    write_verdicts({r.event_id: r}, p)

    loaded = read_verdicts(p)
    assert set(loaded) == {r.event_id}
    got = loaded[r.event_id]
    assert got.verdict is Verdict.MOUNT
    assert got.frame_start == r.frame_start
    assert got.frame_end == r.frame_end
    assert got.animal_a == r.animal_a
    assert got.animal_b == r.animal_b
    assert got.source == r.source
    assert got.reviewer == r.reviewer
    assert got.note == r.note


def test_write_sorts_rows_by_frame_start(tmp_path: Path):
    records = [
        _make_record(frame_start=200),
        _make_record(frame_start=50),
        _make_record(frame_start=125),
    ]
    p = tmp_path / "v.csv"
    write_verdicts(records, p)
    df = pd.read_csv(p)
    assert list(df["frame_start"]) == sorted([r.frame_start for r in records])


def test_write_is_atomic_via_tmp_rename(tmp_path: Path):
    """A pre-existing tmp file from a crash must not interfere with the write."""
    p = tmp_path / "v.csv"
    leftover = p.with_suffix(p.suffix + ".tmp")
    leftover.write_text("crash debris")
    write_verdicts([_make_record()], p)
    assert p.exists()
    assert not leftover.exists(), "tmp file should be gone after replace"


def test_upsert_replaces_existing_event(tmp_path: Path):
    a = _make_record(verdict=Verdict.REJECT_SWAP)
    b = VerdictRecord(
        event_id=a.event_id,
        frame_start=a.frame_start,
        frame_end=a.frame_end,
        animal_a=a.animal_a,
        animal_b=a.animal_b,
        verdict=Verdict.ACCEPT_SWAP,
        source="audit",
        reviewer="bart",
        timestamp_iso=now_iso(),
        note="changed my mind",
    )
    records: dict[str, VerdictRecord] = {}
    upsert_verdict(records, a)
    upsert_verdict(records, b)
    assert len(records) == 1
    assert records[a.event_id].verdict is Verdict.ACCEPT_SWAP


def test_read_rejects_truncated_file(tmp_path: Path):
    p = tmp_path / "v.csv"
    p.write_text("event_id,frame_start\nfoo,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        read_verdicts(p)


def test_empty_string_cells_round_trip_as_empty(tmp_path: Path):
    """note='' should come back as '' not 'nan'."""
    r = VerdictRecord(
        event_id="100-0-1",
        frame_start=100, frame_end=100,
        animal_a=0, animal_b=1,
        verdict=Verdict.UNKNOWN,
        source="",
        reviewer="",
        timestamp_iso="",
        note="",
    )
    p = tmp_path / "v.csv"
    write_verdicts([r], p)
    loaded = read_verdicts(p)
    got = loaded[r.event_id]
    assert got.source == ""
    assert got.note == ""
    assert got.reviewer == ""
