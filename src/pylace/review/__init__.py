# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — review                                                 ║
# ║  « human-in-the-loop swap review: candidates, verdicts, GUI »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Three-pass loop: candidate detection → human review → audit     ║
# ║  re-tag honouring verdicts. See review.verdicts for the file     ║
# ║  contract that connects all three.                               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Swap-review subpackage."""

from pylace.review.verdicts import (
    Verdict,
    VerdictRecord,
    VERDICTS_SUFFIX,
    decode_permutation,
    default_verdicts_path,
    encode_permutation,
    make_event_id,
    now_iso,
    read_verdicts,
    upsert_verdict,
    write_verdicts,
)

__all__ = [
    "Verdict",
    "VerdictRecord",
    "VERDICTS_SUFFIX",
    "decode_permutation",
    "default_verdicts_path",
    "encode_permutation",
    "make_event_id",
    "now_iso",
    "read_verdicts",
    "upsert_verdict",
    "write_verdicts",
]
