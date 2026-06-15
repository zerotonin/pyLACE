"""``--help`` smoke tests for every ``console_scripts`` entry point.

These do not exercise any real behaviour — they just guarantee that
each CLI module imports cleanly and its argparse parser is built
successfully. Catches the bulk of 'you broke an entry point' failures
cheaply on every CI run, on every OS.
"""

from __future__ import annotations

import importlib
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


ENTRY_POINTS: list[tuple[str, str]] = [
    ("pylace-annotate",    "pylace.annotator.cli"),
    ("pylace-audit",       "pylace.posthoc.audit_cli"),
    ("pylace-candidates",  "pylace.review.review_cli"),
    ("pylace-clean",       "pylace.posthoc.cli"),
    ("pylace-detect",      "pylace.detect.cli"),
    ("pylace-explore",     "pylace.posthoc.explorer_cli"),
    ("pylace-fingerprint", "pylace.posthoc.fingerprint_cli"),
    ("pylace-inspect",     "pylace.inspect.cli"),
    ("pylace-metrics",     "pylace.posthoc.metrics_cli"),
    ("pylace-roi",         "pylace.roi.cli"),
    ("pylace-tune",        "pylace.tune.cli"),
]


@pytest.mark.parametrize(
    ("entry_point", "module_name"),
    ENTRY_POINTS,
    ids=[ep for ep, _ in ENTRY_POINTS],
)
def test_cli_help_exits_zero(entry_point: str, module_name: str, capsys):
    """``main(["--help"])`` must exit with status 0 for every CLI."""
    module = importlib.import_module(module_name)
    assert hasattr(module, "main"), f"{module_name} has no main() function"
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0, (
        f"{entry_point} ({module_name}.main) exited with "
        f"code {excinfo.value.code}, expected 0"
    )
    captured = capsys.readouterr()
    # argparse always prints either prog or 'usage:' on --help.
    haystack = (captured.out + captured.err).lower()
    assert "usage" in haystack, (
        f"{entry_point} --help produced no usage text: {captured.out!r}"
    )


def test_entry_points_match_pyproject():
    """Keep ENTRY_POINTS in sync with [project.scripts] in pyproject.toml."""
    try:
        import tomllib  # stdlib on 3.11+
    except ModuleNotFoundError:  # pragma: no cover — py3.10 fallback
        tomllib = pytest.importorskip("tomli")
    from pathlib import Path

    pyproject = tomllib.loads(
        Path(__file__).resolve().parents[1].joinpath(
            "pyproject.toml",
        ).read_text(encoding="utf-8"),
    )
    declared = pyproject["project"]["scripts"]
    listed = {ep: target.split(":", 1)[0] for ep, target in declared.items()}
    tested = dict(ENTRY_POINTS)
    assert listed == tested, (
        f"ENTRY_POINTS drifted from [project.scripts]:\n"
        f"  declared={listed}\n"
        f"  tested  ={tested}"
    )
