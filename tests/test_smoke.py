"""Smoke tests for the pyLACE skeleton."""

import pylace


def test_version_present():
    assert pylace.__version__
