"""Regression tests for current pyproject license metadata.

Setuptools warns on deprecated `project.license = {text = ...}` tables and will
stop supporting that shape. Keep this package on the SPDX-string form so source
and wheel builds stay warning-free on modern build backends.
"""
from __future__ import annotations

import re
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_text() -> str:
    return _PYPROJECT.read_text(encoding="utf-8")


def test_project_license_uses_spdx_string_not_deprecated_table():
    text = _pyproject_text()

    assert 'license = "MIT"' in text
    assert "license = {" not in text
    assert 'license-files = ["LICENSE"]' in text


def test_build_backend_is_new_enough_for_license_files():
    text = _pyproject_text()

    match = re.search(r'^requires\s*=\s*\[(.*?)\]', text, re.MULTILINE)
    assert match, "pyproject.toml must declare build-system.requires"
    requires = match.group(1)
    assert '"setuptools>=77"' in requires
