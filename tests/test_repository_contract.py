"""Repository-level contract checks for release completeness."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_required_project_files_are_present():
    for relative_path in [
        "LICENSE",
        "README.md",
        "README.zh-CN.md",
        "CHANGELOG.md",
        "MANIFEST.in",
        "pyproject.toml",
        ".github/workflows/ci.yml",
    ]:
        path = ROOT / relative_path
        assert path.is_file(), f"Missing required repository file: {relative_path}"
        assert path.read_text(encoding="utf-8").strip(), f"{relative_path} is empty"


def test_readmes_link_license_tests_ci_and_release_history():
    english = read_text("README.md")
    chinese = read_text("README.zh-CN.md")

    assert "[Release history](CHANGELOG.md)" in english
    assert "[MIT license](LICENSE)" in english
    assert "[Tests](tests/test_core.py)" in english
    assert "[CI](.github/workflows/ci.yml)" in english

    assert "[发布历史](CHANGELOG.md)" in chinese
    assert "[MIT 许可证](LICENSE)" in chinese
    assert "[测试](tests/test_core.py)" in chinese
    assert "[CI](.github/workflows/ci.yml)" in chinese


def test_changelog_contains_current_version():
    pyproject = read_text("pyproject.toml")
    version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert version, "pyproject.toml must declare project.version"
    assert f"## v{version.group(1)}" in read_text("CHANGELOG.md")


def test_package_resource_links_are_declared():
    pyproject = read_text("pyproject.toml")
    for required in [
        '[project.urls]',
        'Homepage = "https://github.com/zhuhroscar-tech/recallwatch"',
        'Issues = "https://github.com/zhuhroscar-tech/recallwatch/issues"',
        'Changelog = "https://github.com/zhuhroscar-tech/recallwatch/blob/main/CHANGELOG.md"',
    ]:
        assert required in pyproject


def test_manifest_includes_release_metadata_and_tests():
    manifest = read_text("MANIFEST.in")
    for required in [
        "include CHANGELOG.md",
        "include LICENSE",
        "include README.md",
        "include README.zh-CN.md",
        "recursive-include tests *.py",
        "recursive-include .github/workflows *.yml",
    ]:
        assert required in manifest


def test_ci_builds_release_artifacts_and_codeql_is_configured():
    ci = read_text(".github/workflows/ci.yml")
    assert 'tags: ["v*"]' in ci
    assert "python -m build" in ci
    assert "sha256sum * > SHA256SUMS.txt" in ci
    assert "actions/upload-artifact@v4" in ci

    codeql = ROOT / ".github" / "workflows" / "codeql.yml"
    assert codeql.is_file(), "CodeQL workflow should cover this public package"
    codeql_text = codeql.read_text(encoding="utf-8")
    assert "github/codeql-action/init" in codeql_text
    assert "github/codeql-action/analyze" in codeql_text
