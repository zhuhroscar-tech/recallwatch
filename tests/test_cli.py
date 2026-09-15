"""CLI integration tests: end-to-end snapshot -> diff, exit codes, JSON output."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def run_cli(args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "recallwatch.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_snapshot_synthetic_writes_valid_report(tmp_path):
    out = tmp_path / "report.json"
    result = run_cli(
        [
            "snapshot",
            "--n-items", "300",
            "--n-queries", "40",
            "--dim", "8",
            "--seed", "1",
            "--out", str(out),
            "--no-color",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "overall_recall" in payload
    assert 0.0 <= payload["overall_recall"] <= 1.0
    assert set(payload["segment_recall"].keys()) == {"head", "torso", "tail"}


def test_snapshot_is_deterministic_given_same_seed(tmp_path):
    out1 = tmp_path / "r1.json"
    out2 = tmp_path / "r2.json"
    for out in (out1, out2):
        result = run_cli(
            [
                "snapshot", "--n-items", "200", "--n-queries", "30",
                "--dim", "6", "--seed", "42", "--out", str(out), "--no-color",
            ]
        )
        assert result.returncode == 0
    assert json.loads(out1.read_text()) == json.loads(out2.read_text())


def test_diff_detects_synthetic_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "k": 10,
        "overall_recall": 0.9,
        "segment_recall": {"head": 0.98, "torso": 0.92, "tail": 0.85},
        "segment_counts": {"head": 30, "torso": 30, "tail": 30},
        "n_queries": 90,
    }))
    current = tmp_path / "current.json"
    current.write_text(json.dumps({
        "k": 10,
        "overall_recall": 0.75,
        "segment_recall": {"head": 0.97, "torso": 0.90, "tail": 0.40},
        "segment_counts": {"head": 30, "torso": 30, "tail": 30},
        "n_queries": 90,
    }))
    result = run_cli(["diff", str(baseline), str(current), "--no-color"])
    assert result.returncode == 1, result.stdout  # regression -> nonzero exit
    assert "tail" in result.stdout
    assert "REGRESSION" in result.stdout


def test_diff_no_regression_exits_zero(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "k": 10, "overall_recall": 0.9,
        "segment_recall": {"head": 0.95, "torso": 0.90, "tail": 0.85},
        "segment_counts": {"head": 10, "torso": 10, "tail": 10}, "n_queries": 30,
    }))
    current = tmp_path / "current.json"
    current.write_text(json.dumps({
        "k": 10, "overall_recall": 0.91,
        "segment_recall": {"head": 0.96, "torso": 0.91, "tail": 0.86},
        "segment_counts": {"head": 10, "torso": 10, "tail": 10}, "n_queries": 30,
    }))
    result = run_cli(["diff", str(baseline), str(current), "--no-color"])
    assert result.returncode == 0, result.stdout


def test_diff_json_output_is_valid_json(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "k": 10, "overall_recall": 0.9,
        "segment_recall": {"head": 0.95, "torso": 0.90, "tail": 0.85},
        "segment_counts": {"head": 10, "torso": 10, "tail": 10}, "n_queries": 30,
    }))
    current = tmp_path / "current.json"
    current.write_text(json.dumps({
        "k": 10, "overall_recall": 0.9,
        "segment_recall": {"head": 0.95, "torso": 0.90, "tail": 0.85},
        "segment_counts": {"head": 10, "torso": 10, "tail": 10}, "n_queries": 30,
    }))
    result = run_cli(["diff", str(baseline), str(current), "--json"])
    parsed = json.loads(result.stdout)
    assert len(parsed) == 3


def test_version_flag():
    result = run_cli(["--version"])
    assert result.returncode == 0
    assert "recallwatch" in result.stdout


def test_snapshot_with_explicit_npy_files(tmp_path):
    corpus_path = tmp_path / "corpus.npy"
    queries_path = tmp_path / "queries.npy"
    rng = np.random.default_rng(0)
    np.save(corpus_path, rng.normal(size=(50, 4)))
    np.save(queries_path, rng.normal(size=(10, 4)))
    out = tmp_path / "report.json"
    result = run_cli(
        [
            "snapshot", "--corpus", str(corpus_path), "--queries", str(queries_path),
            "--k", "3", "--out", str(out), "--no-color",
        ]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["n_queries"] == 10
    assert payload["_meta"]["corpus_size"] == 50


def test_snapshot_rejects_1d_array(tmp_path):
    bad_path = tmp_path / "bad.npy"
    np.save(bad_path, np.array([1, 2, 3]))
    out = tmp_path / "report.json"
    result = run_cli(["snapshot", "--corpus", str(bad_path), "--out", str(out)])
    assert result.returncode != 0
