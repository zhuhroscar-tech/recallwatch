"""In-process CLI tests: exercise recallwatch.cli.main() via direct calls
(not subprocess), so pytest-cov actually measures cli.py's statements.

recallwatch was the only owned repo whose test_cli.py drove the CLI
exclusively through `subprocess.run([sys.executable, "-m", ...])`
(kept below as true end-to-end process-boundary integration tests --
they still have value and are NOT removed). But every other CLI in
this fleet (numguard, causality-audit, trim-doctor, zram-doctor,
privaudit, ...) ALSO has a companion in-process test file that calls
`cli.main([...])` directly, because a subprocess is a separate
interpreter process that coverage.py cannot instrument -- so
recallwatch's actual coverage on cli.py was 0% (100/100 lines
"missing") despite 8 passing subprocess tests, silently hiding the
one file most likely to have an untested branch. This file closes
that fleet-wide consistency gap without touching the working
subprocess tests.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from recallwatch import cli


def test_snapshot_inprocess_writes_valid_report(tmp_path, capsys):
    out = tmp_path / "report.json"
    exit_code = cli.main(
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
    assert exit_code == 0
    payload = json.loads(out.read_text())
    assert "overall_recall" in payload
    assert 0.0 <= payload["overall_recall"] <= 1.0
    assert set(payload["segment_recall"].keys()) == {"head", "torso", "tail"}
    captured = capsys.readouterr()
    assert "recallwatch snapshot" in captured.out


def test_snapshot_inprocess_json_flag_prints_payload(tmp_path, capsys):
    out = tmp_path / "report.json"
    exit_code = cli.main(
        [
            "snapshot",
            "--n-items", "200", "--n-queries", "30", "--dim", "6",
            "--seed", "0", "--out", str(out), "--json", "--no-color",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "overall_recall" in payload
    assert payload["_meta"]["seed"] == 0


def test_diff_inprocess_detects_regression(tmp_path, capsys):
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
    exit_code = cli.main(["diff", str(baseline), str(current), "--no-color"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "tail" in captured.out
    assert "REGRESSION" in captured.out


def test_diff_inprocess_no_regression_exits_zero(tmp_path):
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
    exit_code = cli.main(["diff", str(baseline), str(current), "--no-color"])
    assert exit_code == 0


def test_diff_inprocess_text_output_reports_insufficient_data_for_nan_segment(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "k": 10, "overall_recall": 0.9,
        "segment_recall": {"head": 0.95, "torso": 0.90, "tail": None},
        "segment_counts": {"head": 10, "torso": 10, "tail": 0}, "n_queries": 20,
    }))
    current = tmp_path / "current.json"
    current.write_text(json.dumps({
        "k": 10, "overall_recall": 0.9,
        "segment_recall": {"head": 0.95, "torso": 0.90, "tail": None},
        "segment_counts": {"head": 10, "torso": 10, "tail": 0}, "n_queries": 20,
    }))
    exit_code = cli.main(["diff", str(baseline), str(current), "--no-color"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "insufficient data in one snapshot" in captured.out


def test_diff_inprocess_json_output_is_valid_json_with_nan_handling(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "k": 10, "overall_recall": 0.9,
        "segment_recall": {"head": 0.95, "torso": 0.90, "tail": None},
        "segment_counts": {"head": 10, "torso": 10, "tail": 0}, "n_queries": 20,
    }))
    current = tmp_path / "current.json"
    current.write_text(json.dumps({
        "k": 10, "overall_recall": 0.9,
        "segment_recall": {"head": 0.95, "torso": 0.90, "tail": None},
        "segment_counts": {"head": 10, "torso": 10, "tail": 0}, "n_queries": 20,
    }))
    exit_code = cli.main(["diff", str(baseline), str(current), "--json"])
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert len(parsed) == 3
    assert exit_code == 0


def test_version_flag_inprocess_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0


def test_snapshot_inprocess_with_explicit_npy_files(tmp_path):
    corpus_path = tmp_path / "corpus.npy"
    queries_path = tmp_path / "queries.npy"
    rng = np.random.default_rng(0)
    np.save(corpus_path, rng.normal(size=(50, 4)))
    np.save(queries_path, rng.normal(size=(10, 4)))
    out = tmp_path / "report.json"
    exit_code = cli.main(
        [
            "snapshot", "--corpus", str(corpus_path), "--queries", str(queries_path),
            "--k", "3", "--out", str(out), "--no-color",
        ]
    )
    assert exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["n_queries"] == 10
    assert payload["_meta"]["corpus_size"] == 50


def test_snapshot_inprocess_rejects_1d_array(tmp_path):
    bad_path = tmp_path / "bad.npy"
    np.save(bad_path, np.array([1, 2, 3]))
    out = tmp_path / "report.json"
    with pytest.raises(ValueError):
        cli.main(["snapshot", "--corpus", str(bad_path), "--out", str(out)])


def test_build_parser_help_mentions_recall():
    parser = cli.build_parser()
    assert "recall" in parser.description.lower()
