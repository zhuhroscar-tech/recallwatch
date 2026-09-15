"""recallwatch CLI: snapshot and diff segmented ANN recall over time."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .core import (
    RandomProjectionLSH,
    SegmentedRecallReport,
    diff_reports,
    measure_segmented_recall,
)
from .style import Style, print_fields, resolve_style, status_headline


def _load_vectors(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2D array, got shape {arr.shape}")
    return arr


def _synthetic_corpus(n_items: int, dim: int, n_clusters: int, seed: int) -> np.ndarray:
    """Deterministic clustered synthetic corpus: realistic non-uniform density,
    which is exactly the condition under which HNSW/LSH-style recall varies
    by query region (documented production failure mode -- see README).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(n_clusters, dim))
    assignments = rng.integers(0, n_clusters, size=n_items)
    noise = rng.normal(scale=1.0, size=(n_items, dim))
    return centers[assignments] + noise


def _synthetic_queries(n_queries: int, dim: int, corpus: np.ndarray, seed: int) -> np.ndarray:
    """Sample queries as small perturbations of random corpus points, so
    some land in dense clusters (head) and some in sparse regions (tail)
    depending on which corpus point they perturb.
    """
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, corpus.shape[0], size=n_queries)
    noise = rng.normal(scale=0.5, size=(n_queries, corpus.shape[1]))
    return corpus[idx] + noise


def cmd_snapshot(args: argparse.Namespace) -> int:
    if args.corpus:
        corpus = _load_vectors(args.corpus)
    else:
        corpus = _synthetic_corpus(args.n_items, args.dim, args.n_clusters, args.seed)

    if args.queries:
        queries = _load_vectors(args.queries)
    else:
        queries = _synthetic_queries(args.n_queries, corpus.shape[1], corpus, args.seed)

    index = RandomProjectionLSH(
        dim=corpus.shape[1], n_tables=args.n_tables, n_bits=args.n_bits, seed=args.seed
    )
    index.build(corpus)

    report = measure_segmented_recall(
        corpus, queries, index, k=args.k, density_k=args.density_k
    )

    payload = report.to_dict()
    payload["_meta"] = {
        "corpus_size": int(corpus.shape[0]),
        "dim": int(corpus.shape[1]),
        "n_tables": args.n_tables,
        "n_bits": args.n_bits,
        "seed": args.seed,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    style = resolve_style(args.no_color)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(status_headline(style, "info", f"recallwatch snapshot -> {out_path}"))
        print_fields(
            [
                ("overall recall@%d" % args.k, f"{report.overall_recall:.4f}"),
                *[
                    (f"  {seg} recall (n={report.segment_counts[seg]})", f"{val:.4f}" if val == val else "n/a")
                    for seg, val in report.segment_recall.items()
                ],
            ]
        )
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    baseline = SegmentedRecallReport.from_dict(json.loads(Path(args.baseline).read_text()))
    current = SegmentedRecallReport.from_dict(json.loads(Path(args.current).read_text()))
    findings = diff_reports(baseline, current, threshold=args.threshold)

    style = resolve_style(args.no_color)
    any_regression = any(f.exceeds_threshold for f in findings)

    if args.json:
        print(
            json.dumps(
                [f.__dict__ for f in findings],
                indent=2,
                sort_keys=True,
                default=lambda x: None if x != x else x,  # NaN -> null
            )
        )
    else:
        level = "fail" if any_regression else "ok"
        print(status_headline(style, level, "recallwatch diff"))
        rows = []
        for f in findings:
            if f.delta != f.delta:  # NaN
                rows.append((f.segment, "insufficient data in one snapshot"))
                continue
            arrow = "v" if f.delta < 0 else "^"
            flag = "  <-- REGRESSION" if f.exceeds_threshold else ""
            rows.append(
                (
                    f.segment,
                    f"{f.baseline_recall:.4f} -> {f.current_recall:.4f} ({arrow}{abs(f.delta):.4f}){flag}",
                )
            )
        print_fields(rows)

    return 1 if any_regression else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recallwatch",
        description="Segmented recall drift monitor for approximate-nearest-neighbor indexes.",
    )
    parser.add_argument("--version", action="version", version=f"recallwatch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="measure segmented recall@k and save a report")
    snap.add_argument("--corpus", help="path to a .npy corpus array (n_items, dim); synthetic if omitted")
    snap.add_argument("--queries", help="path to a .npy query array (n_queries, dim); synthetic if omitted")
    snap.add_argument("--n-items", type=int, default=5000, help="synthetic corpus size (default 5000)")
    snap.add_argument("--dim", type=int, default=32, help="synthetic vector dimension (default 32)")
    snap.add_argument("--n-clusters", type=int, default=20, help="synthetic cluster count (default 20)")
    snap.add_argument("--n-queries", type=int, default=300, help="synthetic query count (default 300)")
    snap.add_argument("--k", type=int, default=10, help="recall@k (default 10)")
    snap.add_argument("--density-k", type=int, default=10, help="k used for density segmentation (default 10)")
    snap.add_argument("--n-tables", type=int, default=8, help="LSH hash tables (default 8)")
    snap.add_argument("--n-bits", type=int, default=12, help="LSH bits per table (default 12)")
    snap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    snap.add_argument("--out", required=True, help="output JSON report path")
    snap.add_argument("--json", action="store_true", help="also print the report as JSON")
    snap.add_argument("--no-color", action="store_true", help="disable ANSI color output")
    snap.set_defaults(func=cmd_snapshot)

    diff = sub.add_parser("diff", help="compare two snapshot reports and flag segment recall regressions")
    diff.add_argument("baseline", help="baseline report JSON path")
    diff.add_argument("current", help="current report JSON path")
    diff.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="absolute recall drop that counts as a regression (default 0.05)",
    )
    diff.add_argument("--json", action="store_true", help="print findings as JSON")
    diff.add_argument("--no-color", action="store_true", help="disable ANSI color output")
    diff.set_defaults(func=cmd_diff)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
