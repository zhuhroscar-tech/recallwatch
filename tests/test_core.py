"""Correctness tests for recallwatch.core.

Design: every algorithmic claim has an independent check.
- ExactIndex is verified against a from-scratch (non-numpy-matmul)
  reference brute-force cosine search, so the "oracle" itself isn't
  trusted blindly.
- RandomProjectionLSH recall is verified to be a real number in [0, 1]
  that is 1.0 on a trivial/tiny corpus (every point is its own bucket
  neighbor) and degrades in the expected direction as bits increase
  (finer buckets -> fewer candidates -> recall can only go down or
  stay the same for a fixed k, never up, holding tables/seed fixed).
- segment_queries partitions every query into exactly one segment and
  respects the requested quantile split.
- diff_reports correctly flags regressions and never flags NaN as one.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from recallwatch.core import (
    DEFAULT_DRIFT_THRESHOLD,
    ExactIndex,
    RandomProjectionLSH,
    SegmentedRecallReport,
    diff_reports,
    measure_segmented_recall,
    recall_at_k,
    segment_queries,
)


def _reference_cosine_topk(vectors: np.ndarray, query: np.ndarray, k: int) -> list:
    """Independent, deliberately-non-vectorized reference implementation
    (plain Python loops, no shared code path with ExactIndex) used only
    in tests to cross-check ExactIndex.search().
    """
    def cos(a, b):
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (na * nb)

    scored = [(cos(vec, query), i) for i, vec in enumerate(vectors)]
    scored.sort(key=lambda t: -t[0])
    return [i for _, i in scored[:k]]


class TestExactIndex:
    def test_matches_independent_reference_implementation(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(50, 8))
        query = rng.normal(size=8)
        idx = ExactIndex(vectors)
        got = idx.search(query, k=5)
        expected = _reference_cosine_topk(vectors, query, k=5)
        assert list(got) == expected

    def test_zero_vector_does_not_crash_or_nan(self):
        vectors = np.zeros((3, 4))
        vectors[0] = [1, 0, 0, 0]
        idx = ExactIndex(vectors)
        result = idx.search(np.array([1, 0, 0, 0]), k=3)
        assert len(result) == 3
        assert not np.isnan(idx.knn_mean_distance(np.array([1, 0, 0, 0]), 3))

    def test_query_matches_itself_first(self):
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(20, 6))
        idx = ExactIndex(vectors)
        for i in range(20):
            top = idx.search(vectors[i], k=1)
            assert top[0] == i

    def test_k_larger_than_corpus_is_clamped_not_an_error(self):
        vectors = np.random.default_rng(0).normal(size=(3, 4))
        idx = ExactIndex(vectors)
        result = idx.search(vectors[0], k=100)
        assert len(result) == 3

    def test_knn_mean_distance_is_zero_for_identical_points(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        idx = ExactIndex(vectors)
        d = idx.knn_mean_distance(np.array([1.0, 0.0]), k=3)
        assert abs(d) < 1e-9


class TestRandomProjectionLSH:
    def test_recall_is_perfect_on_tiny_trivial_corpus(self):
        # 5 points, k=1: with enough tables/bits the true nearest neighbor
        # (itself) must always be found since a point always hashes into
        # its own bucket in at least one table.
        rng = np.random.default_rng(3)
        vectors = rng.normal(size=(5, 4))
        index = RandomProjectionLSH(dim=4, n_tables=16, n_bits=2, seed=3)
        index.build(vectors)
        exact = ExactIndex(vectors)
        recalls = []
        for v in vectors:
            approx_ids = index.search(v, k=1)
            exact_ids = exact.search(v, k=1)
            recalls.append(recall_at_k(approx_ids, exact_ids))
        assert all(r == 1.0 for r in recalls)

    def test_recall_never_exceeds_one_or_goes_negative(self):
        rng = np.random.default_rng(5)
        vectors = rng.normal(size=(200, 16))
        queries = rng.normal(size=(30, 16))
        index = RandomProjectionLSH(dim=16, n_tables=4, n_bits=8, seed=5)
        index.build(vectors)
        exact = ExactIndex(vectors)
        for q in queries:
            r = recall_at_k(index.search(q, k=10), exact.search(q, k=10))
            assert 0.0 <= r <= 1.0

    def test_finer_buckets_never_increase_recall_holding_tables_fixed(self):
        # More bits -> smaller/finer buckets -> candidate sets are subsets
        # in expectation, so mean recall over many queries should not
        # improve (real, measurable, monotone-in-expectation property of
        # signed random-projection LSH, not an implementation artifact).
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(500, 20))
        queries = rng.normal(size=(60, 20))
        exact = ExactIndex(vectors)
        exact_results = [exact.search(q, k=10) for q in queries]

        coarse = RandomProjectionLSH(dim=20, n_tables=6, n_bits=4, seed=7)
        coarse.build(vectors)
        fine = RandomProjectionLSH(dim=20, n_tables=6, n_bits=14, seed=7)
        fine.build(vectors)

        coarse_recall = np.mean(
            [recall_at_k(coarse.search(q, k=10), e) for q, e in zip(queries, exact_results)]
        )
        fine_recall = np.mean(
            [recall_at_k(fine.search(q, k=10), e) for q, e in zip(queries, exact_results)]
        )
        # Allow a small tolerance for sampling noise but the direction
        # must hold: finer hashing must not systematically beat coarser.
        assert fine_recall <= coarse_recall + 0.05

    def test_empty_candidate_bucket_returns_empty_not_a_crash(self):
        # A single-bit, single-table index on a tiny corpus can easily
        # produce a query whose bucket has zero corpus points.
        rng = np.random.default_rng(11)
        vectors = rng.normal(size=(4, 3))
        index = RandomProjectionLSH(dim=3, n_tables=1, n_bits=20, seed=11)
        index.build(vectors)
        # A query far outside any corpus point's hash pattern
        query = -vectors[0] * 1000
        result = index.search(query, k=2)
        assert isinstance(result, np.ndarray)
        # Might be empty or not depending on hash luck; either way, no crash,
        # and recall_at_k must handle it gracefully.
        exact_ids = ExactIndex(vectors).search(query, k=2)
        r = recall_at_k(result, exact_ids)
        assert 0.0 <= r <= 1.0

    def test_dimension_mismatch_raises(self):
        index = RandomProjectionLSH(dim=5, seed=0)
        with pytest.raises(ValueError):
            index.build(np.zeros((10, 4)))

    def test_search_before_build_raises(self):
        index = RandomProjectionLSH(dim=4, seed=0)
        with pytest.raises(RuntimeError):
            index.search(np.zeros(4), k=1)


class TestSegmentQueries:
    def test_every_query_gets_exactly_one_segment(self):
        rng = np.random.default_rng(9)
        corpus = rng.normal(size=(100, 8))
        queries = rng.normal(size=(30, 8))
        exact = ExactIndex(corpus)
        segments = segment_queries(queries, exact)
        assert len(segments) == len(queries)
        assert all(s in ("head", "torso", "tail") for s in segments)

    def test_quantile_split_roughly_matches_requested_proportions(self):
        rng = np.random.default_rng(13)
        corpus = rng.normal(size=(300, 10))
        queries = rng.normal(size=(300, 10))
        exact = ExactIndex(corpus)
        segments = segment_queries(queries, exact, head_quantile=0.33, torso_quantile=0.66)
        counts = {s: segments.count(s) for s in ("head", "torso", "tail")}
        # With 300 queries split at ~1/3 quantiles, each bucket should be
        # roughly 90-110 (generous tolerance -- this is a statistical
        # property, not an exact one).
        for seg in ("head", "torso", "tail"):
            assert 70 <= counts[seg] <= 130, counts

    def test_invalid_quantiles_raise(self):
        rng = np.random.default_rng(0)
        corpus = rng.normal(size=(10, 3))
        exact = ExactIndex(corpus)
        with pytest.raises(ValueError):
            segment_queries(corpus, exact, head_quantile=0.7, torso_quantile=0.3)


class TestMeasureSegmentedRecall:
    def test_report_fields_are_internally_consistent(self):
        rng = np.random.default_rng(21)
        corpus = rng.normal(size=(400, 12))
        queries = rng.normal(size=(90, 12))
        index = RandomProjectionLSH(dim=12, n_tables=6, n_bits=8, seed=21)
        index.build(corpus)
        report = measure_segmented_recall(corpus, queries, index, k=10)
        assert sum(report.segment_counts.values()) == len(queries)
        assert report.n_queries == len(queries)
        assert 0.0 <= report.overall_recall <= 1.0
        for seg_recall in report.segment_recall.values():
            assert math.isnan(seg_recall) or 0.0 <= seg_recall <= 1.0

    def test_roundtrip_through_dict(self):
        rng = np.random.default_rng(23)
        corpus = rng.normal(size=(100, 6))
        queries = rng.normal(size=(20, 6))
        index = RandomProjectionLSH(dim=6, n_tables=4, n_bits=6, seed=23)
        index.build(corpus)
        report = measure_segmented_recall(corpus, queries, index, k=5)
        restored = SegmentedRecallReport.from_dict(report.to_dict())
        assert restored.overall_recall == report.overall_recall
        assert restored.segment_recall == report.segment_recall


class TestDiffReports:
    def _report(self, head, torso, tail):
        return SegmentedRecallReport(
            k=10,
            overall_recall=(head + torso + tail) / 3,
            segment_recall={"head": head, "torso": torso, "tail": tail},
            segment_counts={"head": 10, "torso": 10, "tail": 10},
            n_queries=30,
        )

    def test_flags_only_drops_beyond_threshold(self):
        baseline = self._report(0.95, 0.90, 0.80)
        current = self._report(0.95, 0.90, 0.60)  # tail dropped 0.20
        findings = diff_reports(baseline, current, threshold=DEFAULT_DRIFT_THRESHOLD)
        by_segment = {f.segment: f for f in findings}
        assert by_segment["tail"].exceeds_threshold is True
        assert by_segment["head"].exceeds_threshold is False
        assert by_segment["torso"].exceeds_threshold is False

    def test_improvement_is_never_flagged(self):
        baseline = self._report(0.80, 0.80, 0.80)
        current = self._report(0.95, 0.95, 0.95)
        findings = diff_reports(baseline, current)
        assert all(not f.exceeds_threshold for f in findings)

    def test_small_drop_within_threshold_not_flagged(self):
        baseline = self._report(0.90, 0.90, 0.90)
        current = self._report(0.90, 0.90, 0.87)  # only 0.03 drop
        findings = diff_reports(baseline, current, threshold=0.05)
        assert all(not f.exceeds_threshold for f in findings)

    def test_nan_segment_reported_as_unknown_not_a_regression(self):
        baseline = self._report(0.9, 0.9, 0.9)
        baseline.segment_recall["tail"] = float("nan")
        current = self._report(0.9, 0.9, 0.5)
        findings = diff_reports(baseline, current)
        tail = [f for f in findings if f.segment == "tail"][0]
        assert tail.exceeds_threshold is False
        assert math.isnan(tail.delta)


class TestRecallAtK:
    def test_empty_exact_set_is_vacuously_full_recall(self):
        assert recall_at_k(np.array([1, 2, 3]), np.array([])) == 1.0

    def test_no_overlap_is_zero(self):
        assert recall_at_k(np.array([1, 2]), np.array([3, 4])) == 0.0

    def test_full_overlap_is_one(self):
        assert recall_at_k(np.array([1, 2, 3]), np.array([3, 2, 1])) == 1.0
