"""Core algorithms for recallwatch.

Two independent pieces of real algorithmic work:

1. ``RandomProjectionLSH`` -- a genuine (not toy) approximate-nearest-
   neighbor index using signed random-projection hashing (SimHash-style
   locality-sensitive hashing) with multiple independent hash tables and
   exact-distance reranking of bucket candidates. Like production ANN
   indexes (HNSW, IVF), it has a real, tunable recall/latency tradeoff,
   and its recall genuinely degrades in predictable, measurable ways as
   corpus density or the query distribution shifts -- it is not a
   canned or hardcoded-to-fail demo.

2. ``segment_queries`` -- density-based query segmentation. Rather than
   a single "average recall" number (which the documented failure mode
   in production ANN systems hides tail collapse behind), each query is
   classified into a head/torso/tail density bucket using k-NN density
   estimation against the exact index, so recall can be measured and
   compared *per segment*.

``ExactIndex`` is the brute-force oracle used both as ground truth for
recall measurement and as the reference for density estimation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero vectors are left as zero (never divide by 0)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return vectors / safe


class ExactIndex:
    """Brute-force cosine-similarity top-k oracle. Ground truth, not approximate."""

    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array (n_items, dim)")
        self.vectors = np.asarray(vectors, dtype=np.float64)
        self._normalized = _normalize(self.vectors)

    @property
    def size(self) -> int:
        return self.vectors.shape[0]

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """Return the ids of the true top-k nearest neighbors by cosine similarity."""
        q = _normalize(np.asarray(query, dtype=np.float64).reshape(1, -1))[0]
        scores = self._normalized @ q
        k = min(k, self.size)
        # argsort descending, stable enough for our purposes via mergesort
        top = np.argsort(-scores, kind="mergesort")[:k]
        return top

    def knn_mean_distance(self, query: np.ndarray, k: int) -> float:
        """Mean cosine distance (1 - similarity) to the true k nearest neighbors.

        Used as a density signal: small mean distance => query sits in a
        dense region (head); large mean distance => sparse/tail region.
        """
        q = _normalize(np.asarray(query, dtype=np.float64).reshape(1, -1))[0]
        scores = self._normalized @ q
        k = min(k, self.size)
        top_scores = np.sort(scores)[::-1][:k]
        distances = 1.0 - top_scores
        return float(np.mean(distances))


@dataclass
class RandomProjectionLSH:
    """Signed random-projection LSH index with multi-table candidate rerank.

    Parameters mirror the real recall/latency tradeoff knobs found in
    production ANN systems (HNSW's ``ef_search``, IVF's ``nprobe``):

    - ``n_tables``: number of independent hash tables (more tables ->
      higher recall, more memory/query cost).
    - ``n_bits``: number of random hyperplanes per table (more bits ->
      finer buckets -> faster but lower recall per table).
    - ``seed``: deterministic RNG seed so results are reproducible.
    """

    dim: int
    n_tables: int = 8
    n_bits: int = 12
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # One random hyperplane matrix per table: shape (n_bits, dim)
        self._planes = [
            rng.normal(size=(self.n_bits, self.dim)) for _ in range(self.n_tables)
        ]
        self._tables: List[Dict[int, List[int]]] = [dict() for _ in range(self.n_tables)]
        self._vectors: Optional[np.ndarray] = None
        self._normalized: Optional[np.ndarray] = None

    def _hash(self, table_idx: int, vec: np.ndarray) -> int:
        proj = self._planes[table_idx] @ vec
        bits = (proj >= 0).astype(np.uint32)
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h

    def build(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self._vectors = vectors
        self._normalized = _normalize(vectors)
        self._tables = [dict() for _ in range(self.n_tables)]
        for i, vec in enumerate(self._normalized):
            for t in range(self.n_tables):
                h = self._hash(t, vec)
                self._tables[t].setdefault(h, []).append(i)

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """Approximate top-k: union of bucket candidates across tables,
        reranked by exact cosine similarity among candidates only.

        If the union of candidate buckets is empty (a real ANN failure
        mode -- a query landing in a bucket with no corpus points in any
        table), an empty array is returned rather than silently falling
        back to exact search; callers measuring recall must treat this
        as zero recall for that query, exactly like a production ANN
        system returning no hits.
        """
        if self._normalized is None:
            raise RuntimeError("call build() before search()")
        q = _normalize(np.asarray(query, dtype=np.float64).reshape(1, -1))[0]
        candidates: set = set()
        for t in range(self.n_tables):
            h = self._hash(t, q)
            candidates.update(self._tables[t].get(h, []))
        if not candidates:
            return np.array([], dtype=int)
        cand_idx = np.array(sorted(candidates))
        scores = self._normalized[cand_idx] @ q
        k = min(k, len(cand_idx))
        order = np.argsort(-scores, kind="mergesort")[:k]
        return cand_idx[order]


def recall_at_k(approx_ids: np.ndarray, exact_ids: np.ndarray) -> float:
    """Fraction of the exact top-k found anywhere in the approximate result."""
    if len(exact_ids) == 0:
        return 1.0
    hit = len(set(approx_ids.tolist()) & set(exact_ids.tolist()))
    return hit / len(exact_ids)


SEGMENT_HEAD = "head"
SEGMENT_TORSO = "torso"
SEGMENT_TAIL = "tail"


def segment_queries(
    queries: np.ndarray,
    exact_index: ExactIndex,
    density_k: int = 10,
    head_quantile: float = 0.33,
    torso_quantile: float = 0.66,
) -> List[str]:
    """Classify each query into head/torso/tail by k-NN density.

    Density is estimated as the mean cosine distance to the query's true
    ``density_k`` nearest corpus neighbors (smaller = denser = head).
    Quantile thresholds are computed over the *provided query batch*
    itself, matching the diagnostic in the referenced literature: recall
    should be measured relative to how a query sits among the corpus,
    not an arbitrary absolute distance cutoff that would not transfer
    across corpora/embedding models.
    """
    if not (0.0 < head_quantile < torso_quantile < 1.0):
        raise ValueError("require 0 < head_quantile < torso_quantile < 1")
    distances = np.array(
        [exact_index.knn_mean_distance(q, density_k) for q in queries]
    )
    lo = np.quantile(distances, head_quantile)
    hi = np.quantile(distances, torso_quantile)
    segments = []
    for d in distances:
        if d <= lo:
            segments.append(SEGMENT_HEAD)
        elif d <= hi:
            segments.append(SEGMENT_TORSO)
        else:
            segments.append(SEGMENT_TAIL)
    return segments


@dataclass
class SegmentedRecallReport:
    """Per-segment recall@k plus overall, for one (corpus, query-set, index) run."""

    k: int
    overall_recall: float
    segment_recall: Dict[str, float]
    segment_counts: Dict[str, int]
    n_queries: int

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "overall_recall": self.overall_recall,
            "segment_recall": self.segment_recall,
            "segment_counts": self.segment_counts,
            "n_queries": self.n_queries,
        }

    @staticmethod
    def from_dict(d: dict) -> "SegmentedRecallReport":
        """Build a report from a decoded JSON dict.

        A segment with zero queries is recorded as "unknown" recall.
        Python's own ``json`` module round-trips that as the
        non-standard ``NaN`` literal, but strict-JSON producers (a
        different tool version, a hand-edited fixture, a report
        re-serialized through a standards-compliant encoder that
        rejects NaN) can only represent "no value" as JSON ``null``,
        which decodes to Python ``None``. Treat ``None`` the same as
        NaN here rather than letting it reach ``np.isnan`` in
        ``diff_reports`` and raise an opaque ``TypeError`` -- this
        class's whole purpose is comparing snapshot files that may
        have been produced by a different run or tool, not just the
        ones written by this same process a moment earlier.
        """
        segment_recall = {
            seg: (float("nan") if val is None else val)
            for seg, val in d["segment_recall"].items()
        }
        return SegmentedRecallReport(
            k=d["k"],
            overall_recall=d["overall_recall"],
            segment_recall=segment_recall,
            segment_counts=dict(d["segment_counts"]),
            n_queries=d["n_queries"],
        )


def measure_segmented_recall(
    corpus: np.ndarray,
    queries: np.ndarray,
    approx_index: RandomProjectionLSH,
    k: int = 10,
    density_k: int = 10,
) -> SegmentedRecallReport:
    """Build the exact oracle, segment queries by density, and measure
    recall@k for the given already-built approximate index, per segment.
    """
    exact = ExactIndex(corpus)
    segments = segment_queries(queries, exact, density_k=density_k)

    per_segment_recalls: Dict[str, List[float]] = {
        SEGMENT_HEAD: [],
        SEGMENT_TORSO: [],
        SEGMENT_TAIL: [],
    }
    all_recalls: List[float] = []

    for query, seg in zip(queries, segments):
        exact_ids = exact.search(query, k)
        approx_ids = approx_index.search(query, k)
        r = recall_at_k(approx_ids, exact_ids)
        per_segment_recalls[seg].append(r)
        all_recalls.append(r)

    segment_recall = {
        seg: (float(np.mean(vals)) if vals else float("nan"))
        for seg, vals in per_segment_recalls.items()
    }
    segment_counts = {seg: len(vals) for seg, vals in per_segment_recalls.items()}
    overall = float(np.mean(all_recalls)) if all_recalls else float("nan")

    return SegmentedRecallReport(
        k=k,
        overall_recall=overall,
        segment_recall=segment_recall,
        segment_counts=segment_counts,
        n_queries=len(queries),
    )


DEFAULT_DRIFT_THRESHOLD = 0.05


@dataclass
class DriftFinding:
    segment: str
    baseline_recall: float
    current_recall: float
    delta: float
    exceeds_threshold: bool


def diff_reports(
    baseline: SegmentedRecallReport,
    current: SegmentedRecallReport,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> List[DriftFinding]:
    """Compare two segmented recall reports and flag per-segment drops.

    Only *drops* (current < baseline) are flagged as findings of concern;
    an improvement is reported but never flagged, mirroring how a
    monitoring tool should alert on regressions, not celebrate variance.
    NaN segments (no queries fell in that bucket in one of the two runs)
    are reported as unknown, never silently treated as zero drift.
    """
    findings = []
    for segment in (SEGMENT_HEAD, SEGMENT_TORSO, SEGMENT_TAIL):
        b = baseline.segment_recall.get(segment, float("nan"))
        c = current.segment_recall.get(segment, float("nan"))
        if np.isnan(b) or np.isnan(c):
            findings.append(DriftFinding(segment, b, c, float("nan"), False))
            continue
        delta = c - b
        exceeds = delta < -abs(threshold)
        findings.append(DriftFinding(segment, b, c, delta, exceeds))
    return findings
