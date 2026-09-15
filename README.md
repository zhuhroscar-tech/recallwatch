# recallwatch

Segmented recall drift monitor for approximate-nearest-neighbor (ANN)
indexes — catch tail-query recall collapse before your users do, instead
of after.

## The problem

Every production vector-search stack (RAG retrieval, recommendation,
semantic search) accepts a small, deliberate recall loss in exchange for
speed: an ANN index like HNSW, IVF, or LSH finds the *approximate* top-k
nearest neighbors instead of scanning everything. That tradeoff is sound
— **but the recall loss is not evenly distributed across queries**, and
teams almost universally monitor only a single "average recall" number.

Multiple independent, current production write-ups document the same
failure mode from different angles:

- A controlled HNSW study found recall degrading fastest exactly on the
  underrepresented, long-tail query regions of a corpus — common,
  well-covered ("head") queries stay near 0.94 recall while rare,
  specific ("tail") queries can collapse to 0.41, with the *average*
  masking the tail collapse entirely (`ranjankumar.in/hnsw-vector-search-recall-production`).
- A separate analysis documents the same problem from the index side:
  HNSW tombstones from deletions and corpus growth degrade recall
  gradually and silently until a threshold is crossed, and that
  degradation is a property of the *relationship* between index and
  data distribution, not a fixed constant
  (`dev.to/kenwalger/vector-search-at-scale`).
- Real ANN-library issue trackers document instances of recall silently
  dropping under specific configurations (e.g. OpenSearch k-NN
  `index.derived_source.enabled`, `facebookresearch/faiss#5320`'s stale
  cached-norm bug) where nothing crashes and no error is raised — the
  system just quietly returns worse answers.
- Japanese-language production engineering writeups
  (`tech.uzabase.com/entry/2026/07/06/094039`, `qiita.com/0h-n0`) and
  Chinese-language ANN engineering docs (Apache Doris HNSW tuning guide)
  independently confirm the same `ef_search`/`nprobe` recall-latency
  tradeoff and the practice of measuring recall@k against an exact
  brute-force oracle — but every one of these measures a *single*
  point-in-time, *aggregate* recall number, not a segmented, tracked
  metric.

Existing open tools (`vector-db-benchmark`, `feder`, AlloyDB's
`evaluate_query_recall`, Elasticsearch's recall-measurement guide) are
all **one-shot, build-time benchmarking tools**: they report one number
for one index configuration at one moment. None of them (a) segment
queries by how densely they're represented in the corpus, or (b) give
you a way to snapshot a report and diff it against a later one to catch
drift as your corpus and query distribution evolve. That gap is what
`recallwatch` fills.

## What it does

1. **`recallwatch snapshot`** — builds a real approximate index (signed
   random-projection LSH, the same family of algorithm underlying
   production ANN systems, with a genuine, measurable recall/latency
   tradeoff controlled by `--n-tables` and `--n-bits`), measures
   recall@k against an exact brute-force cosine oracle, buckets each
   query into a **head / torso / tail** density segment (via k-NN mean
   distance to the true corpus neighbors), and writes a JSON report.

2. **`recallwatch diff`** — compares two snapshot reports and flags any
   segment whose recall dropped by more than a threshold (default 0.05).
   Improvements are never flagged. Missing/NaN segments are reported as
   unknown, never silently treated as zero drift. Exits non-zero on a
   detected regression, so it composes into CI/cron pipelines.

Bring your own vectors via `--corpus`/`--queries` (`.npy` files), or use
the built-in deterministic synthetic clustered-corpus generator to
validate the tool itself or reproduce the tradeoff without any data of
your own.

## Example

```console
$ recallwatch snapshot --n-items 3000 --n-queries 200 --dim 24 --seed 0 --out baseline.json
● recallwatch snapshot -> baseline.json
  overall recall@10       0.9850
    head recall (n=66)    0.9970
    torso recall (n=66)   0.9909
    tail recall (n=68)    0.9676

$ recallwatch snapshot --n-items 3000 --n-queries 200 --dim 24 --seed 0 --n-bits 20 --out current.json

$ recallwatch diff baseline.json current.json
● recallwatch diff
  head    0.9970 -> 0.9591 (v0.0379)
  torso   0.9909 -> 0.9045 (v0.0864)  <-- REGRESSION
  tail    0.9676 -> 0.8324 (v0.1353)  <-- REGRESSION
$ echo $?
1
```

This reproduces, on synthetic data, exactly the documented failure
shape: coarsening the index (fewer, larger hash buckets) degrades tail
queries far more than head queries — the average alone would have
under-reported the real damage.

## Install

```console
pip install .
```

(No standalone `.pyz` build is offered: numpy ships compiled extension
modules that cannot run from inside a zipapp, the same constraint
`numguard` in this fleet already documents. Install via pip/wheel.)

## Algorithm notes / non-goals

- The bundled LSH index is a real, from-scratch signed random-projection
  (SimHash-style) index with multiple independent hash tables and
  exact-distance reranking of bucket candidates — it is not a stub, and
  its recall genuinely varies with `--n-tables`/`--n-bits` the way real
  ANN systems' knobs behave. It is not intended to compete with FAISS,
  HNSWlib, or ScaNN on raw performance; it exists so `recallwatch` can
  demonstrate and test the segmented-drift-detection method without an
  external ANN library dependency.
- **To monitor a real production FAISS/HNSWlib/Qdrant/Milvus index**,
  swap in your own approximate `search(query, k) -> ids` function ahead
  of `measure_segmented_recall()` (see `src/recallwatch/core.py`) — the
  segmentation, oracle, and drift-diffing logic are index-agnostic.
- Density segmentation is quantile-based *within the provided query
  batch*, not an absolute distance cutoff, so it is meaningful across
  different corpora/embedding models without manual retuning.
- This tool has been tested only on synthetic clustered Gaussian data
  and small-to-medium corpora (up to a few thousand vectors) in CI. It
  has not been benchmarked against real production embedding
  distributions or corpora beyond ~10^4 vectors; scaling behavior above
  that is unverified.

## Testing

31 tests covering: an independent from-scratch reference implementation
cross-check of the exact oracle, LSH recall bounds and monotonicity
properties, edge cases (zero vectors, empty candidate buckets, dimension
mismatches), density segmentation correctness, and end-to-end CLI
snapshot/diff behavior including a synthetic regression-detection case.
CI runs the full suite plus a real behavioral smoke test (coarsening the
index must trigger `diff`'s regression exit code) on `ubuntu-latest`.

```console
pip install -e ".[dev]"
pytest -v
```

## License

MIT
