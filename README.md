[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# recallwatch

Measure approximate-nearest-neighbor (ANN) recall by query-density segment, then compare snapshots for regressions. `recallwatch` reports **head / torso / tail** recall alongside the overall average so a weak segment is not hidden by stronger queries.

The CLI includes a signed random-projection LSH index and an exact cosine-search reference. It is a local measurement tool, not a hosted monitor or a ready-made connector to your production vector database.

## Install and try

Requires Python 3.9+ and NumPy 1.23+ (installed by pip).

```bash
git clone https://github.com/zhuhroscar-tech/recallwatch.git
cd recallwatch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
recallwatch snapshot --n-items 3000 --n-queries 200 --dim 24 --seed 0 --out baseline.json
recallwatch snapshot --n-items 3000 --n-queries 200 --dim 24 --seed 0 --n-bits 20 --out current.json
recallwatch diff baseline.json current.json --threshold 0.05
```

This uses deterministic synthetic vectors and changes the LSH bucket resolution. More hash bits make finer buckets; measured recall depends on the data and index settings. `snapshot` writes the requested JSON file, overwriting an existing file at that path.

## Bring your own data

```bash
recallwatch snapshot --corpus corpus.npy --queries queries.npy --out report.json
recallwatch diff baseline.json report.json --json
```

Use two-dimensional numeric `.npy` arrays shaped `(n_items, dim)` and `(n_queries, dim)`, with matching dimensions. See `recallwatch snapshot --help` for `--k`, `--density-k`, `--n-tables`, and other controls.

A segment drop **greater than** the absolute threshold makes `diff` exit `1`; otherwise it exits `0`. Improvements are not regressions. Missing/NaN segments are reported as insufficient data and do not trigger that exit code—inspect the report rather than treating `0` as proof of full coverage.

## Limits and integration

Segments use distance quantiles within each query batch, not fixed categories across runs. Keep datasets and settings comparable when interpreting drift. Exact reference search is expensive; large-scale production performance is unverified.

To measure another ANN index, supply an object with `search(query, k) -> ids` to `measure_segmented_recall()` in [core.py](src/recallwatch/core.py). Returned IDs must correspond to corpus row indices. The bundled LSH is a reference implementation, not a performance competitor to FAISS or HNSWlib.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -v
```

[Tests](tests/test_core.py) · [CI](.github/workflows/ci.yml) · [MIT license](LICENSE)
