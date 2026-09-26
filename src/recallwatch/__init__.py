"""recallwatch: segmented recall drift monitor for ANN indexes.

Core idea (see README): average recall@k hides tail-query collapse.
This package provides:
  - a cosine-similarity random-projection LSH index (the "approximate" side)
  - a brute-force exact top-k oracle (the ground truth)
  - density-based query segmentation (head / torso / tail)
  - snapshot + diff tooling to detect per-segment recall drift over time
"""

__version__ = "0.1.4"
