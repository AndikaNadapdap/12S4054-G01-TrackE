"""Hand-checkable metric example (W10 requirement, prepared early).

recommended = [A, B, C, D], relevant = {B, D, X}, K = 4
  Recall@4  = 2/3                       = 0.6667
  DCG@4     = 1/log2(3) + 1/log2(5)     = 0.6309 + 0.4307 = 1.0616
  IDCG@4    = 1 + 1/log2(3) + 1/log2(4) = 1 + 0.6309 + 0.5 = 2.1309
  NDCG@4    = 1.0616 / 2.1309           = 0.4982
  HitRate@4 = 1,  MRR@4 = 1/2 = 0.5
Run from the project root:  python -m tests.test_metrics
"""
import math
from src.common import recall_at_k, ndcg_at_k, hit_at_k, mrr_at_k

rec, rel, k = ["A", "B", "C", "D"], {"B", "D", "X"}, 4
assert math.isclose(recall_at_k(rec, rel, k), 2 / 3)
assert math.isclose(ndcg_at_k(rec, rel, k), 1.0616 / 2.1309, rel_tol=1e-3)
assert hit_at_k(rec, rel, k) == 1.0
assert mrr_at_k(rec, rel, k) == 0.5
assert math.isclose(ndcg_at_k(["B", "D", "X"], rel, 3), 1.0)   # perfect ranking
assert recall_at_k(["A", "C"], rel, 2) == 0.0                  # no hit
print("All metric tests passed.")
