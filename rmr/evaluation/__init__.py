"""Evaluation package for GRE-MC recommendation metrics.

This package provides recommendation evaluation metrics (Recall@K,
NDCG@K) and an :class:`Evaluator` helper that computes them for a set
of cutoffs.
"""

from rmr.evaluation.evaluator import Evaluator
from rmr.evaluation.metrics import dcg_at_k, ndcg_at_k, recall_at_k

__all__ = [
    "Evaluator",
    "dcg_at_k",
    "ndcg_at_k",
    "recall_at_k",
]
