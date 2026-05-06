"""Recommendation evaluation metrics."""

import numpy as np


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Compute mean Recall@K.

    Recall@K measures the fraction of relevant items that appear in the
    top-K recommendations for each user.

    Args:
        scores: Predicted relevance scores of shape (n_users, n_items).
        labels: Binary relevance matrix of shape (n_users, n_items).
        k: Number of top items to consider.

    Returns:
        Mean Recall@K across all users with at least one relevant item.
    """
    n_users = scores.shape[0]
    recalls = []
    for u in range(n_users):
        top_k = np.argsort(-scores[u])[:k]
        n_relevant = labels[u].sum()
        if n_relevant == 0:
            continue
        n_hit = labels[u][top_k].sum()
        recalls.append(n_hit / n_relevant)
    return float(np.mean(recalls)) if recalls else 0.0


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K.

    Args:
        relevances: Relevance scores for items, in the order they were
            ranked.
        k: Maximum rank to include in the DCG sum.

    Returns:
        The DCG@K value.
    """
    relevances = np.asarray(relevances)[:k]
    positions = np.arange(1, len(relevances) + 1)
    return np.sum(relevances / np.log2(positions + 1))


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Compute mean Normalized Discounted Cumulative Gain at rank K.

    NDCG@K compares the DCG of the predicted ranking to the ideal DCG
    for each user.

    Args:
        scores: Predicted relevance scores of shape (n_users, n_items).
        labels: Binary relevance matrix of shape (n_users, n_items).
        k: Number of top items to consider.

    Returns:
        Mean NDCG@K across all users with a non-zero ideal DCG.
    """
    n_users = scores.shape[0]
    ndcgs = []
    for u in range(n_users):
        top_k = np.argsort(-scores[u])[:k]
        relevances = labels[u][top_k]
        ideal = np.sort(labels[u])[::-1][:k]
        dcg = dcg_at_k(relevances, k)
        idcg = dcg_at_k(ideal, k)
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0
