"""CLI script to evaluate recommendation predictions."""

import argparse

import numpy as np

from rmr.evaluation.evaluator import Evaluator


def main() -> None:
    """Run evaluation on saved predictions and labels.

    Parses command-line arguments, loads NumPy arrays, and prints
    Recall@K and NDCG@K for K in {10, 20}.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate recommendation predictions."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a NumPy file containing predicted scores.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to a NumPy file containing binary relevance labels.",
    )
    args = parser.parse_args()

    scores = np.load(args.predictions)
    labels = np.load(args.labels)
    ev = Evaluator(ks=[10, 20])
    res = ev.evaluate(scores, labels)
    for k, v in res.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
