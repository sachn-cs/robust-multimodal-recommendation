"""Masking utilities for multimodal item representations."""

import numpy as np


def apply_modality_mask(
    n_items: int,
    n_modalities: int,
    mask_ratio: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    """Randomly mask modalities so that each item retains at least one.

    Args:
        n_items: Number of items to mask.
        n_modalities: Number of modalities per item.
        mask_ratio: Probability of masking a single modality.
        seed: Random seed for reproducibility.

    Returns:
        A binary array of shape (n_items, n_modalities) with dtype float32.
        A value of ``1.0`` means the modality is *kept*; ``0.0`` means it is
        masked.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random((n_items, n_modalities)) > mask_ratio
    for i in range(n_items):
        if mask[i].sum() == 0:
            keep = rng.integers(0, n_modalities)
            mask[i, keep] = True
    return mask.astype(np.float32)
