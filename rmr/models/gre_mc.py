"""End-to-end GRE-MC model.

Composes a Laplacian positional encoding module, a joint-encoding graph
transformer, a sparse-routing codebook, and per-modality decoders to
reconstruct missing modalities.
"""

from typing import Dict, Optional

import scipy.sparse as sp
import torch
import torch.nn as nn

from rmr.models.codebook import SparseRoutingCodebook
from rmr.models.decoder import ModalityDecoders
from rmr.models.positional_encoding import LaplacianPE
from rmr.models.transformer import JointEncodingGraphTransformer


class GREMC(nn.Module):
    """End-to-end GRE-MC model: transformer -> codebook -> decoders."""

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        codebook_size: int = 100,
        top_p: int = 4,
        tau: float = 0.5,
        pe_dim: int = 20,
        dropout: float = 0.5,
    ) -> None:
        """Initializes the GRE-MC model.

        Args:
            input_dims: Mapping from modality name to feature dimension.
            hidden_dim: Hidden dimensionality.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            codebook_size: Number of codebook entries.
            top_p: Number of top codebook entries to select.
            tau: Gumbel-Softmax temperature.
            pe_dim: Dimensionality of Laplacian positional encodings.
            dropout: Dropout probability.
        """
        super().__init__()
        self.pe_encoder = LaplacianPE(k=pe_dim)
        self.transformer = JointEncodingGraphTransformer(
            input_dims=input_dims,
            pe_dim=pe_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.codebook = SparseRoutingCodebook(
            input_dim=hidden_dim,
            codebook_size=codebook_size,
            top_p=top_p,
            tau=tau,
        )
        self.decoder = ModalityDecoders(
            latent_dim=hidden_dim,
            output_dims=input_dims,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        adjacency: sp.spmatrix,
        index: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> tuple:
        """Forward pass through GRE-MC.

        Args:
            features: Dict of tensors, each with shape (B, d_m).
            mask: Binary indicator matrix of shape (B, M).
            adjacency: Sparse adjacency matrix for the item graph.
            index: Optional item indices of shape (B,) to index cached PE.
                If None, uses the first N rows of the full PE matrix.
            training: Whether to apply Gumbel noise in the codebook.

        Returns:
            Tuple of (predictions, routing_weights) where predictions is a
            dict of tensors with the same shapes as ``features``, and
            routing_weights has shape (B, codebook_size).
        """
        pe = self.pe_encoder(adjacency)
        if index is not None:
            pe_batch = pe[index]
        else:
            n = mask.shape[0]
            pe_batch = pe[:n]
        z = self.transformer(features, mask, pe_batch)
        q, g = self.codebook(z, training=training)
        x_hat = self.decoder(q)
        return x_hat, g
