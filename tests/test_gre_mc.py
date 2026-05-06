import torch
import scipy.sparse as sp
import numpy as np
from rmr.models.gre_mc import GREMC


def test_gre_mc_forward():
    # Use 5-node graph so Laplacian PE (k=4) has enough eigenvectors
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=np.float32))
    model = GREMC(
        input_dims={"visual": 10, "text": 8},
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        codebook_size=5,
        top_p=2,
        tau=0.5,
        pe_dim=4,
    )
    features = {
        "visual": torch.randn(5, 10),
        "text": torch.randn(5, 8),
    }
    mask = torch.ones(5, 2)
    out, g = model(features, mask, adj)
    assert out["visual"].shape == (5, 10)
    assert out["text"].shape == (5, 8)


def test_gre_mc_training_false():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=np.float32))
    model = GREMC(
        input_dims={"visual": 10, "text": 8},
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        codebook_size=5,
        top_p=2,
        tau=0.5,
        pe_dim=4,
        dropout=0.0,
    )
    model.eval()
    features = {
        "visual": torch.randn(5, 10),
        "text": torch.randn(5, 8),
    }
    mask = torch.ones(5, 2)
    out1, g1 = model(features, mask, adj, training=False)
    out2, g2 = model(features, mask, adj, training=False)
    # Without Gumbel noise and with dropout disabled, outputs should be deterministic
    for key in out1:
        torch.testing.assert_close(out1[key], out2[key], atol=1e-6, rtol=0)
    torch.testing.assert_close(g1, g2, atol=1e-6, rtol=0)


def test_gre_mc_index_parameter():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=np.float32))
    model = GREMC(
        input_dims={"visual": 10, "text": 8},
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        codebook_size=5,
        top_p=2,
        tau=0.5,
        pe_dim=4,
    )
    features = {
        "visual": torch.randn(3, 10),
        "text": torch.randn(3, 8),
    }
    mask = torch.ones(3, 2)
    index = torch.tensor([0, 2, 4])
    out, g = model(features, mask, adj, index=index, training=True)
    assert out["visual"].shape == (3, 10)
    assert out["text"].shape == (3, 8)
    assert g.shape == (3, 5)
