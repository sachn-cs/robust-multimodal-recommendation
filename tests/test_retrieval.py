import numpy as np
import scipy.sparse as sp
from rmr.models.retrieval import (
    bfs_multi_source,
    shortest_path_nodes,
    acs,
    mage,
    relevance_score,
)


def test_acs_basic():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    anchors = [0, 3]
    S = acs(adj, anchors)
    assert 0 in S and 3 in S
    # Path 0-1-2-3 should be included
    assert len(S) >= 4


def test_mage_returns_anchors():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    features = {
        "visual": np.eye(4, dtype=np.float32),
    }
    mask = np.ones((4, 1), dtype=np.float32)
    S = mage(adj, [0, 3], 0, features, mask, T=2)
    assert 0 in S and 3 in S


def test_bfs_multi_source_exact_distances():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    dist = bfs_multi_source(adj, [0])
    assert dist[0] == 0
    assert dist[1] == 1
    assert dist[2] == 2
    assert dist[3] == 3

    dist2 = bfs_multi_source(adj, [0, 3])
    assert dist2[0] == 0
    assert dist2[3] == 0
    assert dist2[1] == 1  # nearest to 0
    assert dist2[2] == 1  # nearest to 3


def test_shortest_path_nodes_start_eq_end():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32))
    path = shortest_path_nodes(adj, 1, 1)
    assert path == [1]


def test_shortest_path_nodes_no_path():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=np.float32))
    path = shortest_path_nodes(adj, 0, 2)
    assert path == []


def test_relevance_score_exact_value():
    features = {
        "visual": np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32),
    }
    mask = np.ones((3, 1), dtype=np.float32)
    # dot([1,0], [0,1]) == 0
    assert relevance_score(0, 1, features, mask) == 0.0
    # dot([1,0], [1,1]) == 1
    assert relevance_score(0, 2, features, mask) == 1.0
    # dot([1,1], [1,1]) == 2
    assert relevance_score(2, 2, features, mask) == 2.0


def test_acs_empty_anchors():
    adj = sp.csr_matrix(np.array([
        [0, 1],
        [1, 0],
    ], dtype=np.float32))
    S = acs(adj, [])
    assert S == set()


def test_acs_single_anchor():
    adj = sp.csr_matrix(np.array([
        [0, 1],
        [1, 0],
    ], dtype=np.float32))
    S = acs(adj, [1])
    assert S == {1}


def test_acs_disconnected_graph():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    anchors = [0, 3]
    S = acs(adj, anchors)
    # In a disconnected graph, not all anchors are reachable from each other
    # The code returns set(anchors) when len(dist) < len(anchors)
    assert S == set(anchors)


def test_mage_T_zero():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    features = {
        "visual": np.eye(4, dtype=np.float32),
    }
    mask = np.ones((4, 1), dtype=np.float32)
    S = mage(adj, [0, 3], 0, features, mask, T=0)
    # With T=0, the loop doesn't run, so S should be the ACS
    expected = acs(adj, [0, 3])
    assert S == expected
