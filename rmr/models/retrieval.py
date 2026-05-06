"""Graph retrieval functions for anchor connecting subgraphs and expansion."""

from collections import deque
from typing import Dict, List, Set

import numpy as np
import scipy.sparse as sp


def bfs_multi_source(
    adj: sp.csr_matrix, sources: List[int]
) -> Dict[int, int]:
    """Run multi-source BFS to compute shortest distances.

    Args:
        adj: Sparse adjacency matrix in CSR format.
        sources: List of source node indices to start BFS from.

    Returns:
        Dictionary mapping visited node indices to their shortest distance
        from the nearest source.
    """
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        row_start = adj.indptr[u]
        row_end = adj.indptr[u + 1]
        for v in adj.indices[row_start:row_end]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def shortest_path_nodes(
    adj: sp.csr_matrix, start: int, end: int
) -> List[int]:
    """Return the list of nodes on a shortest path from start to end.

    Uses BFS to find a shortest path. If start equals end, returns a
    single-element list containing start.

    Args:
        adj: Sparse adjacency matrix in CSR format.
        start: Starting node index.
        end: Target node index.

    Returns:
        List of node indices representing a shortest path from start to
        end. Returns an empty list if no path exists.
    """
    if start == end:
        return [start]
    dist = {start: 0}
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        row_start = adj.indptr[u]
        row_end = adj.indptr[u + 1]
        for v in adj.indices[row_start:row_end]:
            if v not in dist:
                dist[v] = dist[u] + 1
                prev[v] = u
                q.append(v)
                if v == end:
                    path = [v]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
    return []


def acs(adj: sp.csr_matrix, anchors: List[int]) -> Set[int]:
    """Compute the Anchor Connecting Subgraph (ACS).

    Implements Algorithm 1 from the paper. Performs multi-source BFS from
    all anchors to find a connecting root, then returns the union of
    shortest paths from the root to each anchor.

    Args:
        adj: Sparse adjacency matrix in CSR format.
        anchors: List of anchor node indices.

    Returns:
        Set of node indices in the anchor connecting subgraph.
    """
    if not anchors:
        return set()
    if len(anchors) == 1:
        return {anchors[0]}
    dist = bfs_multi_source(adj, anchors)
    if len(dist) < len(anchors):
        return set(anchors)
    candidates = list(dist.keys())
    best_root = None
    best_score = float("inf")
    for cand in candidates:
        d_to_anchors = bfs_multi_source(adj, [cand])
        score = sum(
            d_to_anchors.get(a, float("inf")) for a in anchors
        )
        if score < best_score:
            best_score = score
            best_root = cand
    if best_root is None:
        return set(anchors)
    S = set()
    for a in anchors:
        path = shortest_path_nodes(adj, best_root, a)
        S.update(path)
    return S


def relevance_score(
    i: int,
    v: int,
    features: Dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    """Compute the relevance score r(i, v) between two nodes.

    Args:
        i: Index of the first node (query node).
        v: Index of the second node.
        features: Dictionary mapping modality names to feature arrays.
            Each array should be indexable by node index.
        mask: Binary mask of shape (num_nodes, num_modalities)
            indicating which modalities are available for each node.

    Returns:
        The relevance score as a float. Returns 0.0 if no shared
        modalities have positive masks for both nodes.
    """
    mods = list(features.keys())
    num = 0.0
    den = 0.0
    for m in mods:
        e_i = mask[i, list(features.keys()).index(m)]
        e_v = mask[v, list(features.keys()).index(m)]
        if e_i > 0 and e_v > 0:
            num += np.dot(features[m][i], features[m][v])
            den += 1.0
    return num / den if den > 0 else 0.0


def mean_relevance(
    i: int,
    S: Set[int],
    features: Dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    """Compute the mean relevance of node i over a set of nodes S.

    Args:
        i: Index of the query node.
        S: Set of node indices over which to average relevance.
        features: Dictionary mapping modality names to feature arrays.
        mask: Binary mask indicating available modalities per node.

    Returns:
        Mean relevance score. Returns 0.0 if S is empty or only
        contains i.
    """
    vals = [
        relevance_score(i, v, features, mask) for v in S if v != i
    ]
    return float(np.mean(vals)) if vals else 0.0


def mage(
    adj: sp.csr_matrix,
    anchors: List[int],
    query_item: int,
    features: Dict[str, np.ndarray],
    mask: np.ndarray,
    T: int = 10,
) -> Set[int]:
    """Run Modality-Aware Graph Expansion (MAGE).

    Implements Algorithm 2 from the paper. Greedily adds or removes
    boundary nodes to maximize mean relevance while preserving
    connectivity and all anchors.

    Args:
        adj: Sparse adjacency matrix in CSR format.
        anchors: List of anchor node indices that must be preserved.
        query_item: Index of the query item node.
        features: Dictionary mapping modality names to feature arrays.
        mask: Binary mask indicating available modalities per node.
        T: Maximum number of greedy iterations.

    Returns:
        Set of node indices in the expanded subgraph.
    """
    S = acs(adj, anchors)
    if not S:
        S = set(anchors)
    S = set(S)

    neighbors = {}
    for u in range(adj.shape[0]):
        row_start = adj.indptr[u]
        row_end = adj.indptr[u + 1]
        neighbors[u] = set(adj.indices[row_start:row_end])

    def boundary(S: Set[int]) -> Set[int]:
        b = set()
        for u in S:
            b.update(neighbors[u] - S)
        return b

    def is_connected(subset: Set[int]) -> bool:
        if not subset:
            return True
        start = next(iter(subset))
        visited = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in neighbors[u] & subset:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return visited == subset

    best_score = mean_relevance(query_item, S, features, mask)
    for _ in range(T):
        changed = False
        B = boundary(S)
        for v in B:
            new_S = S | {v}
            if not is_connected(new_S):
                continue
            score = mean_relevance(query_item, new_S, features, mask)
            if score > best_score:
                S = new_S
                best_score = score
                changed = True
                break
        if changed:
            continue
        for v in list(S):
            if v in anchors:
                continue
            new_S = S - {v}
            if not is_connected(new_S):
                continue
            score = mean_relevance(query_item, new_S, features, mask)
            if score > best_score:
                S = new_S
                best_score = score
                changed = True
                break
        if not changed:
            break
    return S