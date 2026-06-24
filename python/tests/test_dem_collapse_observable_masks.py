"""DEM collapse: observable masks must never be merged incorrectly.

Rules locked here:
  * parallel mechanisms with the SAME observable mask keep that mask;
  * parallel mechanisms with DIFFERENT masks keep the lowest-weight (most likely)
    member's mask (PyMatching semantics) — masks are NOT XOR'd together;
  * mechanisms on different detector sets keep their own masks;
  * boundary edges (single detector) and hyperedges retain masks.
"""
import numpy as np

from qector_decoder_v3 import dem


def _edge(model, detset):
    for e in model.errors:
        if e.detectors == tuple(sorted(detset)):
            return e
    raise AssertionError(f"no edge {detset}")


def test_same_pair_same_mask_preserved():
    m = dem.parse_dem("error(0.1) D0 D1 L0\nerror(0.2) D0 D1 L0\n")
    c = m.collapse_to_graph()
    assert c.num_errors == 1
    assert _edge(c, (0, 1)).observables == (0,)


def test_same_pair_different_mask_keeps_most_likely():
    # second mechanism (p=0.2) is more likely -> lower weight -> its mask (L0) wins
    m = dem.parse_dem("error(0.1) D0 D1\nerror(0.2) D0 D1 L0\n")
    c = m.collapse_to_graph()
    assert c.num_errors == 1
    assert _edge(c, (0, 1)).observables == (0,)


def test_masks_are_not_xored_together():
    # if masks were XOR-merged, L0 ^ L1 would give {0,1}; correct behaviour keeps
    # only the most-likely member's mask.
    m = dem.parse_dem("error(0.3) D0 D1 L0\nerror(0.1) D0 D1 L1\n")
    c = m.collapse_to_graph()
    obs = _edge(c, (0, 1)).observables
    assert obs == (0,)            # most likely (p=0.3) carried L0
    assert obs != (0, 1)          # NOT the XOR of both masks


def test_different_pairs_keep_separate_masks():
    m = dem.parse_dem("error(0.1) D0 D1 L0\nerror(0.1) D1 D2 L1\n")
    c = m.collapse_to_graph()
    assert _edge(c, (0, 1)).observables == (0,)
    assert _edge(c, (1, 2)).observables == (1,)


def test_observables_matrix_consistent_after_collapse():
    m = dem.parse_dem(
        "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.2) D0 D1 L0\nerror(0.1) D1 L0\n")
    c = m.collapse_to_graph()
    L = c.observables_matrix()
    assert L.shape == (1, c.num_errors)
    # column for the merged D0-D1 edge flips L0 (the more-likely member had L0)
    cols = {e.detectors: j for j, e in enumerate(c.errors)}
    assert L[0, cols[(0, 1)]] == 1


def test_boundary_edge_mask_retained():
    m = dem.parse_dem("error(0.1) D0 L0\nerror(0.3) D0 L0\n")
    c = m.collapse_to_graph()
    assert _edge(c, (0,)).observables == (0,)
