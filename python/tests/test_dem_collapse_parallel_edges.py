"""DEM collapse: parallel edges merge to one edge per detector signature.

Covers: many parallel mechanisms -> one edge; hyperedges (>2 detectors) pass
through unchanged; boundary edges (single detector) handled; idempotence; and
that collapse preserves syndrome-faithfulness of the resulting matching graph.
"""
import numpy as np

from qector_decoder_v3 import dem


def test_parallel_edges_collapse_to_one_per_signature():
    text = "\n".join([
        "error(0.1) D0 D1", "error(0.1) D0 D1", "error(0.1) D0 D1",
        "error(0.1) D1 D2", "error(0.1) D1 D2",
        "error(0.1) D2 D3",
    ])
    m = dem.parse_dem(text)
    assert m.num_errors == 6
    c = m.collapse_to_graph()
    assert c.num_errors == 3  # {D0D1, D1D2, D2D3}
    sigs = sorted(e.detectors for e in c.errors)
    assert sigs == [(0, 1), (1, 2), (2, 3)]


def test_hyperedges_pass_through_undecomposed():
    # Distinct hyperedges (>2 detectors) are passed through *undecomposed* — they
    # keep all their detectors rather than being split into graph edges. Parallel
    # hyperedges with the SAME detector set still merge (same signature), but
    # different ones remain separate hyperedges.
    m = dem.parse_dem("error(0.1) D0 D1 D2\nerror(0.1) D2 D3 D4\nerror(0.1) D5 D6\n")
    c = m.collapse_to_graph()
    hyper = [e for e in c.errors if len(e.detectors) > 2]
    assert len(hyper) == 2                       # both 3-detector mechanisms kept
    assert all(len(e.detectors) == 3 for e in hyper)
    assert sum(1 for e in c.errors if e.detectors == (5, 6)) == 1


def test_identical_hyperedges_merge():
    m = dem.parse_dem("error(0.1) D0 D1 D2\nerror(0.1) D0 D1 D2\n")
    c = m.collapse_to_graph()
    hyper = [e for e in c.errors if e.detectors == (0, 1, 2)]
    assert len(hyper) == 1                       # parallel identical hyperedges merge
    assert abs(hyper[0].probability - (0.1 * 0.9 + 0.1 * 0.9)) < 1e-12


def test_boundary_edges_collapse():
    m = dem.parse_dem("error(0.1) D0\nerror(0.1) D0\nerror(0.1) D1 D2\n")
    c = m.collapse_to_graph()
    assert sum(1 for e in c.errors if e.detectors == (0,)) == 1


def test_collapse_is_idempotent():
    text = "error(0.1) D0 D1\nerror(0.2) D0 D1\nerror(0.1) D1 D2\n"
    c1 = dem.parse_dem(text).collapse_to_graph()
    c2 = c1.collapse_to_graph()
    assert c1.num_errors == c2.num_errors
    assert sorted(e.detectors for e in c1.errors) == sorted(e.detectors for e in c2.errors)


def test_collapse_preserves_faithfulness():
    lines = ["error(0.05) D0 L0"]
    for i in range(9):
        lines.append(f"error(0.05) D{i} D{i+1}")
        lines.append(f"error(0.03) D{i} D{i+1}")   # parallel duplicate
    lines.append("error(0.05) D9 L0")
    c = dem.parse_dem("\n".join(lines)).collapse_to_graph()
    H = c.check_matrix()
    dec = c.make_decoder("blossom")
    rng = np.random.default_rng(0)
    for _ in range(200):
        err = (rng.random(c.num_errors) < 0.1).astype(np.uint8)
        s = (H @ err) & 1
        corr = np.asarray(dec.decode(s.astype(np.uint8)), np.uint8)
        assert np.array_equal((H @ corr) & 1, s)


def test_collapse_reduces_edge_count_on_circuit_like_dem():
    text_lines = []
    rng = np.random.default_rng(1)
    for _ in range(400):
        a = int(rng.integers(0, 20)); b = int(rng.integers(0, 20))
        if a == b:
            continue
        text_lines.append(f"error({float(rng.uniform(0.001,0.02)):.5f}) D{a} D{b}")
    m = dem.parse_dem("\n".join(text_lines))
    c = m.collapse_to_graph()
    assert c.num_errors < m.num_errors   # parallel edges genuinely merged
