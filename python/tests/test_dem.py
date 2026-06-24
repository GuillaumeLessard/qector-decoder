"""Tests for qector_decoder_v3.dem — the Stim DEM loader.

These run without Stim installed: DEM text is parsed directly.  They lock in the
correct detector-graph semantics (mechanisms = columns, detectors = rows) that
replaced the earlier broken stim_compat heuristic.
"""
import numpy as np

from qector_decoder_v3 import dem


REP_DEM = """
error(0.1) D0 L0
error(0.1) D0 D1
error(0.1) D1 D2
error(0.1) D2 L0
"""


def test_basic_parse():
    m = dem.parse_dem(REP_DEM)
    assert m.num_errors == 4
    assert m.num_detectors == 3
    assert m.num_observables == 1
    assert m.is_graphlike


def test_check_matrix_semantics():
    m = dem.parse_dem(REP_DEM)
    H = m.check_matrix()
    assert H.shape == (3, 4)  # detectors x mechanisms
    # mechanism 0 flips detector 0 only; mechanism 1 flips D0,D1; etc.
    assert H[:, 0].tolist() == [1, 0, 0]
    assert H[:, 1].tolist() == [1, 1, 0]
    assert H[:, 2].tolist() == [0, 1, 1]
    assert H[:, 3].tolist() == [0, 0, 1]


def test_observables_matrix_and_prediction():
    m = dem.parse_dem(REP_DEM)
    L = m.observables_matrix()
    assert L.shape == (1, 4)
    assert L[0].tolist() == [1, 0, 0, 1]  # mechanisms 0 and 3 flip L0
    corr = np.array([1, 0, 0, 0], dtype=np.uint8)
    assert m.predicted_observables(corr).tolist() == [1]


def test_weights_monotone_in_probability():
    m = dem.parse_dem("error(0.01) D0\nerror(0.4) D0 D1\n")
    w = m.weights()
    # lower probability -> higher matching weight
    assert w[0] > w[1] > 0


def test_repeat_and_shift_detectors_flatten():
    text = """
    error(0.1) D0 D1
    repeat 3 {
        error(0.05) D0 D1
        shift_detectors 1
    }
    """
    m = dem.parse_dem(text)
    # 1 base + 3 repeated = 4 mechanisms
    assert m.num_errors == 4
    # detectors used: base D0,D1; iter0 D0,D1; iter1 D1,D2; iter2 D2,D3 -> max det 3
    assert m.num_detectors == 4
    H = m.check_matrix()
    assert H.shape == (4, 4)


def test_decomposition_caret_splits_components():
    m = dem.parse_dem("error(0.1) D0 D1 ^ D2 D3\n")
    assert m.num_errors == 2
    assert m.is_graphlike
    H = m.check_matrix()
    assert H[:, 0].tolist() == [1, 1, 0, 0]
    assert H[:, 1].tolist() == [0, 0, 1, 1]


def test_hyperedge_not_graphlike():
    m = dem.parse_dem("error(0.1) D0 D1 D2\n")
    assert not m.is_graphlike
    assert m.num_errors == 1


def test_make_decoder_is_faithful():
    m = dem.parse_dem(REP_DEM)
    H = m.check_matrix()
    for kind in ("union_find", "blossom", "sparse_blossom"):
        dec = m.make_decoder(kind)
        for s in [[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]]:
            sy = np.array(s, dtype=np.uint8)
            c = np.asarray(dec.decode(sy)).astype(np.uint8)
            assert np.array_equal((H @ c) & 1, sy), kind


def test_to_code_carries_observables():
    m = dem.parse_dem(REP_DEM)
    code = m.to_code()
    assert code.n_qubits == 4
    assert code.n_checks == 3
    assert "observables_matrix" in code._meta


def test_load_dem_file(tmp_path):
    p = tmp_path / "rep.dem"
    p.write_text(REP_DEM, encoding="utf-8")
    m = dem.load_dem_file(str(p))
    assert m.num_errors == 4


def test_stim_compat_reroute_matches_dem():
    from qector_decoder_v3 import stim_compat

    c2q, nq = stim_compat.from_stim_detector_error_model(REP_DEM)
    m = dem.parse_dem(REP_DEM)
    assert nq == m.num_errors
    assert c2q == m.check_to_qubits()


def test_collapse_merges_parallel_edges():
    text = (
        "error(0.1) D0 D1\n"
        "error(0.2) D0 D1 L0\n"   # parallel to the first, more likely, flips L0
        "error(0.1) D1 D2\n"
    )
    m = dem.parse_dem(text)
    assert m.num_errors == 3
    c = m.collapse_to_graph()
    assert c.num_errors == 2  # the two D0-D1 mechanisms merged
    # combined probability for the merged edge > each individual probability
    sigs = {e.detectors: e for e in c.errors}
    merged = sigs[(0, 1)]
    assert merged.probability > 0.2
    assert merged.observables == (0,)  # keeps the lower-weight (more likely) member's obs


def test_collapse_preserves_faithfulness_and_observables():
    # repetition-style graphlike DEM with duplicated mechanisms
    lines = ["error(0.05) D0 L0"]
    for i in range(7):
        lines.append(f"error(0.05) D{i} D{i+1}")
        lines.append(f"error(0.03) D{i} D{i+1}")  # parallel duplicate
    lines.append("error(0.05) D7 L0")
    m = dem.parse_dem("\n".join(lines))
    c = m.collapse_to_graph()
    assert c.num_errors < m.num_errors
    Hc = c.check_matrix()
    dec = c.make_decoder("blossom")
    rng = np.random.default_rng(0)
    for _ in range(100):
        err = (rng.random(c.num_errors) < 0.1).astype(np.uint8)
        s = (Hc @ err) & 1
        corr = np.asarray(dec.decode(s.astype(np.uint8))).astype(np.uint8)
        assert np.array_equal((Hc @ corr) & 1, s)


def test_collapse_idempotent_on_graphlike():
    m = dem.parse_dem(REP_DEM)
    c1 = m.collapse_to_graph()
    c2 = c1.collapse_to_graph()
    assert c1.num_errors == c2.num_errors


def test_surface_code_dem_roundtrip_is_decodable():
    # Build a repetition-style graphlike DEM programmatically and decode it.
    lines = ["error(0.05) D0 L0"]
    d = 8
    for i in range(d - 1):
        lines.append(f"error(0.05) D{i} D{i+1}")
    lines.append(f"error(0.05) D{d-2} L0")
    m = dem.parse_dem("\n".join(lines))
    H = m.check_matrix()
    dec = m.make_decoder("sparse_blossom")
    rng = np.random.default_rng(1)
    for _ in range(100):
        err = (rng.random(m.num_errors) < 0.1).astype(np.uint8)
        s = (H @ err) & 1
        c = np.asarray(dec.decode(s.astype(np.uint8))).astype(np.uint8)
        assert np.array_equal((H @ c) & 1, s)
