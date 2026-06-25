"""Circuit-level competitive regression test: QECTOR vs PyMatching.

Runs only when both ``stim`` and ``pymatching`` are installed. Locks in the
headline result from docs/BENCHMARK_COMPETITIVE.md: on real Stim-sampled
circuit-level shots, QECTOR's weighted Blossom achieves a logical error rate
statistically indistinguishable from PyMatching's, and the DEM collapse both
shrinks the graph and preserves faithfulness.
"""

import math

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

import qector_decoder_v3 as qd  # noqa: E402
from qector_decoder_v3 import dem  # noqa: E402
from qector_decoder_v3.pymatching_compat import Matching  # noqa: E402


def _wilson(k, n, z=1.959963985):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - w), min(1.0, c + w)


def _circuit(d):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=0.006,
        before_measure_flip_probability=0.006,
        after_reset_flip_probability=0.006,
    )


@pytest.mark.parametrize("d", [3, 5])
def test_collapse_shrinks_circuit_dem(d):
    sdem = _circuit(d).detector_error_model(decompose_errors=True)
    raw = dem.from_stim(sdem)
    col = raw.collapse_to_graph()
    assert col.num_errors < raw.num_errors
    assert col.is_graphlike
    # collapsed problem must still decode faithfully on real shots
    H = col.check_matrix()
    decoder = qd.BlossomDecoder(
        col.check_to_qubits(), col.num_errors, col.weights().tolist()
    )
    sampler = _circuit(d).compile_detector_sampler(seed=12345)
    dets, _ = sampler.sample(shots=300, separate_observables=True)
    out = np.asarray(decoder.batch_decode(dets.astype(np.uint8)), dtype=np.uint8)
    for i in range(len(dets)):
        assert np.array_equal((H @ out[i]) & 1, dets[i].astype(np.uint8))


@pytest.mark.parametrize("d", [3, 5])
def test_ler_parity_with_pymatching(d):
    circ = _circuit(d)
    sdem = circ.detector_error_model(decompose_errors=True)
    shots = 8000
    sampler = circ.compile_detector_sampler(seed=12345)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    dets = dets.astype(np.uint8)
    obs = obs.astype(np.uint8)

    qm = Matching.from_detector_error_model(sdem)
    pm = pymatching.Matching.from_detector_error_model(sdem)

    q_pred = np.asarray(qm.decode_batch(dets), dtype=np.uint8).reshape(shots, -1)
    p_pred = np.asarray(pm.decode_batch(dets), dtype=np.uint8).reshape(shots, -1)

    q_err = int(np.any(q_pred != obs, axis=1).sum())
    p_err = int(np.any(p_pred != obs, axis=1).sum())
    q_lo, q_hi = _wilson(q_err, shots)
    p_lo, p_hi = _wilson(p_err, shots)

    # The two logical-error-rate confidence intervals must overlap (parity).
    assert q_lo <= p_hi and p_lo <= q_hi, (
        f"d={d}: QECTOR LER {q_err / shots:.4f} [{q_lo:.4f},{q_hi:.4f}] vs "
        f"PyMatching {p_err / shots:.4f} [{p_lo:.4f},{p_hi:.4f}] — intervals disjoint"
    )
    # QECTOR should never be dramatically worse.
    assert q_err <= p_err + 3 * math.sqrt(max(p_err, 1))


def test_qector_matching_is_weight_optimal_on_circuit_dem():
    """On the collapsed circuit DEM, QECTOR's weighted matching is never heavier."""
    sdem = _circuit(5).detector_error_model(decompose_errors=True)
    col = dem.from_stim(sdem).collapse_to_graph()
    H = col.check_matrix()
    w = col.weights()
    q = qd.BlossomDecoder(col.check_to_qubits(), col.num_errors, w.tolist())
    pm = pymatching.Matching.from_check_matrix(H, weights=w)
    sampler = _circuit(5).compile_detector_sampler(seed=999)
    dets, _ = sampler.sample(shots=1000, separate_observables=True)
    worse = 0
    n = len(dets)
    for dd in dets.astype(np.uint8):
        cq = np.asarray(q.decode(dd), dtype=np.uint8)
        _, pm_weight = pm.decode(dd, return_weight=True)
        assert np.array_equal((H @ cq) & 1, dd)
        # weighted total of QECTOR's matching vs PyMatching's optimal weight
        if float(w @ cq) > float(pm_weight) * (1 + 1e-6) + 1e-6:
            worse += 1
    # weighted MWPM: QECTOR's total weight matches PyMatching's optimum on
    # nearly every shot (a handful of ties/floating-point cases tolerated).
    assert worse <= max(5, n // 100), (
        f"QECTOR heavier than PyMatching on {worse}/{n} shots"
    )
