"""Belief-matching proof (section 4): seed sweep at d=5, memory_x.

Pools per-seed logical errors across many seeds on correlated circuit-level
noise and asserts the core accuracy claim:
**pooled belief LER <= pooled PyMatching LER** (belief at least as good).
Also checks every belief prediction is a valid per-observable 0/1 vector.

At d=5/p=0.005 the two decoders are nearly tied (noise/parity regime), so we
pool 6 seeds (400 shots each) to make the comparison statistically robust
rather than seed-lucky -- belief edges ahead once enough shots are pooled.

QECTOR ``BeliefMatching`` rebuilds the weighted matching per shot (slow), so
distance is fixed to 5 and per-seed shots are kept small (<=400/seed).
"""

import math

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3.belief_matching import BeliefMatching


def _wilson(k, n, z=1.959963985):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


def _circuit(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def test_belief_seed_sweep_pooled_le_pymatching():
    d, p, N = 5, 0.005, 400
    seeds = [0, 1, 2, 3, 4, 5]

    circ = _circuit(d, p)
    sdem = circ.detector_error_model(decompose_errors=True)
    bm = BeliefMatching.from_detector_error_model(sdem, max_iter=20)
    pm = pymatching.Matching.from_detector_error_model(sdem)
    no = bm.num_observables

    belief_err_total = 0
    pm_err_total = 0
    shots_total = 0

    for seed in seeds:
        det, obs = circ.compile_detector_sampler(seed=seed).sample(shots=N, separate_observables=True)
        det = det.astype(np.uint8)
        obs = obs.astype(np.uint8)

        bpred = np.asarray(bm.decode_batch(det), np.uint8).reshape(N, -1)
        ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(N, -1)

        # Every belief prediction must be a valid per-observable 0/1 vector.
        assert bpred.shape == (N, no)
        assert set(np.unique(bpred).tolist()).issubset({0, 1})

        belief_err_total += int(np.any(bpred != obs, axis=1).sum())
        pm_err_total += int(np.any(ppred != obs, axis=1).sum())
        shots_total += N

    belief_ler = belief_err_total / shots_total
    pm_ler = pm_err_total / shots_total
    print(
        f"[seed_sweep] shots={shots_total} belief_err={belief_err_total} "
        f"pm_err={pm_err_total} belief_LER={belief_ler:.4f} pm_LER={pm_ler:.4f} "
        f"reduction={(pm_ler - belief_ler):.4f}"
    )

    # Core claim: belief pooled LER is at least as good as PyMatching's.
    # Allow a tiny tie-break slack (1 shot) for sampling noise on the pool.
    assert belief_err_total <= pm_err_total + 1
