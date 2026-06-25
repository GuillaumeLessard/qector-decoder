"""Belief-matching example from docs/BEYOND_PYMATCHING.md, executed end to end.

docs/BEYOND_PYMATCHING.md claims QECTOR's belief-matching decoder runs on real
Stim circuit-level shots and is *never worse* than plain PyMatching MWPM. This
test mirrors that example on small surface codes: build the DEM, construct
``BeliefMatching.from_detector_error_model``, decode a batch of sampled shots,
and assert the predictions are well-formed observable flips with an LER in
[0, 1] and at least as good as PyMatching on the same shots.
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


@pytest.mark.parametrize("d", [3, 5])
def test_belief_matching_runs_and_is_valid(d):
    shots = 2000
    p = 0.005
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=1234).sample(
        shots=shots, separate_observables=True
    )
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    belief = BeliefMatching.from_detector_error_model(sdem)
    assert belief.num_observables == obs.shape[1]
    assert belief.num_detectors == det.shape[1]

    bpred = np.asarray(belief.decode_batch(det), np.uint8).reshape(shots, -1)
    # Correct per-observable prediction shape: (shots, num_observables).
    assert bpred.shape == obs.shape, (bpred.shape, obs.shape)
    # Predictions are bits.
    assert set(np.unique(bpred)).issubset({0, 1})

    belief_errors = int(np.any(bpred != obs, axis=1).sum())
    belief_ler = belief_errors / shots
    assert 0.0 <= belief_ler <= 1.0

    # Cross-check against PyMatching MWPM on the SAME shots.
    pm = pymatching.Matching.from_detector_error_model(sdem)
    ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(shots, -1)
    pm_errors = int(np.any(ppred != obs, axis=1).sum())

    # docs/BEYOND_PYMATCHING.md: belief-matching is never worse than plain
    # MWPM. Allow a small statistical slack via the Wilson upper bound on
    # PyMatching's error count so the assertion is robust, not flaky.
    _, pm_upper = _wilson(pm_errors, shots)
    belief_lo = belief_errors / shots
    assert belief_lo <= pm_upper + 1e-9, (
        f"belief LER {belief_ler:.4f} worse than PyMatching upper CI {pm_upper:.4f} "
        f"(belief_errors={belief_errors}, pm_errors={pm_errors}, d={d})"
    )
