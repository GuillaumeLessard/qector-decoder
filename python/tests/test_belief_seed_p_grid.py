"""Belief-matching proof (section 4): seed x p grid at d=5.

Pools logical errors over a {seed 0,1,2} x {p 0.004, 0.006} grid (<=400 shots
per cell) and asserts pooled-over-grid belief LER <= pooled PyMatching LER.
Pooling many small cells keeps the comparison statistically meaningful while
total shots stay small (QECTOR ``BeliefMatching`` rebuilds matching per shot).
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3.belief_matching import BeliefMatching


def _circuit(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def test_belief_seed_p_grid_pooled_le_pymatching():
    d, N = 5, 400
    seeds = [0, 1, 2]
    ps = [0.004, 0.006]

    # Cache decoders per p (they only depend on the DEM, not the seed).
    decoders = {}
    for p in ps:
        circ = _circuit(d, p)
        sdem = circ.detector_error_model(decompose_errors=True)
        bm = BeliefMatching.from_detector_error_model(sdem, max_iter=20)
        pm = pymatching.Matching.from_detector_error_model(sdem)
        decoders[p] = (circ, bm, pm, bm.num_observables)

    belief_total = 0
    pm_total = 0
    shots_total = 0

    for p in ps:
        circ, bm, pm, no = decoders[p]
        for seed in seeds:
            det, obs = circ.compile_detector_sampler(seed=seed).sample(shots=N, separate_observables=True)
            det = det.astype(np.uint8)
            obs = obs.astype(np.uint8)

            bpred = np.asarray(bm.decode_batch(det), np.uint8).reshape(N, -1)
            ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(N, -1)

            assert bpred.shape == (N, no)
            assert set(np.unique(bpred).tolist()).issubset({0, 1})

            be = int(np.any(bpred != obs, axis=1).sum())
            pe = int(np.any(ppred != obs, axis=1).sum())
            print(f"[grid] p={p} seed={seed} belief_err={be} pm_err={pe}")

            belief_total += be
            pm_total += pe
            shots_total += N

    belief_ler = belief_total / shots_total
    pm_ler = pm_total / shots_total
    print(
        f"[grid:pooled] cells={len(seeds) * len(ps)} shots={shots_total} "
        f"belief_err={belief_total} pm_err={pm_total} "
        f"belief_LER={belief_ler:.4f} pm_LER={pm_ler:.4f} "
        f"reduction={(pm_ler - belief_ler):.4f}"
    )

    # Pooled over the whole grid: belief at least as good (1-shot tie slack).
    assert belief_total <= pm_total + 1
