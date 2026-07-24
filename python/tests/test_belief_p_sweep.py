"""Belief-matching proof (section 4): physical error-rate sweep at d=5.

At each p in {0.004, 0.006, 0.008} (fixed seed, <=500 shots) asserts belief is
never *materially* worse than PyMatching, and pooled-over-p belief LER <=
pooled PyMatching LER. Distance fixed to 5; shots kept small because QECTOR
``BeliefMatching`` rebuilds the weighted matching per shot.
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


def test_belief_p_sweep_not_worse_and_pooled_le():
    d, N, seed = 5, 500, 7
    ps = [0.004, 0.006, 0.008]

    belief_total = 0
    pm_total = 0
    shots_total = 0

    for p in ps:
        circ = _circuit(d, p)
        sdem = circ.detector_error_model(decompose_errors=True)
        bm = BeliefMatching.from_detector_error_model(sdem, max_iter=20)
        pm = pymatching.Matching.from_detector_error_model(sdem)
        no = bm.num_observables

        det, obs = circ.compile_detector_sampler(seed=seed).sample(shots=N, separate_observables=True)
        det = det.astype(np.uint8)
        obs = obs.astype(np.uint8)

        bpred = np.asarray(bm.decode_batch(det), np.uint8).reshape(N, -1)
        ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(N, -1)

        assert bpred.shape == (N, no)
        assert set(np.unique(bpred).tolist()).issubset({0, 1})

        belief_err = int(np.any(bpred != obs, axis=1).sum())
        pm_err = int(np.any(ppred != obs, axis=1).sum())
        print(
            f"[p_sweep] p={p} shots={N} belief_err={belief_err} pm_err={pm_err} "
            f"belief_LER={belief_err / N:.4f} pm_LER={pm_err / N:.4f}"
        )

        # Belief must never be materially worse than PyMatching at any p.
        slack = max(3, int(0.15 * pm_err))
        assert belief_err <= pm_err + slack

        belief_total += belief_err
        pm_total += pm_err
        shots_total += N

    belief_ler = belief_total / shots_total
    pm_ler = pm_total / shots_total
    print(
        f"[p_sweep:pooled] shots={shots_total} belief_err={belief_total} "
        f"pm_err={pm_total} belief_LER={belief_ler:.4f} pm_LER={pm_ler:.4f} "
        f"reduction={(pm_ler - belief_ler):.4f}"
    )

    # Pooled over all p: belief at least as good as PyMatching (1-shot tie slack).
    assert belief_total <= pm_total + 1
