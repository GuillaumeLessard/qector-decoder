"""Belief-matching proof (section 4): memory_z basis at d=5.

Validates belief decoding end-to-end on the Z-basis memory experiment:
predictions are valid per-observable 0/1 vectors of the right shape, the
decoder is faithful (runs over real sampled shots), and pooled belief LER
<= PyMatching LER + small slack. ~600 shots; distance fixed to 5 because
QECTOR ``BeliefMatching`` rebuilds the weighted matching per shot.
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3.belief_matching import BeliefMatching


def test_belief_memory_z_valid_and_le_pymatching():
    d, p, N, seed = 5, 0.005, 600, 3

    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    bm = BeliefMatching.from_detector_error_model(sdem, max_iter=20)
    pm = pymatching.Matching.from_detector_error_model(sdem)
    no = bm.num_observables

    det, obs = circ.compile_detector_sampler(seed=seed).sample(shots=N, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    bpred = np.asarray(bm.decode_batch(det), np.uint8).reshape(N, -1)
    ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(N, -1)

    # Valid output: correct shape and strictly 0/1 per observable.
    assert bpred.shape == (N, no)
    assert ppred.shape[0] == N
    assert set(np.unique(bpred).tolist()).issubset({0, 1})

    # Faithful end-to-end: per-shot single decode matches the batch decode.
    for i in (0, N // 2, N - 1):
        single = np.asarray(bm.decode(det[i]), np.uint8).reshape(-1)
        assert single.shape == (no,)
        assert np.array_equal(single, bpred[i])

    belief_err = int(np.any(bpred != obs, axis=1).sum())
    pm_err = int(np.any(ppred != obs, axis=1).sum())
    print(
        f"[memory_z] shots={N} belief_err={belief_err} pm_err={pm_err} "
        f"belief_LER={belief_err / N:.4f} pm_LER={pm_err / N:.4f} "
        f"reduction={((pm_err - belief_err) / N):.4f}"
    )

    # Belief LER no worse than PyMatching beyond a small slack.
    slack = max(3, int(0.15 * pm_err))
    assert belief_err <= pm_err + slack
