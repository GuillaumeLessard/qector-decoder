"""Belief-matching proof (section 4): agreement with reference ``beliefmatching``.

On identical sampled shots at d in {3, 5}, QECTOR's belief LER tracks the
reference ``beliefmatching`` package's LER closely:
``abs(q_err - ref_err) <= max(5, int(0.25*ref_err))``.

NOTE: the reference uses keyword ``max_bp_iters`` (not ``max_iter``). Both
implementations are slow, so shots are kept small (<=600 total at each d).
"""
import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")
beliefmatching = pytest.importorskip("beliefmatching")

from beliefmatching import BeliefMatching as RefBM

from qector_decoder_v3.belief_matching import BeliefMatching as QBM


def _circuit(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


@pytest.mark.parametrize("d,N", [(3, 600), (5, 600)])
def test_belief_tracks_reference_package(d, N):
    p, seed = 0.006, 11

    circ = _circuit(d, p)
    sdem = circ.detector_error_model(decompose_errors=True)

    qbm = QBM.from_detector_error_model(sdem, max_iter=20)
    ref = RefBM.from_detector_error_model(sdem, max_bp_iters=20)
    no = qbm.num_observables

    det, obs = circ.compile_detector_sampler(seed=seed).sample(
        shots=N, separate_observables=True
    )
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    qpred = np.asarray(qbm.decode_batch(det), np.uint8).reshape(N, -1)
    rpred = np.asarray(ref.decode_batch(det), np.uint8).reshape(N, -1)

    assert qpred.shape == (N, no)
    assert set(np.unique(qpred).tolist()).issubset({0, 1})

    q_err = int(np.any(qpred != obs, axis=1).sum())
    ref_err = int(np.any(rpred != obs, axis=1).sum())
    print(
        f"[ref_pkg] d={d} shots={N} q_err={q_err} ref_err={ref_err} "
        f"q_LER={q_err / N:.4f} ref_LER={ref_err / N:.4f} diff={abs(q_err - ref_err)}"
    )

    # QECTOR belief and reference beliefmatching must track each other.
    tol = max(5, int(0.25 * ref_err))
    assert abs(q_err - ref_err) <= tol
