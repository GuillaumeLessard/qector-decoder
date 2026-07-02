"""Permanent regression: QECTOR-Blossom decodes as well as exact MWPM on the
collapsed circuit-level graph.

This locks in the fix for the large-dense-instance optimality bug. Before the
fix, QECTOR's fixed k=12 candidate set undershot the optimum on big graphs
(d>=15), giving heavier matchings and a ~3x worse logical error rate than
PyMatching. We now require QECTOR's logical error count to be no worse than
PyMatching's (within small statistical/tie slack) on the SAME seeded Stim shots,
QECTOR and PyMatching decoding the IDENTICAL collapsed graph.

The comparison is on logical outcomes (not raw matching weight), which is the
quantity that matters and is immune to QECTOR's internal weight quantization.
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3 import dem, pymatching_compat  # noqa: E402


def _logical_errors(d, shots, seed, basis="x", noise=0.005):
    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=d,
        rounds=d,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=seed).sample(shots=shots, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)
    qm = pymatching_compat.Matching.from_detector_error_model(sdem)
    pm = pymatching.Matching.from_detector_error_model(sdem)
    qpred = np.asarray(qm.decode_batch(det), np.uint8).reshape(shots, -1)
    ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(shots, -1)
    q_err = int(np.any(qpred != obs, axis=1).sum())
    p_err = int(np.any(ppred != obs, axis=1).sum())
    return q_err, p_err


@pytest.mark.parametrize("d", [5, 9, 13, 15])
def test_blossom_no_worse_than_exact_mwpm(d):
    """QECTOR-Blossom logical error rate <= PyMatching's on the collapsed graph.

    Pre-fix this failed at d=15 (QECTOR ~3x worse). The slack absorbs equal-weight
    ties / Poisson noise without admitting a real optimality regression.
    """
    shots = 6000 if d <= 13 else 4000
    q_err, p_err = _logical_errors(d, shots=shots, seed=20260622)
    slack = max(4, int(0.30 * p_err))
    assert q_err <= p_err + slack, (
        f"d={d}: QECTOR {q_err} logical errors vs PyMatching {p_err} "
        f"(+{slack} slack) — large-dense-instance optimality regression"
    )
