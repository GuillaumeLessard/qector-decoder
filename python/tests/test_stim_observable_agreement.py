"""QECTOR's predicted observables match Stim's observable convention.

Section 6 (logical-observable proof). For a surface-code memory circuit we sample
detector events together with the true logical observables (``separate_observables=True``)
and decode with ``pymatching_compat.Matching.from_detector_error_model``. A logical error
is a shot whose *predicted observables* disagree with Stim's *sampled observables* -- this
is the only correct logical-failure metric for a circuit-level experiment.

We assert:

* The QECTOR LER (disagreement fraction) is a valid probability in ``[0, 1]`` and small.
* It matches the reference ``pymatching.Matching`` LER (decoded the same way) to within
  Wilson-95% confidence-interval slack.

If the predicted-observable convention disagreed with Stim's, the LER would be ~0.5
(random) rather than the small, matching value we require.
"""

import math

import numpy as np
import pytest


def _wilson(k, n, z=1.959963985):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _decode_failures(d, N, seed, basis="x", p=0.006):
    import stim
    import pymatching
    from qector_decoder_v3 import pymatching_compat

    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=seed).sample(shots=N, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    qm = pymatching_compat.Matching.from_detector_error_model(sdem)
    pm = pymatching.Matching.from_detector_error_model(sdem)

    qpred = np.asarray(qm.decode_batch(det), np.uint8).reshape(N, -1)
    ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(N, -1)
    assert qpred.shape == obs.shape == ppred.shape

    q_fail = int(np.any(qpred != obs, axis=1).sum())
    p_fail = int(np.any(ppred != obs, axis=1).sum())
    return q_fail, p_fail


@pytest.mark.parametrize("d", [3, 5])
def test_qector_observable_ler_matches_stim_and_reference(d):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    N = 1500
    q_fail, p_fail = _decode_failures(d, N, seed=1234)

    q_ler = q_fail / N
    p_ler = p_fail / N

    # Valid probabilities, and far from the 0.5 we would see if the observable
    # convention were wrong.
    assert 0.0 <= q_ler <= 1.0
    assert 0.0 <= p_ler <= 1.0
    assert q_ler < 0.2

    # QECTOR's observable LER agrees with the reference within combined Wilson slack.
    q_lo, q_hi = _wilson(q_fail, N)
    p_lo, p_hi = _wilson(p_fail, N)
    # Intervals must overlap (the two decoders agree on the logical metric).
    assert q_lo <= p_hi and p_lo <= q_hi, (q_fail, p_fail, (q_lo, q_hi), (p_lo, p_hi))


def test_observable_metric_is_not_random():
    """Higher distance lowers (or holds) the observable LER -- proves real decoding."""
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    N = 1500
    q3, _ = _decode_failures(3, N, seed=99)
    q5, _ = _decode_failures(5, N, seed=99)
    # Both are genuine logical error counts, nowhere near random (~0.5 * N).
    assert q3 < 0.3 * N
    assert q5 < 0.3 * N
    # At this low physical error rate the larger code does no worse (Wilson slack).
    lo5, _ = _wilson(q5, N)
    _, hi3 = _wilson(q3, N)
    assert lo5 <= hi3
