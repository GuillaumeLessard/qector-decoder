"""Phenomenological repetition code with measurement errors over multiple rounds.

Measurement-flip errors create *time-like* edges in the matching graph (a defect
appears in one round and disappears the next).  This test builds a multi-round
phenomenological repetition-code circuit with both data depolarization and
measurement flips, decodes the detector samples with QECTOR's PyMatching-compat
``Matching`` built from the (graphlike) detector error model, and checks:

* predictions have the right shape (one bit per observable), and
* QECTOR's logical error rate is within a Wilson-CI slack of upstream PyMatching
  on the SAME shots — i.e. the time edges are wired up correctly.
"""

import math

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3 import pymatching_compat  # noqa: E402


def _wilson(k, n, z=1.959963985):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


@pytest.mark.parametrize("d", [5, 7])
def test_measurement_error_time_edges_match_pymatching(d):
    r = d
    p = 0.02
    N = 3000
    circ = stim.Circuit.generated(
        "repetition_code:memory",
        distance=d,
        rounds=r,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
    )
    sdem = circ.detector_error_model(decompose_errors=True)
    assert sdem.num_observables >= 1

    det, obs = circ.compile_detector_sampler(seed=2025).sample(shots=N, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)
    assert det.shape == (N, sdem.num_detectors)

    qm = pymatching_compat.Matching.from_detector_error_model(sdem)
    pm = pymatching.Matching.from_detector_error_model(sdem)

    qpred = np.asarray(qm.decode_batch(det), np.uint8).reshape(N, -1)
    ppred = np.asarray(pm.decode_batch(det), np.uint8).reshape(N, -1)

    # predictions are one bit per observable
    assert qpred.shape == (N, sdem.num_observables)
    assert ppred.shape == (N, sdem.num_observables)

    q_fail = int(np.any(qpred != obs, axis=1).sum())
    p_fail = int(np.any(ppred != obs, axis=1).sum())

    # QECTOR's LER must sit inside the Wilson 95% CI of PyMatching's LER
    # (with a tiny absolute slack to absorb decoder tie-breaking differences).
    lo, hi = _wilson(p_fail, N)
    q_ler = q_fail / N
    slack = 0.01
    assert lo - slack <= q_ler <= hi + slack, (
        f"d={d}: QECTOR LER {q_ler:.4f} outside PyMatching Wilson CI "
        f"[{lo:.4f},{hi:.4f}] (q_fail={q_fail}, p_fail={p_fail})"
    )
