"""Belief-matching proof (section 4): honest latency characterization.

Belief-matching is the ACCURACY mode, not the fast mode: it rebuilds a weighted
MWPM per shot using BP posteriors, so it costs substantially more per shot than
plain ``pymatching_compat.Matching`` MWPM. This test times both decoders on the
SAME small shot set (d=5, <=300 shots), checks both produce valid output, and
records the latency multiplier. It does NOT hard-fail on wall-clock: it only
asserts belief ran, produced valid output, and took measurable, non-negative
time -- per cheatsheet rule 8 (no absolute wall-clock thresholds).
"""
import time

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qector_decoder_v3 import pymatching_compat
from qector_decoder_v3.belief_matching import BeliefMatching


def test_belief_latency_cost_characterization():
    d, p, N, seed = 5, 0.005, 300, 1

    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    sdem = circ.detector_error_model(decompose_errors=True)

    bm = BeliefMatching.from_detector_error_model(sdem, max_iter=20)
    mwpm = pymatching_compat.Matching.from_detector_error_model(sdem)
    no = bm.num_observables

    det, obs = circ.compile_detector_sampler(seed=seed).sample(
        shots=N, separate_observables=True
    )
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    # Time plain MWPM.
    t0 = time.perf_counter()
    mpred = np.asarray(mwpm.decode_batch(det), np.uint8).reshape(N, -1)
    mwpm_seconds = time.perf_counter() - t0

    # Time belief-matching (the accuracy mode).
    t0 = time.perf_counter()
    bpred = np.asarray(bm.decode_batch(det), np.uint8).reshape(N, -1)
    belief_seconds = time.perf_counter() - t0

    # Both must produce valid output.
    assert bpred.shape == (N, no)
    assert mpred.shape[0] == N
    assert set(np.unique(bpred).tolist()).issubset({0, 1})
    assert set(np.unique(mpred).tolist()).issubset({0, 1})

    multiplier = belief_seconds / mwpm_seconds if mwpm_seconds > 0 else float("inf")
    print(
        f"[latency] shots={N} d={d} belief={belief_seconds:.3f}s "
        f"mwpm={mwpm_seconds:.4f}s multiplier={multiplier:.1f}x "
        f"(belief is the accuracy mode: higher per-shot cost)"
    )

    # Robust, non-flaky assertions only: belief ran and took measurable,
    # non-negative time. No absolute wall-clock thresholds (cheatsheet rule 8).
    assert belief_seconds >= 0.0
    assert belief_seconds > 0.0  # decoding N>0 shots takes measurable time
