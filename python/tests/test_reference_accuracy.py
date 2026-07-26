"""Reference accuracy tests comparing QECTOR decoders to PyMatching ground truth."""
from __future__ import annotations

import numpy as np
import pytest
import stim
import pymatching

from qector_decoder_v3 import dem as dem_module


def test_reference_accuracy_surface_code():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.005,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    model = dem_module.from_stim(dem)

    shots = 500
    sampler = circuit.compile_detector_sampler(seed=42)
    samples = sampler.sample(shots, append_observables=True)
    det_data = samples[:, :-1].astype(np.uint8)
    actual_obs = samples[:, -1:].astype(np.uint8)

    pm = pymatching.Matching.from_detector_error_model(dem)
    pm_pred = pm.decode_batch(det_data).astype(np.uint8)
    pm_errors = int(np.sum(np.any(pm_pred != actual_obs, axis=1)))

    q_blossom = model.make_decoder("blossom")
    q_corr = q_blossom.batch_decode(det_data)
    if q_corr.ndim == 1:
        q_corr = q_corr.reshape(shots, -1)

    L = model.observables_matrix().astype(np.uint8)
    q_pred = ((q_corr @ L.T) & 1).astype(np.uint8)
    q_errors = int(np.sum(np.any(q_pred != actual_obs, axis=1)))

    # 100% syndrome faithfulness
    H = model.check_matrix().astype(np.uint8)
    faithful = int(np.sum(np.all(((q_corr @ H.T) & 1) == det_data, axis=1)))
    assert faithful == shots, f"Expected 100% syndrome faithfulness, got {faithful}/{shots}"

    # LER parity within reasonable bound
    assert abs(q_errors - pm_errors) <= 2, f"Blossom errors ({q_errors}) mismatched PyMatching ({pm_errors})"
