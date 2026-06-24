"""Section 8 (GPU proof): AutoDecoder.calibrate tunes the GPU threshold honestly.

We calibrate the CPU/GPU crossover on this machine and assert:
  * ``calibrate`` returns ``gpu_backend``, ``crossover`` and ``gpu_threshold``;
  * tiny batches stay on a CPU backend while huge batches respect the
    calibrated ``gpu_threshold`` (GPU iff ``n >= gpu_threshold`` and a GPU is
    available);
  * after calibration ``batch_decode`` of a moderate batch is syndrome-faithful
    and bit-identical to :class:`CPUBatchDecoder`;
  * ``diagnostics()`` reports ``calibrated == True``.
"""
import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes
from qector_decoder_v3.backend import AutoDecoder, Backend


def test_calibrate_keys_routing_and_faithfulness():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    c2q, nq = code.check_to_qubits, code.n_qubits

    dec = AutoDecoder(c2q, nq)
    cal = dec.calibrate(sizes=(64, 256, 1024, 4096), repeats=2)

    # Calibration report has the required keys.
    for key in ("gpu_backend", "crossover", "gpu_threshold"):
        assert key in cal
    assert cal["gpu_threshold"] == dec.config.gpu_threshold

    gpu_present = qd.cuda_is_available() or qd.opencl_is_available()
    threshold = dec.config.gpu_threshold

    # Small batches must stay on the CPU.
    assert dec.select(1) in (Backend.CPU_SINGLE, Backend.CPU_RAYON)
    assert dec.select(8) in (Backend.CPU_SINGLE, Backend.CPU_RAYON)

    # A huge batch respects the calibrated threshold.
    huge = max(threshold * 2, 1 << 16) if threshold < (1 << 40) else (1 << 20)
    sel_huge = dec.select(huge)
    if gpu_present and huge >= threshold:
        assert sel_huge in (Backend.CUDA, Backend.OPENCL)
    else:
        # GPU never won (threshold pinned huge) -> stays CPU. Correct outcome.
        assert sel_huge in (Backend.CPU_SINGLE, Backend.CPU_RAYON)

    # Routing rule holds exactly: GPU iff n >= threshold (and GPU available).
    for n in (1, 64, 256, 1024, 4096, huge):
        sel = dec.select(n)
        if gpu_present and n >= threshold:
            assert sel in (Backend.CUDA, Backend.OPENCL)
        else:
            assert sel in (Backend.CPU_SINGLE, Backend.CPU_RAYON)

    # After calibration the decoder still decodes correctly.
    cpu = qd.CPUBatchDecoder(c2q, nq)
    rng = np.random.default_rng(7)
    n = 1024
    err = (rng.random((n, nq)) < 0.05).astype(np.uint8)
    syn = ((err @ H.T) & 1).astype(np.uint8)
    out = np.asarray(dec.batch_decode(syn), np.uint8)
    cpu_out = np.asarray(cpu.batch_decode(syn), np.uint8)
    assert np.array_equal((out @ H.T) & 1, syn)
    assert np.array_equal(out, cpu_out)

    # Diagnostics confirm calibration ran.
    assert dec.diagnostics()["calibrated"] is True
