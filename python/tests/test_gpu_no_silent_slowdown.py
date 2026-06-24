"""Section 8 (GPU proof): the GPU is only routed when intended, never silently wrong.

Two failure modes are ruled out here:

  1. *Silent slowdown / mis-routing* — after ``calibrate`` the router must use
     the GPU exactly when ``n >= gpu_threshold`` (and a GPU is available), and
     CPU otherwise.  If the GPU never beats CPU, calibration pins the threshold
     enormous and routing stays CPU for all realistic sizes — that is the
     *correct* "no silent slowdown" outcome and we assert it explicitly.
  2. *Silent divergence* — every batch actually routed to the GPU, plus a direct
     4096-shot ``CUDABatchDecoder`` call, must be bit-identical to the CPU
     reference.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes
from qector_decoder_v3.backend import AutoDecoder, Backend


def _syn(code, H, n, p=0.05, seed=0):
    rng = np.random.default_rng(seed)
    err = (rng.random((n, code.n_qubits)) < p).astype(np.uint8)
    return ((err @ H.T) & 1).astype(np.uint8)


def test_routing_matches_threshold_and_gpu_never_diverges():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    c2q, nq = code.check_to_qubits, code.n_qubits

    dec = AutoDecoder(c2q, nq)
    dec.calibrate(sizes=(64, 256, 1024, 4096), repeats=2)
    threshold = dec.config.gpu_threshold
    gpu_present = qd.cuda_is_available() or qd.opencl_is_available()

    cpu = qd.CPUBatchDecoder(c2q, nq)

    # Sizes spanning the threshold (plus a moderate "above" size we can run).
    sizes = sorted({1, 64, 256, 1024, 4096, max(threshold, 1)} & set(range(1, 4097)))
    if threshold <= 4096:
        sizes = sorted(set(sizes) | {threshold, min(threshold * 2, 4096)})
    sizes = [n for n in sorted(set(sizes)) if 1 <= n <= 4096]

    gpu_was_used = False
    for n in sizes:
        sel = dec.select(n)
        # Routing rule: GPU iff (GPU available and n >= threshold).
        if gpu_present and n >= threshold:
            assert sel in (Backend.CUDA, Backend.OPENCL)
        else:
            assert sel in (Backend.CPU_SINGLE, Backend.CPU_RAYON)

        syn = _syn(code, H, n, seed=100 + n)
        out = np.asarray(dec.batch_decode(syn), np.uint8)
        cpu_out = np.asarray(cpu.batch_decode(syn), np.uint8)
        assert np.array_equal((out @ H.T) & 1, syn)

        if dec.last_backend in (Backend.CUDA, Backend.OPENCL):
            gpu_was_used = True
            # Anything actually routed to the GPU must match CPU bit-for-bit.
            assert np.array_equal(out, cpu_out)

    if not gpu_present:
        pytest.skip("no GPU device")

    # If the GPU never beat CPU, calibration pins the threshold huge -> stays
    # CPU.  That is the correct "no silent slowdown" behaviour; assert it.
    if threshold > 4096:
        assert not gpu_was_used
        assert dec.select(4096) in (Backend.CPU_SINGLE, Backend.CPU_RAYON)


def test_direct_cuda_4096_matches_cpu():
    if not qd.cuda_is_available():
        pytest.skip("no CUDA")
    code = codes.rotated_surface_code(7)
    H = code.parity_check_matrix()
    c2q, nq = code.check_to_qubits, code.n_qubits
    syn = _syn(code, H, 4096, seed=2024)

    cpu = qd.CPUBatchDecoder(c2q, nq)
    cuda = qd.CUDABatchDecoder(c2q, nq)
    cpu_out = np.asarray(cpu.batch_decode(syn), np.uint8)
    cuda_out = np.asarray(cuda.batch_decode(syn), np.uint8)

    # No silent divergence at a realistic GPU batch size.
    assert np.array_equal(cuda_out, cpu_out)
    assert np.array_equal((cuda_out @ H.T) & 1, syn)
