"""Section 8 (GPU proof): graceful fallback always yields a CPU-identical answer.

We exercise :class:`AutoDecoder` / :class:`BackendConfig` routing:

(a) ``allow_gpu=False`` routes *every* batch size to a CPU backend
    (``CPU_SINGLE`` or ``CPU_RAYON``) and the result is syndrome-faithful and
    bit-identical to :class:`CPUBatchDecoder`.
(b) With GPU allowed and ``force=Backend.CUDA`` (when CUDA is present) the GPU
    path produces the *same* answer as the CPU reference.
(c) ``allow_gpu=False`` reports no GPU in ``available_backends()``.

The invariant under test: the fallback must never diverge from CPU.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes
from qector_decoder_v3.backend import AutoDecoder, Backend, BackendConfig


def _build(distance=5, batch=512, p=0.05, seed=11):
    code = codes.rotated_surface_code(distance)
    H = code.parity_check_matrix()
    rng = np.random.default_rng(seed)
    err = (rng.random((batch, code.n_qubits)) < p).astype(np.uint8)
    syn = ((err @ H.T) & 1).astype(np.uint8)
    return code, H, syn


def test_allow_gpu_false_routes_to_cpu_and_matches():
    code, H, _ = _build()
    cfg = BackendConfig(allow_gpu=False)
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits, cfg)
    cpu = qd.CPUBatchDecoder(code.check_to_qubits, code.n_qubits)

    # Every batch size — including a very large one — stays on a CPU backend.
    for n in (1, 7, 8, 64, 1024, 100000):
        assert ad.select(n) in (Backend.CPU_SINGLE, Backend.CPU_RAYON)

    rng = np.random.default_rng(99)
    for n in (1, 64, 1024):
        err = (rng.random((n, code.n_qubits)) < 0.05).astype(np.uint8)
        syn = ((err @ H.T) & 1).astype(np.uint8)
        out = np.asarray(ad.batch_decode(syn), np.uint8)
        cpu_out = np.asarray(cpu.batch_decode(syn), np.uint8)
        # Faithful and bit-identical to the dedicated CPU decoder.
        assert np.array_equal((out @ H.T) & 1, syn)
        assert np.array_equal(out, cpu_out)
        # And we really did route to a CPU backend.
        assert ad.last_backend in (Backend.CPU_SINGLE, Backend.CPU_RAYON)


def test_force_cuda_matches_cpu_exactly():
    if not qd.cuda_is_available():
        pytest.skip("no CUDA")
    code, H, syn = _build(batch=1024)
    cfg = BackendConfig(allow_gpu=True, force=Backend.CUDA)
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits, cfg)
    cpu = qd.CPUBatchDecoder(code.check_to_qubits, code.n_qubits)

    # The forced GPU path is selected for any size.
    assert ad.select(1) == Backend.CUDA
    assert ad.select(1024) == Backend.CUDA

    out = np.asarray(ad.batch_decode(syn), np.uint8)
    cpu_out = np.asarray(cpu.batch_decode(syn), np.uint8)
    # GPU path actually ran and produced the SAME answer as CPU.
    assert ad.last_backend == Backend.CUDA
    assert np.array_equal(out, cpu_out)
    assert np.array_equal((out @ H.T) & 1, syn)


def test_allow_gpu_false_reports_no_gpu():
    code, _, _ = _build()
    ad = AutoDecoder(code.check_to_qubits, code.n_qubits, BackendConfig(allow_gpu=False))
    avail = ad.available_backends()
    assert Backend.CUDA not in avail
    assert Backend.OPENCL not in avail
    # CPU backends are always present.
    assert Backend.CPU_SINGLE in avail
    assert Backend.CPU_RAYON in avail
