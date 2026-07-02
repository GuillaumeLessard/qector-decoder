"""Miniature master correctness audit (docs/CORRECTNESS_AUDIT.md).

docs/CORRECTNESS_AUDIT.md states the core invariant — every decoder must satisfy
``H . decode(s) == s (mod 2)`` for every reachable syndrome — and that the GPU
batch paths are bit-identical to the CPU reference.

This test compresses that audit: for rotated surface codes at d in (3, 5, 7) and
a fixed seed, it builds ~1000 reachable syndromes and runs BlossomDecoder,
SparseBlossomDecoder, UnionFindDecoder and CPUBatchDecoder over the *same*
syndromes, asserting:

* every correction from every decoder is syndrome-faithful,
* ``CPUBatchDecoder.batch_decode`` is bit-identical to per-shot ``decode``, and
* if CUDA is available, ``CUDABatchDecoder.batch_decode`` is bit-identical to
  ``CPUBatchDecoder.batch_decode`` on the same batch.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes

_DISTANCES = (3, 5, 7)
_N_SHOTS = 1000
_SEED = 4242


def _make_batch(code, seed):
    rng = np.random.default_rng(seed)
    return np.array(
        [code.syndrome(code.random_error(0.08, rng)) for _ in range(_N_SHOTS)],
        dtype=np.uint8,
    )


@pytest.mark.parametrize("d", _DISTANCES)
def test_all_decoders_syndrome_faithful(d):
    code = codes.rotated_surface_code(d)
    H = code.parity_check_matrix()
    batch = _make_batch(code, _SEED + d)

    decoders = {
        "BlossomDecoder": qd.BlossomDecoder(code.check_to_qubits, code.n_qubits),
        "SparseBlossomDecoder": qd.SparseBlossomDecoder(code.check_to_qubits, code.n_qubits),
        "UnionFindDecoder": qd.UnionFindDecoder(code.check_to_qubits, code.n_qubits),
        "CPUBatchDecoder": qd.CPUBatchDecoder(code.check_to_qubits, code.n_qubits),
    }

    for name, dec in decoders.items():
        for i in range(_N_SHOTS):
            corr = np.asarray(dec.decode(batch[i]), np.uint8)
            assert corr.shape == (code.n_qubits,)
            assert np.array_equal((H @ corr) & 1, batch[i]), f"{name} not syndrome-faithful on shot {i} (d={d})"


@pytest.mark.parametrize("d", _DISTANCES)
def test_cpu_batch_matches_per_shot_decode(d):
    code = codes.rotated_surface_code(d)
    batch = _make_batch(code, _SEED + d)
    cpu = qd.CPUBatchDecoder(code.check_to_qubits, code.n_qubits)

    batched = np.asarray(cpu.batch_decode(batch), np.uint8).reshape(_N_SHOTS, -1)
    per_shot = np.stack([np.asarray(cpu.decode(batch[i]), np.uint8) for i in range(_N_SHOTS)])
    assert batched.shape == per_shot.shape
    assert np.array_equal(batched, per_shot), f"CPUBatchDecoder.batch_decode != per-shot decode (d={d})"


@pytest.mark.parametrize("d", _DISTANCES)
def test_cuda_batch_bit_identical_to_cpu(d):
    if not qd.cuda_is_available():
        pytest.skip("no CUDA")
    code = codes.rotated_surface_code(d)
    batch = _make_batch(code, _SEED + d)

    cpu = qd.CPUBatchDecoder(code.check_to_qubits, code.n_qubits)
    cuda = qd.CUDABatchDecoder(code.check_to_qubits, code.n_qubits)

    cpu_out = np.asarray(cpu.batch_decode(batch), np.uint8).reshape(_N_SHOTS, -1)
    cuda_out = np.asarray(cuda.batch_decode(batch), np.uint8).reshape(_N_SHOTS, -1)

    assert cuda_out.shape == cpu_out.shape
    assert np.array_equal(cuda_out, cpu_out), f"CUDABatchDecoder not bit-identical to CPUBatchDecoder (d={d})"
