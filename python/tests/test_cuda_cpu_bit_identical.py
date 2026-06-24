"""Section 8 (GPU proof): CUDA batch decoding is bit-identical to the CPU path.

For a range of distances and batch sizes we build reachable syndromes from
random errors, decode with both :class:`CUDABatchDecoder` and
:class:`CPUBatchDecoder`, and assert the corrections are *byte-for-byte* equal
(``np.array_equal``) and both syndrome-faithful (``(out @ H.T) & 1 == syn``).

Skipped automatically when no CUDA device is present so the suite stays portable.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _build(distance, batch, p=0.05, seed=1234):
    code = codes.rotated_surface_code(distance)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    rng = np.random.default_rng(seed + 7 * distance + batch)
    err = (rng.random((batch, nq)) < p).astype(np.uint8)
    syn = ((err @ H.T) & 1).astype(np.uint8)
    return code, H, syn


@pytest.mark.parametrize("distance", [3, 5, 7, 9])
@pytest.mark.parametrize("batch", [1, 64, 1024, 4096])
def test_cuda_bit_identical_to_cpu(distance, batch):
    if not qd.cuda_is_available():
        pytest.skip("no CUDA")
    code, H, syn = _build(distance, batch)
    cpu = qd.CPUBatchDecoder(code.check_to_qubits, code.n_qubits)
    cuda = qd.CUDABatchDecoder(code.check_to_qubits, code.n_qubits)

    cpu_out = np.asarray(cpu.batch_decode(syn), np.uint8)
    cuda_out = np.asarray(cuda.batch_decode(syn), np.uint8)

    assert cpu_out.shape == (batch, code.n_qubits)
    assert cuda_out.shape == (batch, code.n_qubits)

    # Bit-for-bit identical to the CPU reference.
    assert np.array_equal(cuda_out, cpu_out)

    # Both are syndrome-faithful.
    assert np.array_equal((cpu_out @ H.T) & 1, syn)
    assert np.array_equal((cuda_out @ H.T) & 1, syn)
