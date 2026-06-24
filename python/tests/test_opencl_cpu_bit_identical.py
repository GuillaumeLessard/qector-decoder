"""Section 8 (GPU proof): OpenCL batch decoding is bit-identical to the CPU path.

Mirror of the CUDA proof but for :class:`OpenCLBatchDecoder`.  For a range of
distances and batch sizes the OpenCL correction must be byte-for-byte equal
(``np.array_equal``) to the CPU correction and both syndrome-faithful
(``(out @ H.T) & 1 == syn``).

Skipped automatically when no OpenCL device is present so the suite stays
portable.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _build(distance, batch, p=0.05, seed=4321):
    code = codes.rotated_surface_code(distance)
    H = code.parity_check_matrix()
    nq = code.n_qubits
    rng = np.random.default_rng(seed + 7 * distance + batch)
    err = (rng.random((batch, nq)) < p).astype(np.uint8)
    syn = ((err @ H.T) & 1).astype(np.uint8)
    return code, H, syn


@pytest.mark.parametrize("distance", [3, 5, 7, 9])
@pytest.mark.parametrize("batch", [1, 64, 1024, 4096])
def test_opencl_bit_identical_to_cpu(distance, batch):
    if not qd.opencl_is_available():
        pytest.skip("no OpenCL")
    code, H, syn = _build(distance, batch)
    cpu = qd.CPUBatchDecoder(code.check_to_qubits, code.n_qubits)
    ocl = qd.OpenCLBatchDecoder(code.check_to_qubits, code.n_qubits)

    cpu_out = np.asarray(cpu.batch_decode(syn), np.uint8)
    ocl_out = np.asarray(ocl.batch_decode(syn), np.uint8)

    assert cpu_out.shape == (batch, code.n_qubits)
    assert ocl_out.shape == (batch, code.n_qubits)

    # Bit-for-bit identical to the CPU reference.
    assert np.array_equal(ocl_out, cpu_out)

    # Both are syndrome-faithful.
    assert np.array_equal((cpu_out @ H.T) & 1, syn)
    assert np.array_equal((ocl_out @ H.T) & 1, syn)
