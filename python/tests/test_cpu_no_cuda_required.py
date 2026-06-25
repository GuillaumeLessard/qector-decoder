"""The package and its CPU decoders work without any GPU.

QECTOR builds GPU kernels but must never *require* a GPU to import or to decode.
This is proven two ways:

1. A fresh subprocess imports the package, builds a CPU BlossomDecoder on a
   rotated surface code, decodes a reachable syndrome faithfully, and prints
   ``OK`` — exercising the cold import path with no GPU dependency forced.
2. In-process, an ``AutoDecoder`` configured with ``allow_gpu=False`` is asserted
   to select a CPU backend for every batch size (never CUDA/OpenCL) and to
   produce syndrome-faithful batch corrections.
"""

import subprocess
import sys
import textwrap

import numpy as np

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes
from qector_decoder_v3.backend import AutoDecoder, Backend, BackendConfig


def test_cpu_decode_in_fresh_subprocess():
    code = textwrap.dedent(
        """
        import numpy as np
        import qector_decoder_v3 as qd
        from qector_decoder_v3 import codes

        c = codes.rotated_surface_code(5)
        H = c.parity_check_matrix()
        rng = np.random.default_rng(0)
        e = c.random_error(0.1, rng)
        s = c.syndrome(e)

        dec = qd.BlossomDecoder(c.check_to_qubits, c.n_qubits)
        corr = np.asarray(dec.decode(s), np.uint8)
        assert np.array_equal((H @ corr) & 1, s), "not syndrome-faithful"
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "OK" in proc.stdout, f"missing OK marker; stdout={proc.stdout!r}"


def test_autodecoder_never_picks_gpu_when_disabled():
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    auto = AutoDecoder(
        code.check_to_qubits, code.n_qubits, BackendConfig(allow_gpu=False)
    )

    # No GPU backend should appear in the available set.
    avail = auto.available_backends()
    assert Backend.CUDA not in avail, avail
    assert Backend.OPENCL not in avail, avail

    # select() must return a CPU backend at every size, including huge batches.
    for n in (1, 8, 1000, 4096, 100000, 1 << 20):
        chosen = auto.select(n)
        assert chosen in (Backend.CPU_SINGLE, Backend.CPU_RAYON), (
            f"batch={n} routed to {chosen} despite allow_gpu=False"
        )
    assert auto.select(100000) in (Backend.CPU_SINGLE, Backend.CPU_RAYON)

    # And the CPU batch path must actually decode faithfully.
    rng = np.random.default_rng(3)
    batch = np.array(
        [code.syndrome(code.random_error(0.08, rng)) for _ in range(128)],
        dtype=np.uint8,
    )
    out = np.asarray(auto.batch_decode(batch), np.uint8).reshape(batch.shape[0], -1)
    assert out.shape[1] == code.n_qubits
    for i in range(batch.shape[0]):
        assert np.array_equal((H @ out[i]) & 1, batch[i]), f"shot {i} not faithful"
    # Confirm the backend that ran was CPU.
    assert auto.last_backend in (Backend.CPU_SINGLE, Backend.CPU_RAYON), (
        auto.last_backend
    )


def test_cuda_availability_flag_is_callable_without_gpu_requirement():
    # The flag must be queryable regardless of hardware; importing the package
    # never requires CUDA to be present.
    assert isinstance(qd.cuda_is_available(), bool)
    assert isinstance(qd.opencl_is_available(), bool)
