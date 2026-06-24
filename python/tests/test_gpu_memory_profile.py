"""Section 8 (GPU proof): repeated CUDA batch decoding is stable and leak-free.

We run several repeated ``CUDABatchDecoder.batch_decode`` calls on a 16384-shot
batch at d=9 and assert:
  * every output is syndrome-faithful and bit-identical to the first call
    (deterministic, no silent drift across repeats); and
  * the process RSS (via psutil) does not grow unboundedly across the repeats.

The first call pays a one-time CUDA context / buffer initialisation cost, so the
growth check compares from the *second* call onward.

Requires psutil and a CUDA device; skipped otherwise.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes

psutil = pytest.importorskip("psutil")


def test_cuda_repeated_batch_no_leak_and_identical():
    if not qd.cuda_is_available():
        pytest.skip("no CUDA")

    code = codes.rotated_surface_code(9)
    H = code.parity_check_matrix()
    c2q, nq = code.check_to_qubits, code.n_qubits

    N = 16384
    rng = np.random.default_rng(2026)
    err = (rng.random((N, nq)) < 0.05).astype(np.uint8)
    syn = ((err @ H.T) & 1).astype(np.uint8)

    cuda = qd.CUDABatchDecoder(c2q, nq)
    proc = psutil.Process()

    REPEATS = 8
    GROWTH_LIMIT = 200 * 1024 * 1024  # 200 MiB

    first_out = None
    rss_samples = []
    for i in range(REPEATS):
        out = np.asarray(cuda.batch_decode(syn), np.uint8)
        # Syndrome-faithful on every call.
        assert np.array_equal((out @ H.T) & 1, syn)
        if first_out is None:
            first_out = out
        else:
            # Deterministic: identical to the first call, no silent drift.
            assert np.array_equal(out, first_out)
        rss_samples.append(proc.memory_info().rss)

    # Compare from the 2nd call onward (the 1st pays one-time CUDA init).
    baseline = rss_samples[1]
    growth = rss_samples[-1] - baseline
    assert growth < GROWTH_LIMIT, (
        f"RSS grew {growth / (1024 * 1024):.1f} MiB across repeats "
        f"(samples MiB: {[int(r / (1024 * 1024)) for r in rss_samples]})"
    )
