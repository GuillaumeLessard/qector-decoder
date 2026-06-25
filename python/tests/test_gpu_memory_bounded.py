"""Section 10 (memory): repeated GPU batch decoding stays RSS-bounded and faithful.

Call ``CUDABatchDecoder.batch_decode`` 10x on a 16384-shot batch at d=9. The
first call pays one-time CUDA context initialization, so we measure RSS growth
from the 2nd call to the last and require it to be bounded. Every output batch is
verified syndrome-faithful via the parity-check matrix.

Skips cleanly if psutil is missing or CUDA is unavailable. Observed growth on the
dev machine (GTX 1660 Ti) is ~0 MiB after init; the 200 MiB bound is generous yet
still flags a per-call GPU/host leak.
"""

import gc
import os

import numpy as np
import pytest

psutil = pytest.importorskip("psutil")

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes

MAX_GROWTH_MIB = 200.0


def _rss_mib(proc):
    return proc.memory_info().rss / (1024 * 1024)


def test_gpu_memory_bounded():
    if not qd.cuda_is_available():
        pytest.skip("no CUDA")

    code = codes.rotated_surface_code(9)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()

    rng = np.random.default_rng(9)
    batch_n = 16384
    syns = np.array(
        [
            np.asarray(code.syndrome(code.random_error(0.08, rng)), np.uint8)
            for _ in range(batch_n)
        ],
        np.uint8,
    )

    dec = qd.CUDABatchDecoder(c2q, nq)
    proc = psutil.Process(os.getpid())

    rss_after_call = []
    for call in range(10):
        out = np.asarray(dec.batch_decode(syns), np.uint8).reshape(batch_n, nq)
        assert np.array_equal((H @ out.T).T & 1, syns), (
            f"GPU batch_decode unfaithful on call {call}"
        )
        gc.collect()
        rss_after_call.append(_rss_mib(proc))

    # First call pays CUDA init; measure growth from the 2nd call onward.
    growth = rss_after_call[-1] - rss_after_call[1]
    assert growth < MAX_GROWTH_MIB, (
        f"GPU RSS grew {growth:.1f} MiB from call 2 to last "
        f"(limit {MAX_GROWTH_MIB} MiB); samples={['%.1f' % s for s in rss_after_call]}"
    )
