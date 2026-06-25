"""Section 10 (memory): the native (Rust) decode path does not leak process RSS.

Decode in a tight loop (30000 single decodes at d=7 Blossom over a 256-syndrome
pool), sampling resident set size (RSS) at intervals via psutil. We compare RSS
in the first quarter of the run against the last quarter and require the growth
to be bounded -- a real unbounded native leak would show monotonic RSS climb.

Observed growth on the dev machine is ~0 MiB; the 32 MiB bound is robust to GC
and allocator jitter while still catching a genuine leak.
"""

import gc
import os

import numpy as np
import pytest

psutil = pytest.importorskip("psutil")

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes

MAX_GROWTH_MIB = 32.0


def _rss_mib(proc):
    return proc.memory_info().rss / (1024 * 1024)


def test_no_native_rss_leak():
    code = codes.rotated_surface_code(7)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()

    rng = np.random.default_rng(7)
    pool_size = 256
    pool = np.array(
        [
            np.asarray(code.syndrome(code.random_error(0.08, rng)), np.uint8)
            for _ in range(pool_size)
        ],
        np.uint8,
    )

    dec = qd.BlossomDecoder(c2q, nq)
    proc = psutil.Process(os.getpid())

    n = 30000
    sample_every = 2000
    samples = []
    for i in range(n):
        dec.decode(pool[i % pool_size])
        if i % sample_every == 0:
            gc.collect()
            samples.append(_rss_mib(proc))
    gc.collect()
    samples.append(_rss_mib(proc))

    assert len(samples) >= 4
    first_quarter = samples[len(samples) // 4]
    last_quarter = samples[-1]
    growth = last_quarter - first_quarter

    assert growth < MAX_GROWTH_MIB, (
        f"RSS grew {growth:.1f} MiB (first-quarter {first_quarter:.1f} -> "
        f"last {last_quarter:.1f}); limit {MAX_GROWTH_MIB} MiB -- possible native leak. "
        f"samples={['%.1f' % s for s in samples]}"
    )

    # The loop did real, correct work.
    c = np.asarray(dec.decode(pool[0]), np.uint8)
    assert np.array_equal((H @ c) & 1, pool[0])
