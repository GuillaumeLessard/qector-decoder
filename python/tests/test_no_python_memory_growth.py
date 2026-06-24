"""Section 10 (memory): the hot decode path does not grow Python heap memory.

Using ``tracemalloc``, decode a fixed pool of syndromes through a Blossom decoder
in a long hot loop. After a warmup we snapshot current traced memory, run many
more iterations, and assert the delta is small -- proving the hot path reuses
buffers rather than leaking Python objects per decode.

Observed delta on the dev machine is < 1 KiB; the 5 MiB bound is generous but
still catches a real unbounded per-decode allocation.
"""
import gc
import tracemalloc

import numpy as np

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes

MAX_GROWTH_BYTES = 5 * 1024 * 1024  # 5 MiB


def test_no_python_memory_growth():
    code = codes.rotated_surface_code(7)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()

    rng = np.random.default_rng(7)
    pool_size = 256
    pool = np.array(
        [np.asarray(code.syndrome(code.random_error(0.08, rng)), np.uint8)
         for _ in range(pool_size)],
        np.uint8,
    )

    dec = qd.BlossomDecoder(c2q, nq)

    tracemalloc.start()
    try:
        # Warmup: fill any one-time caches/buffers.
        for i in range(3000):
            dec.decode(pool[i % pool_size])
        gc.collect()
        cur_start, _ = tracemalloc.get_traced_memory()

        # Long hot loop.
        for i in range(40000):
            dec.decode(pool[i % pool_size])
        gc.collect()
        cur_end, peak_end = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    growth = cur_end - cur_start
    assert growth < MAX_GROWTH_BYTES, (
        f"Python heap grew {growth / 1024:.1f} KiB across the hot loop "
        f"(limit {MAX_GROWTH_BYTES / 1024:.0f} KiB) -- possible per-decode leak"
    )

    # Spot-check correctness so the loop did real work.
    c = np.asarray(dec.decode(pool[0]), np.uint8)
    assert np.array_equal((H @ c) & 1, pool[0])
