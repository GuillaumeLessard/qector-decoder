"""Section 10 (memory): a long-run decode soak stays RSS-bounded and faithful.

Run 60000 decodes at d=9 SparseBlossom over a fixed syndrome pool, sampling RSS
throughout. We assert process RSS stays bounded over the whole run and that every
periodic spot-check decode is syndrome-faithful -- i.e. correctness does not
degrade and memory does not climb under sustained load.

Observed RSS spread on the dev machine is < 1 MiB; the 48 MiB bound is generous
yet still catches a real unbounded leak.
"""

import gc
import os

import numpy as np
import pytest

psutil = pytest.importorskip("psutil")

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes

MAX_GROWTH_MIB = 48.0


def _rss_mib(proc):
    return proc.memory_info().rss / (1024 * 1024)


def test_long_run_decode_memory():
    code = codes.rotated_surface_code(9)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()

    rng = np.random.default_rng(9)
    pool_size = 512
    pool = np.array(
        [
            np.asarray(code.syndrome(code.random_error(0.08, rng)), np.uint8)
            for _ in range(pool_size)
        ],
        np.uint8,
    )

    dec = qd.SparseBlossomDecoder(c2q, nq)
    proc = psutil.Process(os.getpid())

    n = 60000
    sample_every = 5000
    samples = []
    faithful_checks = 0
    for i in range(n):
        corr = dec.decode(pool[i % pool_size])
        if i % sample_every == 0:
            corr = np.asarray(corr, np.uint8)
            # Periodic spot-check: every sampled decode must be faithful.
            assert np.array_equal((H @ corr) & 1, pool[i % pool_size]), (
                f"unfaithful decode at iteration {i}"
            )
            faithful_checks += 1
            gc.collect()
            samples.append(_rss_mib(proc))
    gc.collect()
    samples.append(_rss_mib(proc))

    assert faithful_checks >= 5
    spread = max(samples) - min(samples)
    assert spread < MAX_GROWTH_MIB, (
        f"RSS spread {spread:.1f} MiB over the soak (limit {MAX_GROWTH_MIB} MiB); "
        f"samples={['%.1f' % s for s in samples]}"
    )
