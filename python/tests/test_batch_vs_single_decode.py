"""Section 9 (latency): batched decoding is BIT-IDENTICAL to per-shot decoding.

The batched fast path must produce exactly the same corrections as decoding each
shot one at a time -- batching is an optimization, never a different answer.

We test the two decoders that are deterministic per-shot and order-independent:
  * CPUBatchDecoder.batch_decode  vs  per-shot .decode
  * BlossomDecoder.batch_decode   vs  per-shot .decode  (exact MWPM, deterministic)

All corrections are also verified syndrome-faithful via the parity-check matrix.

NOTE: ``rotated_surface_code`` takes a single ``distance`` here, so the spec's
"(5,7)" is realized as two distances, d=5 and d=7. Region-growing decoders
(SparseBlossom / UnionFind) may legitimately differ batch-vs-single if internal
state is order-dependent, so they are intentionally not asserted here.
"""
import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _make_syndromes(code, n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        err = code.random_error(0.08, rng)
        rows.append(np.asarray(code.syndrome(err), np.uint8))
    return np.array(rows, np.uint8)


def _per_shot(decoder, syn, nq):
    return np.array(
        [np.asarray(decoder.decode(syn[i]), np.uint8) for i in range(len(syn))],
        np.uint8,
    )


def _codes():
    return [
        ("rotated_surface_d5", codes.rotated_surface_code(5)),
        ("rotated_surface_d7", codes.rotated_surface_code(7)),
        ("repetition_d11", codes.repetition_code(11)),
    ]


@pytest.mark.parametrize("code_name,code", _codes())
def test_cpu_batch_bit_identical_to_single(code_name, code):
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    syn = _make_syndromes(code, 128, seed=12345)

    dec = qd.CPUBatchDecoder(c2q, nq)
    batch = np.asarray(dec.batch_decode(syn), np.uint8).reshape(len(syn), nq)
    single = _per_shot(dec, syn, nq)

    assert np.array_equal(batch, single), f"CPUBatch != single on {code_name}"
    # Faithful: (H @ correction) & 1 == syndrome, for every shot.
    assert np.array_equal((H @ batch.T).T & 1, syn), f"CPUBatch unfaithful on {code_name}"


@pytest.mark.parametrize("code_name,code", _codes())
def test_blossom_batch_bit_identical_to_single(code_name, code):
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    syn = _make_syndromes(code, 128, seed=54321)

    dec = qd.BlossomDecoder(c2q, nq)
    batch = np.asarray(dec.batch_decode(syn), np.uint8).reshape(len(syn), nq)
    single = _per_shot(dec, syn, nq)

    assert np.array_equal(batch, single), f"Blossom batch != single on {code_name}"
    assert np.array_equal((H @ batch.T).T & 1, syn), f"Blossom unfaithful on {code_name}"
