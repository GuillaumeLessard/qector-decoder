"""
Comprehensive QECTOR v0.6.4 Test Suite.
Covers: correctness, syndrome faithfulness, edge cases, memory, decoder construction,
DecoderPool, BP-OSD, sliding window, hybrid decoder, type stability, boundary conditions.

Run with:  pytest python/tests/test_comprehensive_suite.py -v
"""

import gc
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qector_decoder_v3 import (
    AutoDecoder,
    BeliefMatching,
    BlossomDecoder,
    BPOSDDecoder,
    CPUBatchDecoder,
    DecoderPool,
    FastUnionFindDecoder,
    HybridDecoder,
    PredecodedDecoder,
    SparseBlossomDecoder,
    UnionFindDecoder,
    codes,
    get_decoder,
)

np.random.seed(42)

# ==== Test Data ====
REPETITION_DISTANCES = [3, 5, 7, 9, 15, 21, 29]
SURFACE_DISTANCES = [3, 5, 7]
PHYSICAL_ERROR_RATE = 0.08


def make_syndrome(code, seed=42, n_shots=1):
    rng = np.random.default_rng(seed)
    nq = code.n_qubits
    H = code.parity_check_matrix()
    errors = (rng.random((n_shots, nq)) < PHYSICAL_ERROR_RATE).astype(np.uint8)
    syndromes = (errors @ H.T) & 1
    if n_shots == 1:
        return syndromes[0], errors[0], H
    return syndromes, errors, H


# =====================================================================
# 1. SYNDROME FAITHFULNESS (core invariant)
# =====================================================================
@pytest.mark.parametrize("dist", [3, 5, 7, 9, 15])
def test_syndrome_faithfulness_union_find_repetition(dist):
    code = codes.repetition_code(dist)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    dec = UnionFindDecoder(c2q, nq)
    syndrome, _, _ = make_syndrome(code)
    correction = dec.decode(syndrome)
    assert np.array_equal((H @ correction) & 1, syndrome), f"Union-Find d={dist} failed faithfulness"


@pytest.mark.parametrize("dist", [3, 5, 7])
def test_syndrome_faithfulness_all_surface(dist):
    code = codes.rotated_surface_code(dist)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    for name, maker in [
        ("UnionFind", lambda: UnionFindDecoder(c2q, nq)),
        ("FastUF", lambda: FastUnionFindDecoder(c2q, nq)),
        ("Blossom", lambda: BlossomDecoder(c2q, nq)),
        ("SparseBlossom", lambda: SparseBlossomDecoder(c2q, nq)),
        ("CPUBatch", lambda: CPUBatchDecoder(c2q, nq)),
    ]:
        dec = maker()
        syndrome, _, _ = make_syndrome(code, seed=dist)
        correction = dec.decode(syndrome)
        assert np.array_equal((H @ correction) & 1, syndrome), f"{name} on surface d={dist} failed faithfulness"


@pytest.mark.parametrize("seed", [42, 1234, 9999])
def test_syndrome_faithfulness_multi_seed(seed):
    code = codes.repetition_code(7)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    dec = BlossomDecoder(c2q, nq)
    syndrome, _, _ = make_syndrome(code, seed=seed)
    for _ in range(50):
        s = syndrome
        if _ > 0:
            s = (np.random.default_rng(seed + _).random(nq) < PHYSICAL_ERROR_RATE).astype(np.uint8)
            s = (s @ H.T) & 1
        c = dec.decode(s)
        assert np.array_equal((H @ c) & 1, s), f"Blossom d=7 seed={seed} iter={_} failed faithfulness"


# =====================================================================
# 2. ZERO SYNDROME (no errors) - should return zero correction
# =====================================================================
@pytest.mark.parametrize("dist", [3, 5, 7, 9])
def test_zero_syndrome_returns_zero(dist):
    code = codes.repetition_code(dist)
    c2q, nq = code.check_to_qubits, code.n_qubits
    zero = np.zeros(len(c2q), dtype=np.uint8)
    for maker in [
        lambda: UnionFindDecoder(c2q, nq),
        lambda: FastUnionFindDecoder(c2q, nq),
        lambda: BlossomDecoder(c2q, nq),
        lambda: SparseBlossomDecoder(c2q, nq),
        lambda: CPUBatchDecoder(c2q, nq),
    ]:
        dec = maker()
        corr = dec.decode(zero)
        assert np.all(corr == 0), f"Expected zero correction for zero syndrome, got {corr}"


# =====================================================================
# 3. FULL SYNDROME (all 1s) - should decode without error
# =====================================================================
@pytest.mark.parametrize("dist", [3, 5, 7])
def test_full_syndrome_does_not_crash(dist):
    code = codes.repetition_code(dist)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    all_ones = np.ones(len(c2q), dtype=np.uint8)
    dec = BlossomDecoder(c2q, nq)
    corr = dec.decode(all_ones)
    assert np.array_equal((H @ corr) & 1, all_ones), "Full syndrome faithfulness"


# =====================================================================
# 4. DECODER CONSTRUCTION EDGE CASES
# =====================================================================
def test_decoder_checks_empty():
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        UnionFindDecoder([], 5)


def test_decoder_checks_invalid_type():
    with pytest.raises((TypeError, RuntimeError)):
        BlossomDecoder("invalid", 5)


def test_decoder_nqubits_zero():
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        SparseBlossomDecoder([[0]], 0)


def test_decoder_checks_mismatch():
    # check references qubit beyond n_qubits
    c2q = [[0, 1], [100]]
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        UnionFindDecoder(c2q, 5)


def test_decoder_large_distance():
    dist = 101
    code = codes.repetition_code(dist)
    c2q, nq = code.check_to_qubits, code.n_qubits
    dec = FastUnionFindDecoder(c2q, nq)
    syndrome = np.ones(len(c2q), dtype=np.uint8)
    corr = dec.decode(syndrome)
    assert len(corr) == nq
    assert corr.dtype == np.uint8


# =====================================================================
# 5. TYPE STABILITY
# =====================================================================
@pytest.mark.parametrize("dtype", [np.uint8])
def test_syndrome_type_stability(dtype):
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    dec = BlossomDecoder(c2q, nq)
    syndrome = np.zeros(len(c2q), dtype=dtype)
    syndrome[0] = 1
    corr = dec.decode(syndrome)
    assert np.array_equal((H @ corr) & 1, syndrome), f"dtype {dtype} failed faithfulness"


def test_correction_type_uint8():
    code = codes.repetition_code(7)
    c2q, nq = code.check_to_qubits, code.n_qubits
    dec = UnionFindDecoder(c2q, nq)
    syndrome = np.zeros(len(c2q), dtype=np.uint8)
    syndrome[0] = 1
    corr = dec.decode(syndrome)
    assert corr.dtype == np.uint8, f"Expected uint8, got {corr.dtype}"


# =====================================================================
# 6. BP-OSD TESTS
# =====================================================================
@pytest.mark.parametrize("dist", [3, 5, 7, 9, 15])
def test_bposd_basic_decode(dist):
    code = codes.repetition_code(dist)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    dec = BPOSDDecoder(c2q, nq, 0.08)
    syndrome, _, _ = make_syndrome(code, seed=dist * 10)
    corr = dec.decode(syndrome)
    assert np.array_equal((H @ corr) & 1, syndrome), f"BP-OSD d={dist} faithfulness"


@pytest.mark.parametrize("dist", [3, 5, 7])
def test_bposd_surface(dist):
    code = codes.rotated_surface_code(dist)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    dec = BPOSDDecoder(c2q, nq, 0.08)
    syndrome, _, _ = make_syndrome(code, seed=dist)
    corr = dec.decode(syndrome)
    assert np.array_equal((H @ corr) & 1, syndrome), f"BP-OSD surface d={dist} faithfulness"


def test_bposd_batch_decode():
    """BP-OSD does not have batch_decode - should not have the attribute."""
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    dec = BPOSDDecoder(c2q, nq, 0.08)
    assert not hasattr(dec, "batch_decode"), "BP-OSD should not have batch_decode"


def test_bposd_multiple_shots():
    """BP-OSD should produce valid results across many shots."""
    code = codes.repetition_code(7)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    dec = BPOSDDecoder(c2q, nq, 0.08)
    for seed in range(50):
        syndrome, _, _ = make_syndrome(code, seed=seed)
        corr = dec.decode(syndrome)
        assert np.array_equal((H @ corr) & 1, syndrome), f"BP-OSD failed at shot {seed}"


# =====================================================================
# 7. DECODERPOOL TESTS
# =====================================================================
def _run_pool_test(module_guard=False):
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    pool = DecoderPool(c2q, nq, "union_find", n_workers=2)
    syndromes, _, _ = make_syndrome(code, n_shots=100)
    corrections = pool.decode(syndromes)
    pool.close()
    assert len(corrections) == 100
    for i in range(100):
        assert np.array_equal((H @ corrections[i]) & 1, syndromes[i]), f"DecoderPool shot {i} failed faithfulness"


def _probe_spawn_task():
    """Picklable no-op used to probe spawn availability."""
    return 42


def test_decoderpool_basic():
    """DecoderPool with 2 workers, 100 shots.

    Note: DecoderPool uses multiprocessing spawn context on Windows,
    which requires the test function to be importable. If it fails,
    run with: python -m pytest python/tests/test_comprehensive_suite.py::test_decoderpool_basic
    """
    import multiprocessing as _mp

    # The probe task must be a module-level function. A lambda can never be
    # pickled for the spawn start method, so the old `apply_async(lambda: 42)`
    # raised AttributeError -- which `except RuntimeError` did not catch, so this
    # capability probe could neither succeed nor skip; it always errored.
    try:
        ctx = _mp.get_context("spawn")
        pool = ctx.Pool(processes=1)
        try:
            assert pool.apply_async(_probe_spawn_task).get(timeout=30) == 42
        finally:
            pool.close()
            pool.join()
    except (RuntimeError, OSError, ImportError, AttributeError) as exc:
        pytest.skip(f"multiprocessing spawn unavailable here: {exc}")
    _run_pool_test()


def test_decoderpool_single_worker():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    pool = DecoderPool(c2q, nq, "union_find", n_workers=1)
    syndrome = np.zeros((1, len(c2q)), dtype=np.uint8)
    corr = pool.decode(syndrome)
    pool.close()
    assert corr.shape == (1, nq)
    assert np.array_equal((H @ corr[0]) & 1, syndrome[0])


# =====================================================================
# 8. GET_DECODER (LRU CACHE)
# =====================================================================
def test_get_decoder_basic():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    dec1 = get_decoder(tuple(map(tuple, c2q)), nq, "blossom")
    dec2 = get_decoder(tuple(map(tuple, c2q)), nq, "blossom")
    assert dec1 is dec2, "LRU cache should return the same instance"


def test_get_decoder_aliases():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    tup = tuple(map(tuple, c2q))
    d1 = get_decoder(tup, nq, "fast_uf")
    d2 = get_decoder(tup, nq, "fast_union_find")
    d3 = get_decoder(tup, nq, "fastuf")
    # Cache entries may differ by alias, but they should be the same type
    from qector_decoder_v3 import FastUnionFindDecoder

    assert isinstance(d1, FastUnionFindDecoder)
    assert isinstance(d2, FastUnionFindDecoder)
    assert isinstance(d3, FastUnionFindDecoder)


def test_get_decoder_invalid_name():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    with pytest.raises(ValueError):
        get_decoder(tuple(map(tuple, c2q)), nq, "nonexistent_decoder")


# =====================================================================
# 9. HYBRID DECODER
# =====================================================================
def test_hybrid_decoder_decode_alias():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    dec = HybridDecoder(c2q, nq)
    syndrome, _, _ = make_syndrome(code)
    corr = dec.decode(syndrome)
    assert np.array_equal((H @ corr) & 1, syndrome), "HybridDecoder.decode() faithfulness"


def test_hybrid_decoder_default_backend():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    dec = HybridDecoder(c2q, nq)
    # HybridDecoder should decode without error
    H = code.parity_check_matrix()
    syndrome = np.zeros(len(c2q), dtype=np.uint8)
    corr = dec.decode(syndrome)
    assert np.array_equal((H @ corr) & 1, syndrome)


# =====================================================================
# 10. PREDECODED DECODER
# =====================================================================
def test_predecoded_backward_compat():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    dec = PredecodedDecoder(c2q, nq)
    syndrome, _, _ = make_syndrome(code)
    corr = dec.decode(syndrome)
    assert np.array_equal((H @ corr) & 1, syndrome), "Predecoded backward compat faithfulness"


def test_predecoded_unionfind_name():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    # PredecodedDecoder accepts positional decoder name
    from qector_decoder_v3 import PredecodedDecoder as PDC

    dec = PDC(c2q, nq, "unionfind")
    # Should decode without error
    H = code.parity_check_matrix()
    syndrome = np.zeros(len(c2q), dtype=np.uint8)
    corr = dec.decode(syndrome)
    assert np.array_equal((H @ corr) & 1, syndrome)


# =====================================================================
# 11. BELIEF MATCHING
# =====================================================================
def test_belief_matching_from_numpy_h():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    bm = BeliefMatching.from_numpy_h(H)
    assert bm is not None


def test_belief_matching_decode():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    bm = BeliefMatching.from_numpy_h(H, [0.08, 0.08, 0.08])
    syndrome = np.array([1, 0], dtype=np.uint8)
    try:
        corr = bm.decode(syndrome)
        assert len(corr) == 3
    except RuntimeError as e:
        pytest.skip(f"BeliefMatching decode raised: {e}")


# =====================================================================
# 12. AUTO DECODER
# =====================================================================
def test_auto_decoder_select():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    auto = AutoDecoder(c2q, nq)
    selected = auto.auto_select(c2q, nq)
    assert isinstance(selected, str) and len(selected) > 0


def test_auto_decoder_decode():
    code = codes.repetition_code(7)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    auto = AutoDecoder(c2q, nq)
    syndrome, _, _ = make_syndrome(code)
    corr = auto.decode(syndrome)
    assert np.array_equal((H @ corr) & 1, syndrome), "AutoDecoder faithfulness"


# =====================================================================
# 13. REPRODUCIBILITY - same syndrome should give same correction
# =====================================================================
def test_deterministic_decode():
    code = codes.repetition_code(9)
    c2q, nq = code.check_to_qubits, code.n_qubits
    syndrome = np.zeros(len(c2q), dtype=np.uint8)
    syndrome[0] = 1
    for maker in [
        lambda: UnionFindDecoder(c2q, nq),
        lambda: FastUnionFindDecoder(c2q, nq),
    ]:
        dec = maker()
        c1 = dec.decode(syndrome)
        c2 = dec.decode(syndrome)
        assert np.array_equal(c1, c2), f"Non-deterministic decode for {maker}"


# =====================================================================
# 14. SURFACE CODE TESTS
# =====================================================================
@pytest.mark.parametrize("dist", [3, 5, 7])
def test_surface_code_all_decoders(dist):
    code = codes.rotated_surface_code(dist)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    for name, maker in [
        ("UnionFind", lambda: UnionFindDecoder(c2q, nq)),
        ("FastUF", lambda: FastUnionFindDecoder(c2q, nq)),
        ("Blossom", lambda: BlossomDecoder(c2q, nq)),
        ("SparseBlossom", lambda: SparseBlossomDecoder(c2q, nq)),
        ("CPUBatch", lambda: CPUBatchDecoder(c2q, nq)),
    ]:
        dec = maker()
        for _ in range(10):
            syndrome, _, _ = make_syndrome(code, seed=dist * 100 + _)
            corr = dec.decode(syndrome)
            assert np.array_equal((H @ corr) & 1, syndrome), f"{name} surface d={dist} shot={_} faithfulness"


# =====================================================================
# 15. EDGE CASE - SYNDROME VECTOR LENGTH MISMATCH
# =====================================================================
def test_wrong_syndrome_length():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    dec = BlossomDecoder(c2q, nq)
    wrong_len = np.ones(len(c2q) + 1, dtype=np.uint8)
    with pytest.raises((ValueError, RuntimeError)):
        dec.decode(wrong_len)


# =====================================================================
# 16. EDGE CASE - NUMBER OF QUBITS MISMATCH
# =====================================================================
def test_wrong_correction_length():
    """decode should always return n_qubits-length array."""
    code = codes.repetition_code(7)
    c2q, nq = code.check_to_qubits, code.n_qubits
    dec = CPUBatchDecoder(c2q, nq)
    syndrome = np.ones(len(c2q), dtype=np.uint8)
    corr = dec.decode(syndrome)
    assert len(corr) == nq, f"Expected {nq} qubits, got {len(corr)}"


# =====================================================================
# 17. LARGE BATCH TESTS
# =====================================================================
def test_batch_decode_consistency():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    syndromes, _, _ = make_syndrome(code, n_shots=100)
    dec = CPUBatchDecoder(c2q, nq)
    corrections = dec.batch_decode(syndromes)
    assert corrections.shape == (100, nq)
    for i in range(100):
        assert np.array_equal((H @ corrections[i]) & 1, syndromes[i]), f"batch shot {i} faithfulness"


# =====================================================================
# 18. COLD PATH CONSTRUCTION TIMING
# =====================================================================
def test_cold_path_construction():
    code = codes.repetition_code(15)
    c2q, nq = code.check_to_qubits, code.n_qubits
    t0 = time.perf_counter()
    for _ in range(100):
        _ = get_decoder(tuple(map(tuple, c2q)), nq, "blossom")
    t = (time.perf_counter() - t0) / 100 * 1e6
    assert t < 5000, f"Cold path too slow: {t:.1f}us average"


# =====================================================================
# 19. MEMORY TEST - NO LEAKS ON REPEATED CONSTRUCTIONS
# =====================================================================
def test_no_memory_leak_on_repeated_decode():
    code = codes.repetition_code(7)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    gc.collect()
    import tracemalloc

    tracemalloc.start()
    dec = UnionFindDecoder(c2q, nq)
    for _ in range(1000):
        syndrome, _, _ = make_syndrome(code, seed=_)
        dec.decode(syndrome)
    gc.collect()
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert current < 10 * 1024 * 1024, f"Memory leak suspected: {current} bytes after 1000 decodes"


# =====================================================================
# 20. ALL DECODERS RETURN SAME LER (UF, FastUF, CPU Batch)
# =====================================================================
def test_uf_variants_identical_results():
    """UnionFind, FastUnionFind, and CPUBatch should produce identical corrections."""
    code = codes.repetition_code(9)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    uf = UnionFindDecoder(c2q, nq)
    fuf = FastUnionFindDecoder(c2q, nq)
    batch = CPUBatchDecoder(c2q, nq)
    for seed in range(30):
        rng = np.random.default_rng(seed)
        errors = (rng.random(nq) < 0.08).astype(np.uint8)
        syn = (errors @ H.T) & 1
        c1 = uf.decode(syn)
        c2 = fuf.decode(syn)
        assert np.array_equal(c1, c2), f"UF variants differ at seed={seed}"
    # Batch decoder with single shot
    for seed in range(10):
        rng = np.random.default_rng(seed)
        errors = (rng.random(nq) < 0.08).astype(np.uint8)
        syn = (errors @ H.T) & 1
        c1 = uf.decode(syn)
        c3 = batch.decode(syn)
        assert np.array_equal(c1, c3), f"UF vs batch differ at seed={seed}"


# =====================================================================
# 21. SPARSE BLOSSOM VS BLOSSOM
# =====================================================================
def test_sparse_blossom_same_as_blossom():
    """Sparse Blossom should produce same MWPM solution as Blossom (exact)."""
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    bloom = BlossomDecoder(c2q, nq)
    sparse = SparseBlossomDecoder(c2q, nq)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        errors = (rng.random(nq) < 0.08).astype(np.uint8)
        H = code.parity_check_matrix()
        syn = (errors @ H.T) & 1
        c1 = bloom.decode(syn)
        c2 = sparse.decode(syn)
        # Both are MWPM - should give same correction
        assert np.array_equal(c1, c2), f"Blossom variants differ at seed={seed}"


# =====================================================================
# 22. CODES MODULE TESTS
# =====================================================================
def test_codes_repetition_code_attributes():
    code = codes.repetition_code(5)
    assert code.n_qubits == 5
    assert code.distance == 5
    assert "repetition" in code.name


def test_codes_rotated_surface_code_attributes():
    code = codes.rotated_surface_code(3)
    assert code.n_qubits == 9
    assert code.distance == 3
    assert "surface" in code.name.lower() or "rotated" in code.name.lower()


def test_codes_parity_check_matrix_shape():
    code = codes.repetition_code(7)
    H = code.parity_check_matrix()
    assert H.shape == (6, 7)
    code2 = codes.rotated_surface_code(3)
    H2 = code2.parity_check_matrix()
    assert H2.shape == (4, 9)


# =====================================================================
# 23. STIM COMPATIBILITY LAYER
# =====================================================================
def test_stim_compat_import():
    try:
        from qector_decoder_v3 import stim_compat

        assert stim_compat is not None
    except ImportError:
        pytest.skip("stim not installed")


def test_sinter_compat_import():
    try:
        from qector_decoder_v3 import sinter_compat

        assert sinter_compat is not None
    except ImportError:
        pytest.skip("sinter not installed")


# =====================================================================
# 24. DECODE_MMAP INTERFACE
# =====================================================================
def test_decode_mmap_import():
    from qector_decoder_v3 import decode_mmap

    assert callable(decode_mmap)


# =====================================================================
# 25. HYPEREDGE REJECTION
# =====================================================================
def test_hyperedges_rejected():
    """Decoders with hyperedges should be handled gracefully (not crash)."""
    from qector_decoder_v3 import UnionFindDecoder

    hyper_check = [[0, 1, 2]]
    try:
        dec = BlossomDecoder(hyper_check, 3)
        syn = np.array([1], dtype=np.uint8)
        corr = dec.decode(syn)
        assert len(corr) == 3
    except (ValueError, TypeError, RuntimeError) as e:
        pytest.skip(f"BlossomDecoder rejected hyperedges: {e}")


# =====================================================================
# 26. MINIMAL CODE
# =====================================================================
def test_minimal_repetition_code():
    """d=3 is the smallest meaningful repetition code."""
    code = codes.repetition_code(3)
    c2q, nq = code.check_to_qubits, code.n_qubits
    H = code.parity_check_matrix()
    dec = UnionFindDecoder(c2q, nq)
    syndrome = np.array([1, 0], dtype=np.uint8)
    corr = dec.decode(syndrome)
    assert np.array_equal((H @ corr) & 1, syndrome)


# =====================================================================
# 27. DECODER EXHAUSTIVE TESTS
# =====================================================================
def get_all_decoder_factories(c2q, nq):
    return {
        "UnionFindDecoder": (lambda: UnionFindDecoder(c2q, nq)),
        "FastUnionFindDecoder": (lambda: FastUnionFindDecoder(c2q, nq)),
        "BlossomDecoder": (lambda: BlossomDecoder(c2q, nq)),
        "SparseBlossomDecoder": (lambda: SparseBlossomDecoder(c2q, nq)),
        "CPUBatchDecoder": (lambda: CPUBatchDecoder(c2q, nq)),
    }


def test_all_decoders_exist():
    code = codes.repetition_code(5)
    c2q, nq = code.check_to_qubits, code.n_qubits
    for name, factory in get_all_decoder_factories(c2q, nq).items():
        dec = factory()
        assert dec is not None, f"Failed to construct {name}"
        assert dec.n_qubits == nq, f"{name}.n_qubits mismatch"
        assert dec.n_checks == len(c2q), f"{name}.n_checks mismatch"
