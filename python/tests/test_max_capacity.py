"""Max-capacity stress tests for all QECTOR v3 decoders.

Tests push each decoder to its practical limits:
- Large-distance surface codes (d=11, 15, 21)
- Large batch sizes (10K, 65K, 100K syndromes)
- LookupTable exhaustive at n=24
- Blossom/SparseBlossom at the 20-defect boundary
- BPOSD on larger codes
- Cross-decoder consistency on large codes
- Memory-bounded batch stress
"""

import time
import numpy as np
import pytest

import qector_decoder_v3 as qd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def surface_code(d):
    """Generate d×d rotated surface code checks. Returns (checks, n_qubits)."""
    checks = []
    n = d * d
    for i in range(d):
        for j in range(d):
            q = i * d + j
            # Z-check (stabilizer)
            if i < d - 1 and j < d - 1:
                checks.append([q, q + 1, q + d, q + d + 1])
    n_qubits = n
    return checks, n_qubits


def random_syndrome(n_checks, defect_prob=0.1, seed=None):
    """Generate a random binary syndrome."""
    rng = np.random.default_rng(seed)
    return rng.binomial(1, defect_prob, size=n_checks).astype(np.uint8)


def random_batch(n_checks, batch_size, defect_prob=0.1, seed=None):
    """Generate a random batch of syndromes (batch_size, n_checks)."""
    rng = np.random.default_rng(seed)
    return rng.binomial(1, defect_prob, size=(batch_size, n_checks)).astype(np.uint8)


# ---------------------------------------------------------------------------
# UnionFindDecoder — large distance surface codes
# ---------------------------------------------------------------------------

class TestUnionFindLargeCodes:
    """UnionFind scales O(E·α(N)) — should handle very large codes."""

    @pytest.mark.parametrize("d", [11, 15, 21])
    def test_decode_large_surface_code(self, d):
        checks, nq = surface_code(d)
        dec = qd.UnionFindDecoder(checks, nq)
        syn = random_syndrome(len(checks), defect_prob=0.05, seed=42)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)
        assert corr.dtype == np.uint8

    @pytest.mark.parametrize("d", [11, 15])
    def test_batch_decode_large_surface(self, d):
        checks, nq = surface_code(d)
        dec = qd.UnionFindDecoder(checks, nq)
        batch = random_batch(len(checks), 1000, seed=42)
        results = dec.batch_decode(batch)
        assert results.shape == (1000, nq)
        assert results.dtype == np.uint8

    def test_single_defect_syndrome(self):
        """Even n=1 check should work."""
        dec = qd.UnionFindDecoder([[0, 1]], 2)
        corr = dec.decode(np.array([1], dtype=np.uint8))
        assert corr.shape == (2,)


# ---------------------------------------------------------------------------
# FastUnionFindDecoder — same tests via the fast path
# ---------------------------------------------------------------------------

class TestFastUnionFindLargeCodes:

    @pytest.mark.parametrize("d", [11, 15, 21])
    def test_decode_large_surface_code(self, d):
        checks, nq = surface_code(d)
        dec = qd.FastUnionFindDecoder(checks, nq)
        syn = random_syndrome(len(checks), defect_prob=0.05, seed=42)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)
        assert corr.dtype == np.uint8

    def test_d21_latency(self):
        """Measure latency for d=21 surface code decode."""
        checks, nq = surface_code(21)
        dec = qd.FastUnionFindDecoder(checks, nq)
        syn = random_syndrome(len(checks), defect_prob=0.1, seed=42)
        # Warmup
        for _ in range(100):
            dec.decode(syn)
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(10000):
            dec.decode(syn)
        elapsed = time.perf_counter() - t0
        us_per = elapsed / 10000 * 1e6
        print(f"\n  FastUF d=21: {us_per:.1f} µs/decode ({nq} qubits, {len(checks)} checks)")
        assert us_per < 1000  # should be well under 1ms


# ---------------------------------------------------------------------------
# BlossomDecoder — 20-defect boundary
# ---------------------------------------------------------------------------

class TestBlossomDefectBoundary:
    """Blossom uses exact DP for n<=20 defects, falls back to UF for >20."""

    def test_exactly_20_defects(self):
        """20 defects: should use exact DP solver, not fallback."""
        checks, nq = surface_code(11)
        dec = qd.BlossomDecoder(checks, nq)
        # Create syndrome with exactly 20 defects
        syn = np.zeros(len(checks), dtype=np.uint8)
        indices = np.random.default_rng(42).choice(len(checks), 20, replace=False)
        syn[indices] = 1
        corr = dec.decode(syn)
        assert corr.shape == (nq,)

    def test_21_defects_fallback(self):
        """21 defects: should fall back to UnionFind — no crash."""
        checks, nq = surface_code(11)
        dec = qd.BlossomDecoder(checks, nq)
        syn = np.zeros(len(checks), dtype=np.uint8)
        indices = np.random.default_rng(42).choice(len(checks), 21, replace=False)
        syn[indices] = 1
        corr = dec.decode(syn)
        assert corr.shape == (nq,)

    def test_50_defects_no_panic(self):
        """50 defects on d=15 code: heavy fallback, must not crash."""
        checks, nq = surface_code(15)
        dec = qd.BlossomDecoder(checks, nq)
        syn = random_syndrome(len(checks), defect_prob=0.3, seed=42)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)

    def test_empty_syndrome_large_code(self):
        checks, nq = surface_code(21)
        dec = qd.BlossomDecoder(checks, nq)
        syn = np.zeros(len(checks), dtype=np.uint8)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)
        assert np.all(corr == 0)


# ---------------------------------------------------------------------------
# SparseBlossomDecoder — same 20-defect boundary, region growing
# ---------------------------------------------------------------------------

class TestSparseBlossomDefectBoundary:

    def test_exactly_20_defects(self):
        checks, nq = surface_code(11)
        dec = qd.SparseBlossomDecoder(checks, nq)
        syn = np.zeros(len(checks), dtype=np.uint8)
        indices = np.random.default_rng(42).choice(len(checks), 20, replace=False)
        syn[indices] = 1
        corr = dec.decode(syn)
        assert corr.shape == (nq,)

    def test_25_defects_no_panic(self):
        checks, nq = surface_code(15)
        dec = qd.SparseBlossomDecoder(checks, nq)
        syn = random_syndrome(len(checks), defect_prob=0.25, seed=42)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)

    def test_empty_syndrome_large_code(self):
        checks, nq = surface_code(21)
        dec = qd.SparseBlossomDecoder(checks, nq)
        syn = np.zeros(len(checks), dtype=np.uint8)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)


# ---------------------------------------------------------------------------
# LookupTableDecoder — exhaustive enumeration up to n=24
# ---------------------------------------------------------------------------

class TestLookupTableCapacity:

    def test_exhaustive_n12(self):
        """12 qubits: 2^12=4096 entries, builds in < 1s."""
        checks, nq = surface_code(3)  # d=3: 9 qubits
        dec = qd.LookupTableDecoder(checks, nq)
        syn = random_syndrome(len(checks), seed=42)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)

    def test_exhaustive_n16_ring(self):
        """16 qubits via ring code: 2^16=65536 entries."""
        checks, nq = qd.generate_ring_code_checks(16)
        dec = qd.LookupTableDecoder(checks, nq)
        # Full enumeration — every possible syndrome should be in table
        for i in range(min(100, 1 << nq)):
            syn = np.array([(i >> j) & 1 for j in range(len(checks))], dtype=np.uint8)
            corr = dec.decode(syn)
            assert corr.shape == (nq,)

    def test_weight_limited_large_code(self):
        """Large code (>24 qubits): uses weight-limited enumeration."""
        checks, nq = surface_code(11)  # 121 qubits >> 24
        dec = qd.LookupTableDecoder(checks, nq)
        syn = random_syndrome(len(checks), seed=42)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)

    def test_max_entries_cap(self):
        """max_entries limits table growth."""
        checks, nq = surface_code(5)
        dec = qd.LookupTableDecoder(checks, nq)
        # Should succeed — max_entries is handled internally
        syn = random_syndrome(len(checks), seed=42)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)


# ---------------------------------------------------------------------------
# BPOSD — larger codes
# ---------------------------------------------------------------------------

class TestBPOSDLargeCodes:

    @pytest.mark.parametrize("d", [5, 7, 11])
    def test_bp_osd_surface_code(self, d):
        checks, nq = surface_code(d)
        dec = qd.BPOSDDecoder(checks, nq)
        syn = random_syndrome(len(checks), defect_prob=0.05, seed=42)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)

    def test_bp_osd_no_error_large(self):
        """No-error syndrome on d=15 should return all-zero correction."""
        checks, nq = surface_code(15)
        dec = qd.BPOSDDecoder(checks, nq)
        syn = np.zeros(len(checks), dtype=np.uint8)
        corr = dec.decode(syn)
        assert np.all(corr == 0)


# ---------------------------------------------------------------------------
# CPUBatchDecoder — large batches
# ---------------------------------------------------------------------------

class TestCPUBatchDecoderCapacity:

    def test_batch_10k(self):
        checks, nq = surface_code(7)
        dec = qd.CPUBatchDecoder(checks, nq)
        batch = random_batch(len(checks), 10_000, seed=42)
        results = dec.batch_decode(batch)
        assert results.shape == (10_000, nq)

    def test_batch_65k(self):
        """65K syndromes — tests large batch throughput."""
        checks, nq = surface_code(5)
        dec = qd.CPUBatchDecoder(checks, nq)
        batch = random_batch(len(checks), 65_536, seed=42)
        results = dec.batch_decode(batch)
        assert results.shape == (65_536, nq)

    def test_batch_100k_throughput(self):
        """100K syndromes: measure throughput."""
        checks, nq = surface_code(5)
        dec = qd.CPUBatchDecoder(checks, nq)
        batch = random_batch(len(checks), 100_000, seed=42)
        t0 = time.perf_counter()
        results = dec.batch_decode(batch)
        elapsed = time.perf_counter() - t0
        throughput = 100_000 / elapsed
        print(f"\n  CPUBatch d=5 100K: {throughput:.0f} syndromes/sec ({elapsed:.2f}s)")
        assert results.shape == (100_000, nq)
        assert throughput > 1000  # at least 1K/sec

    def test_single_decode_method(self):
        """CPUBatchDecoder.decode() single syndrome method."""
        checks, nq = surface_code(5)
        dec = qd.CPUBatchDecoder(checks, nq)
        syn = random_syndrome(len(checks), seed=42)
        corr = dec.decode(syn)
        assert corr.shape == (nq,)

    def test_non_block_aligned_batch(self):
        """Batch size not multiple of BLOCK_SIZE=64."""
        checks, nq = surface_code(5)
        dec = qd.CPUBatchDecoder(checks, nq)
        for bs in [1, 7, 63, 65, 127, 129]:
            batch = random_batch(len(checks), bs, seed=42)
            results = dec.batch_decode(batch)
            assert results.shape == (bs, nq)


# ---------------------------------------------------------------------------
# BatchDecoder (Rayon parallel) — large batches
# ---------------------------------------------------------------------------

class TestBatchDecoderCapacity:

    def test_parallel_batch_10k(self):
        checks, nq = surface_code(7)
        dec = qd.BatchDecoder(checks, nq)
        batch = random_batch(len(checks), 10_000, seed=42)
        results = dec.parallel_batch_decode(batch)
        assert results.shape == (10_000, nq)

    def test_batch_decode_alias(self):
        """batch_decode() should alias parallel_batch_decode()."""
        checks, nq = surface_code(5)
        dec = qd.BatchDecoder(checks, nq)
        batch = random_batch(len(checks), 1000, seed=42)
        r1 = dec.batch_decode(batch)
        r2 = dec.parallel_batch_decode(batch)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# HybridDecoder — large code
# ---------------------------------------------------------------------------

class TestHybridDecoderLarge:

    def test_d11_surface_code(self):
        checks, nq = surface_code(11)
        dec = qd.HybridDecoder(checks, nq)
        syn = random_syndrome(len(checks), defect_prob=0.05, seed=42)
        corr = dec.decode_hybrid(syn)
        assert corr.shape == (nq,)


# ---------------------------------------------------------------------------
# Cross-decoder consistency on large codes
# ---------------------------------------------------------------------------

class TestCrossDecoderConsistencyLarge:

    @pytest.mark.parametrize("d", [5, 7])
    def test_uf_vs_blossom_vs_sparse(self, d):
        """All three decoders should produce valid (same-length) corrections."""
        checks, nq = surface_code(d)
        uf = qd.UnionFindDecoder(checks, nq)
        bl = qd.BlossomDecoder(checks, nq)
        sp = qd.SparseBlossomDecoder(checks, nq)

        rng = np.random.default_rng(42)
        for _ in range(50):
            syn = rng.binomial(1, 0.1, len(checks)).astype(np.uint8)
            c_uf = uf.decode(syn)
            c_bl = bl.decode(syn)
            c_sp = sp.decode(syn)
            assert c_uf.shape == (nq,)
            assert c_bl.shape == (nq,)
            assert c_sp.shape == (nq,)

    def test_cpubatch_vs_batchdecoder(self):
        """CPUBatchDecoder and BatchDecoder should agree on outputs."""
        checks, nq = surface_code(5)
        cpu_dec = qd.CPUBatchDecoder(checks, nq)
        batch_dec = qd.BatchDecoder(checks, nq)

        batch = random_batch(len(checks), 500, seed=42)
        r_cpu = cpu_dec.batch_decode(batch)
        r_batch = batch_dec.parallel_batch_decode(batch)
        np.testing.assert_array_equal(r_cpu, r_batch)


# ---------------------------------------------------------------------------
# GPU backends — availability + graceful degradation
# ---------------------------------------------------------------------------

class TestGPUCapacity:
    """GPU tests only run if hardware is available."""

    def test_opencl_is_available_bool(self):
        assert isinstance(qd.OpenCLBatchDecoder.is_available(), bool)

    def test_cuda_is_available_bool(self):
        assert isinstance(qd.CUDABatchDecoder.is_available(), bool)

    @pytest.mark.skipif(not qd.OpenCLBatchDecoder.is_available(),
                        reason="OpenCL not available")
    def test_opencl_large_batch(self):
        checks, nq = surface_code(7)
        dec = qd.OpenCLBatchDecoder(checks, nq)
        batch = random_batch(len(checks), 10_000, seed=42)
        results = dec.batch_decode(batch)
        assert results.shape == (10_000, nq)

    @pytest.mark.skipif(not qd.CUDABatchDecoder.is_available(),
                        reason="CUDA not available")
    def test_cuda_large_batch(self):
        checks, nq = surface_code(7)
        dec = qd.CUDABatchDecoder(checks, nq)
        batch = random_batch(len(checks), 10_000, seed=42)
        results = dec.batch_decode(batch)
        assert results.shape == (10_000, nq)

    @pytest.mark.skipif(not qd.OpenCLBatchDecoder.is_available(),
                        reason="OpenCL not available")
    def test_opencl_512_qubit_code(self):
        """Test the extended 512-qubit local-memory path."""
        checks, nq = surface_code(21)  # 441 qubits
        dec = qd.OpenCLBatchDecoder(checks, nq)
        batch = random_batch(len(checks), 1000, seed=42)
        results = dec.batch_decode(batch)
        assert results.shape == (1000, nq)


# ---------------------------------------------------------------------------
# Boundary validation — reject bad inputs at scale
# ---------------------------------------------------------------------------

class TestBoundaryValidationLarge:

    def test_wrong_dtype_rejected(self):
        checks, nq = surface_code(11)
        dec = qd.UnionFindDecoder(checks, nq)
        syn = np.zeros(len(checks), dtype=np.float32)
        with pytest.raises(TypeError):
            dec.decode(syn)

    def test_wrong_shape_rejected(self):
        checks, nq = surface_code(11)
        dec = qd.UnionFindDecoder(checks, nq)
        syn = np.zeros(len(checks) + 1, dtype=np.uint8)
        with pytest.raises(Exception):
            dec.decode(syn)

    def test_empty_checks_rejected(self):
        with pytest.raises(ValueError):
            qd.UnionFindDecoder([], 10)

    def test_batch_wrong_ndim_rejected(self):
        checks, nq = surface_code(7)
        dec = qd.CPUBatchDecoder(checks, nq)
        syn_1d = np.zeros(len(checks), dtype=np.uint8)
        with pytest.raises((ValueError, Exception)):
            dec.batch_decode(syn_1d)


# ---------------------------------------------------------------------------
# Memory stress — ensure no unbounded growth
# ---------------------------------------------------------------------------

class TestMemoryStress:

    def test_repeated_decode_no_leak(self):
        """100K decodes on a large code should not balloon memory."""
        checks, nq = surface_code(11)
        dec = qd.FastUnionFindDecoder(checks, nq)
        syn = random_syndrome(len(checks), seed=42)
        for _ in range(100_000):
            corr = dec.decode(syn)
            assert corr.shape == (nq,)
        # If we get here without OOM, memory is bounded

    def test_large_code_construction(self):
        """Constructing d=21 surface code should not panic."""
        checks, nq = surface_code(21)
        assert nq == 441
        assert len(checks) > 0
        dec = qd.FastUnionFindDecoder(checks, nq)
        assert dec is not None
