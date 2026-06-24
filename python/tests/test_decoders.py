"""Comprehensive tests for QECTOR decoder v3."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

import qector_decoder_v3 as qd


@pytest.fixture
def simple_checks():
    """Small linear check chain for deterministic testing."""
    return [[0, 1], [1, 2], [2, 3]], 4


class TestUnionFindDecoder:
    """Tests for the core Union-Find decoder."""

    def test_empty_syndrome(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.UnionFindDecoder(checks, n_qubits)
        syndrome = np.zeros(len(checks), dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert corr.shape == (n_qubits,)
        assert np.all(corr == 0)

    def test_single_active_check(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.UnionFindDecoder(checks, n_qubits)
        syndrome = np.array([1, 0, 0], dtype=np.uint8)
        corr = dec.decode(syndrome)
        # At least one of the two qubits in the active check must be marked
        assert (corr[0] == 1 or corr[1] == 1)

    def test_multiple_active_checks(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.UnionFindDecoder(checks, n_qubits)
        syndrome = np.array([1, 0, 1], dtype=np.uint8)
        corr = dec.decode(syndrome)
        # Check 0 (0,1) and check 2 (2,3) are active but disjoint
        # Only qubits in clusters with active checks should be corrected
        assert corr.shape == (n_qubits,)

    def test_batch_decode_shape(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.UnionFindDecoder(checks, n_qubits)
        syndromes = np.random.randint(0, 2, size=(10, len(checks)), dtype=np.uint8)
        corr = dec.batch_decode(syndromes)
        assert corr.shape == (10, n_qubits)

    def test_batch_decode_consistency(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.UnionFindDecoder(checks, n_qubits)
        syndromes = np.random.randint(0, 2, size=(5, len(checks)), dtype=np.uint8)
        batch_corr = dec.batch_decode(syndromes)
        for i in range(5):
            single_corr = dec.decode(syndromes[i])
            assert np.array_equal(batch_corr[i], single_corr)

    def test_invalid_shape(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.UnionFindDecoder(checks, n_qubits)
        with pytest.raises(Exception):
            dec.decode(np.zeros(5, dtype=np.uint8))

    def test_invalid_dtype(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.UnionFindDecoder(checks, n_qubits)
        with pytest.raises(Exception):
            dec.decode(np.zeros(len(checks), dtype=np.float32))

    def test_n_qubits_property(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.UnionFindDecoder(checks, n_qubits)
        assert dec.n_qubits == n_qubits
        assert dec.n_checks == len(checks)

    @given(st.lists(st.integers(0, 1), min_size=3, max_size=3))
    def test_syndrome_length_invariant(self, syndrome_list):
        checks = [[0, 1], [1, 2], [2, 3]]
        n_qubits = 4
        dec = qd.UnionFindDecoder(checks, n_qubits)
        syndrome = np.array(syndrome_list, dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert len(corr) == n_qubits
        assert corr.dtype == np.uint8

    def test_large_ring_code(self):
        checks, n_qubits = qd.generate_ring_code_checks(20)
        dec = qd.UnionFindDecoder(checks, n_qubits)
        syndrome = np.random.randint(0, 2, size=(len(checks),), dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert corr.shape == (n_qubits,)
        assert corr.dtype == np.uint8

    def test_high_weight_check(self):
        """Syndrome-faithfulness on a high-weight (>2 qubit) check.

        A weight-4 check with syndrome [1] must receive an ODD-parity correction
        so that H @ correction == syndrome. The previous assertion (`corr == 1`)
        encoded an over-flip bug — flipping all four qubits gives even parity (0),
        which does NOT reproduce syndrome [1].
        """
        checks = [[0, 1, 2, 3]]
        n_qubits = 4
        dec = qd.UnionFindDecoder(checks, n_qubits)
        syndrome = np.array([1], dtype=np.uint8)
        corr = dec.decode(syndrome)
        parity = int(np.bitwise_xor.reduce(corr)) if corr.size else 0
        assert parity == 1, f"correction must reproduce syndrome [1], got {corr}"

    def test_high_weight_check_partial(self):
        """Two high-weight checks, one active: correction must reproduce [1, 0]."""
        checks = [[0, 1, 2, 3], [4, 5, 6, 7]]
        n_qubits = 8
        dec = qd.UnionFindDecoder(checks, n_qubits)
        syndrome = np.array([1, 0], dtype=np.uint8)
        corr = dec.decode(syndrome)
        # check 0 (qubits 0-3) must have odd parity, check 1 (qubits 4-7) even.
        assert int(np.bitwise_xor.reduce(corr[:4])) == 1, f"check 0 not reproduced: {corr}"
        assert int(np.bitwise_xor.reduce(corr[4:])) == 0, f"check 1 not reproduced: {corr}"

    def test_generate_surface_code_checks(self):
        checks, n_qubits = qd.generate_surface_code_checks(5)
        assert n_qubits == 25
        # For distance 5 toric code: 25 X-plaquette + 25 Z-star checks = 50 checks
        assert len(checks) == 50
        # Each check should have 4 qubits (4-body stabilizer)
        for check in checks:
            assert len(check) == 4

    def test_generate_ring_code_checks(self):
        checks, n_qubits = qd.generate_ring_code_checks(5)
        assert n_qubits == 25
        assert len(checks) == 25
        assert checks[0] == [0, 1]
        assert checks[24] == [24, 0]

    def test_generate_repetition_code_checks(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        assert n_qubits == 5
        assert len(checks) == 4
        assert checks[0] == [0, 1]
        assert checks[3] == [3, 4]

    def test_zero_checks(self):
        checks = []
        with pytest.raises(Exception):
            qd.UnionFindDecoder(checks, 4)


class TestStreamingDecoder:
    """Tests for the streaming multi-round decoder."""

    def test_multi_round(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.StreamingDecoder(checks, n_qubits, history_size=5)
        r1 = np.array([1, 0, 0], dtype=np.uint8)
        r2 = np.array([0, 1, 0], dtype=np.uint8)
        dec.update(r1)
        c2 = dec.update(r2)
        # Cumulative syndrome is [1,1,0] -> cluster spans qubits 0,1,2
        assert c2.shape == (n_qubits,)
        assert c2[1] == 1 or c2[2] == 1

    def test_flush(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.StreamingDecoder(checks, n_qubits, history_size=5)
        dec.update(np.array([1, 0, 0], dtype=np.uint8))
        dec.flush()
        c = dec.update(np.zeros(len(checks), dtype=np.uint8))
        assert np.all(c == 0)

    def test_decode_without_history(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.StreamingDecoder(checks, n_qubits, history_size=5)
        syndrome = np.array([1, 0, 0], dtype=np.uint8)
        corr = dec.decode(syndrome)
        assert corr.shape == (n_qubits,)

    def test_n_qubits_property(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.StreamingDecoder(checks, n_qubits, history_size=5)
        assert dec.n_qubits == n_qubits
        assert dec.n_checks == len(checks)


class TestBatchDecoder:
    """Tests for the parallel batch decoder."""

    def test_parallel_batch_decode_shape(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.BatchDecoder(checks, n_qubits)
        syndromes = np.random.randint(0, 2, size=(100, len(checks)), dtype=np.uint8)
        corr = dec.parallel_batch_decode(syndromes)
        assert corr.shape == (100, n_qubits)

    def test_parallel_batch_consistency(self, simple_checks):
        checks, n_qubits = simple_checks
        batch_dec = qd.BatchDecoder(checks, n_qubits)
        uf_dec = qd.UnionFindDecoder(checks, n_qubits)
        syndromes = np.random.randint(0, 2, size=(20, len(checks)), dtype=np.uint8)
        batch_corr = batch_dec.parallel_batch_decode(syndromes)
        for i in range(20):
            single_corr = uf_dec.decode(syndromes[i])
            assert np.array_equal(batch_corr[i], single_corr)

    def test_large_batch(self):
        checks, n_qubits = qd.generate_surface_code_checks(10)
        dec = qd.BatchDecoder(checks, n_qubits)
        syndromes = np.random.randint(0, 2, size=(1000, len(checks)), dtype=np.uint8)
        corr = dec.parallel_batch_decode(syndromes)
        assert corr.shape == (1000, n_qubits)

    def test_n_qubits_property(self, simple_checks):
        checks, n_qubits = simple_checks
        dec = qd.BatchDecoder(checks, n_qubits)
        assert dec.n_qubits == n_qubits
        assert dec.n_checks == len(checks)


class TestCUDABatchDecoder:
    """Tests for the native CUDA batch decoder."""

    def test_cuda_available_and_device_metadata(self):
        if not qd.CUDABatchDecoder.is_available():
            pytest.skip("CUDA device or CUDA runtime compiler not available")
        checks, n_qubits = qd.generate_surface_code_checks(5)
        dec = qd.CUDABatchDecoder(checks, n_qubits)
        assert "NVIDIA" in dec.device_name
        major, minor = dec.compute_capability
        assert major >= 5
        assert minor >= 0

    def test_cuda_matches_cpu(self):
        if not qd.CUDABatchDecoder.is_available():
            pytest.skip("CUDA device or CUDA runtime compiler not available")
        checks, n_qubits = qd.generate_surface_code_checks(5)
        cpu = qd.CPUBatchDecoder(checks, n_qubits)
        cuda = qd.CUDABatchDecoder(checks, n_qubits)
        rng = np.random.default_rng(2026)
        syndromes = rng.integers(0, 2, size=(4096, len(checks)), dtype=np.uint8)
        assert np.array_equal(cuda.batch_decode(syndromes), cpu.batch_decode(syndromes))
        assert cuda.total_failures == 0
        assert not cuda.is_degraded

    def test_cuda_rejects_wrong_shape(self):
        if not qd.CUDABatchDecoder.is_available():
            pytest.skip("CUDA device or CUDA runtime compiler not available")
        checks, n_qubits = qd.generate_surface_code_checks(3)
        dec = qd.CUDABatchDecoder(checks, n_qubits)
        with pytest.raises(ValueError):
            dec.batch_decode(np.zeros((8, len(checks) + 1), dtype=np.uint8))


class TestUtils:
    """Tests for utility functions."""

    def test_check_to_edges(self):
        checks = [[0, 1, 2], [2, 3]]
        edges = qd.check_to_edges(checks)
        assert len(edges) == 3
        assert edges[0] == (0, 1)
        assert edges[1] == (1, 2)
        assert edges[2] == (2, 3)

    def test_generate_surface_code_checks(self):
        checks, n_qubits = qd.generate_surface_code_checks(5)
        assert n_qubits == 25
        assert len(checks) == 50
        for check in checks:
            assert len(check) == 4

    def test_generate_ring_code_checks(self):
        checks, n_qubits = qd.generate_ring_code_checks(5)
        assert n_qubits == 25
        assert len(checks) == 25
        assert checks[0] == [0, 1]
        assert checks[24] == [24, 0]

    def test_generate_repetition_code_checks(self):
        checks, n_qubits = qd.generate_repetition_code_checks(5)
        assert n_qubits == 5
        assert len(checks) == 4
        assert checks[0] == [0, 1]
        assert checks[3] == [3, 4]

    def test_check_to_edges_empty(self):
        edges = qd.check_to_edges([])
        assert len(edges) == 0
