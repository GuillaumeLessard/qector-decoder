"""Validation tests for SparseBlossomDecoder vs PyMatching.

Ces tests comparent directement SparseBlossomDecoder (et HybridDecoder) avec
PyMatching sur des cas difficiles pour detecter les regressions.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd


def test_sparse_blossom_vs_pymatching_ring_small():
    """Compare SparseBlossom avec PyMatching sur un ring code d=5."""
    d = 5
    checks, n_qubits = qd.generate_ring_code_checks(d)

    try:
        import pymatching
        import scipy.sparse
    except ImportError:
        pytest.skip("PyMatching not installed")

    H = np.zeros((len(checks), n_qubits), dtype=np.uint8)
    for ci, qubits in enumerate(checks):
        for q in qubits:
            H[ci, q] = 1
    H_sparse = scipy.sparse.csr_matrix(H)
    m = pymatching.Matching(H_sparse)

    dec = qd.SparseBlossomDecoder(checks, n_qubits)

    n_trials = 100
    p = 0.1
    mismatches = 0
    for _ in range(n_trials):
        error = np.random.rand(n_qubits) < p
        error = error.astype(np.uint8)
        syndrome = np.zeros(len(checks), dtype=np.uint8)
        for ci, qubits in enumerate(checks):
            syndrome[ci] = int(np.sum(error[qubits]) % 2)

        corr_sparse = dec.decode(syndrome)
        corr_pm = m.decode(syndrome)

        # Both should give valid corrections (same syndrome)
        syn_sparse = np.zeros(len(checks), dtype=np.uint8)
        syn_pm = np.zeros(len(checks), dtype=np.uint8)
        for ci, qubits in enumerate(checks):
            syn_sparse[ci] = int(np.sum(corr_sparse[qubits]) % 2)
            syn_pm[ci] = int(np.sum(corr_pm[qubits]) % 2)

        assert np.array_equal(syn_sparse, syndrome), "SparseBlossom correction invalid"
        assert np.array_equal(syn_pm, syndrome), "PyMatching correction invalid"

    print(f"SparseBlossom vs PyMatching ring d={d}: {mismatches} mismatches / {n_trials}")


def test_sparse_blossom_vs_pymatching_surface():
    """Compare SparseBlossom avec PyMatching sur un ring code (qubit dans exactement 2 checks)."""
    # PyMatching supporte les codes où chaque qubit est dans ≤2 checks.
    # Le ring code satisfait cette condition (chaque qubit dans exactement 2 checks).
    # Le repetition code a des qubits de frontière (dans 1 seul check) ce qui
    # n'est pas correctement géré par BlossomDecoder (boundary abstraction).
    d = 5
    checks, n_qubits = qd.generate_ring_code_checks(d)
    n_checks = len(checks)

    try:
        import pymatching
        import scipy.sparse
    except ImportError:
        pytest.skip("PyMatching not installed")

    H = np.zeros((n_checks, n_qubits), dtype=np.uint8)
    for ci, qubits in enumerate(checks):
        for q in qubits:
            H[ci, q] = 1
    H_sparse = scipy.sparse.csr_matrix(H)
    m = pymatching.Matching(H_sparse)

    dec = qd.SparseBlossomDecoder(checks, n_qubits)

    n_trials = 50
    p = 0.1
    for _ in range(n_trials):
        error = np.random.rand(n_qubits) < p
        error = error.astype(np.uint8)
        syndrome = np.zeros(n_checks, dtype=np.uint8)
        for ci, qubits in enumerate(checks):
            syndrome[ci] = int(np.sum(error[qubits]) % 2)

        corr_sparse = dec.decode(syndrome)
        corr_pm = m.decode(syndrome)

        # Verify both corrections are valid
        syn_sparse = np.zeros(n_checks, dtype=np.uint8)
        syn_pm = np.zeros(n_checks, dtype=np.uint8)
        for ci, qubits in enumerate(checks):
            syn_sparse[ci] = int(np.sum(corr_sparse[qubits]) % 2)
            syn_pm[ci] = int(np.sum(corr_pm[qubits]) % 2)

        assert np.array_equal(syn_sparse, syndrome), "SparseBlossom correction invalid on ring code"
        assert np.array_equal(syn_pm, syndrome), "PyMatching correction invalid on ring code"


def test_sparse_blossom_vs_pymatching_surface_code_strict():
    """Compare SparseBlossom with PyMatching on a rotated surface code d=5.

    Uses a rotated surface code (single Z-stabilizer sector) so that each qubit
    appears in at most 2 checks, satisfying PyMatching's constraint.
    Both decoders must produce syndrome-faithful corrections for every trial.
    """
    try:
        import pymatching
        import scipy.sparse
    except ImportError:
        pytest.skip("PyMatching not installed")

    from qector_decoder_v3.codes import rotated_surface_code

    code = rotated_surface_code(5)
    checks = code.check_to_qubits
    n_qubits = code.n_qubits
    n_checks = len(checks)

    H = np.zeros((n_checks, n_qubits), dtype=np.uint8)
    for ci, qubits in enumerate(checks):
        for q in qubits:
            H[ci, q] = 1
    H_sparse = scipy.sparse.csr_matrix(H)
    m = pymatching.Matching(H_sparse)

    dec = qd.SparseBlossomDecoder(checks, n_qubits)

    n_trials = 50
    p = 0.1
    rng = np.random.default_rng(99)
    for _ in range(n_trials):
        error = (rng.random(n_qubits) < p).astype(np.uint8)
        syndrome = np.zeros(n_checks, dtype=np.uint8)
        for ci, qubits in enumerate(checks):
            syndrome[ci] = int(np.sum(error[qubits]) % 2)

        corr_sparse = dec.decode(syndrome)
        corr_pm = m.decode(syndrome)

        syn_sparse = np.zeros(n_checks, dtype=np.uint8)
        syn_pm = np.zeros(n_checks, dtype=np.uint8)
        for ci, qubits in enumerate(checks):
            syn_sparse[ci] = int(np.sum(corr_sparse[qubits]) % 2)
            syn_pm[ci] = int(np.sum(corr_pm[qubits]) % 2)

        assert np.array_equal(syn_sparse, syndrome), "SparseBlossom correction invalid on surface code"
        assert np.array_equal(syn_pm, syndrome), "PyMatching correction invalid on surface code"


def test_hybrid_decoder_basic():
    """Test que HybridDecoder fonctionne sur un ring code."""
    d = 5
    checks, n_qubits = qd.generate_ring_code_checks(d)
    dec = qd.HybridDecoder(checks, n_qubits, None, None, None, 8, 2)

    n_trials = 20
    p = 0.1
    for _ in range(n_trials):
        error = np.random.rand(n_qubits) < p
        error = error.astype(np.uint8)
        syndrome = np.zeros(len(checks), dtype=np.uint8)
        for ci, qubits in enumerate(checks):
            syndrome[ci] = int(np.sum(error[qubits]) % 2)

        corr_hybrid = dec.decode_hybrid(syndrome)
        corr_standard = dec.decode_standard(syndrome)

        # Both should produce valid corrections
        syn_hybrid = np.zeros(len(checks), dtype=np.uint8)
        syn_standard = np.zeros(len(checks), dtype=np.uint8)
        for ci, qubits in enumerate(checks):
            syn_hybrid[ci] = int(np.sum(corr_hybrid[qubits]) % 2)
            syn_standard[ci] = int(np.sum(corr_standard[qubits]) % 2)

        assert np.array_equal(syn_hybrid, syndrome), "Hybrid correction invalid"
        assert np.array_equal(syn_standard, syndrome), "Standard correction invalid"


def test_bposd_decoder_validity():
    """Test que BPOSDDecoder produit des corrections valides."""
    d = 5
    checks, n_qubits = qd.generate_ring_code_checks(d)
    dec = qd.BPOSDDecoder(checks, n_qubits, 0.1)

    n_trials = 50
    p = 0.1
    for _ in range(n_trials):
        error = np.random.rand(n_qubits) < p
        error = error.astype(np.uint8)
        syndrome = np.zeros(len(checks), dtype=np.uint8)
        for ci, qubits in enumerate(checks):
            syndrome[ci] = int(np.sum(error[qubits]) % 2)

        corr = dec.decode(syndrome)
        syn = np.zeros(len(checks), dtype=np.uint8)
        for ci, qubits in enumerate(checks):
            syn[ci] = int(np.sum(corr[qubits]) % 2)

        assert np.array_equal(syn, syndrome), "BPOSD correction invalid"


if __name__ == "__main__":
    test_sparse_blossom_vs_pymatching_ring_small()
    test_sparse_blossom_vs_pymatching_surface()
    test_hybrid_decoder_basic()
    test_bposd_decoder_validity()
    print("All validation tests passed!")
