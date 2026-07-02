"""Tests Python pour le LookupTableDecoder.

Verifie que :
1. La table est correctement peuplee pour les petits codes.
2. Le lookup retourne la meme correction que UnionFind pour les syndromes precalcules.
3. Le fallback fonctionne pour les syndromes non precalcules sur les grands codes.
4. La taille de la table est coherente avec max_entries.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd


class TestLookupTableDecoder:
    def test_ring_code_exhaustive(self):
        """Code anneau d=4 (4 qubits, 4 checks) - enumeration exhaustive."""
        check_to_qubits = [[0, 1], [1, 2], [2, 3], [3, 0]]
        n_qubits = 4

        lt = qd.LookupTableDecoder(check_to_qubits, n_qubits)
        lt.build_table(max_entries=1 << n_qubits)  # 16 entrees

        assert lt.table_size > 0
        assert lt.n_qubits == n_qubits
        assert lt.n_checks == 4

        uf = qd.UnionFindDecoder(check_to_qubits, n_qubits)

        # Teste tous les syndromes possibles (2^4 = 16)
        for syndrome_bits in range(1 << 4):
            syndrome = np.array([(syndrome_bits >> i) & 1 for i in range(4)], dtype=np.uint8)
            lt_corr = lt.decode(syndrome)
            uf_corr = uf.decode(syndrome)
            assert np.array_equal(lt_corr, uf_corr), (
                f"Syndrome {syndrome}: LT={lt_corr.tolist()} != UF={uf_corr.tolist()}"
            )

    def test_surface_code_d3_partial(self):
        """Code de surface d=3 - table partielle (low-weight)."""
        check_to_qubits, n_qubits = qd.generate_surface_code_checks(3)
        n_checks = len(check_to_qubits)

        lt = qd.LookupTableDecoder(check_to_qubits, n_qubits)
        lt.build_table(max_entries=5000)

        assert lt.table_size > 0
        assert lt.n_qubits == n_qubits
        assert lt.n_checks == n_checks

        uf = qd.UnionFindDecoder(check_to_qubits, n_qubits)

        # Teste que les syndromes des erreurs de poids faible sont dans la table
        rng = np.random.default_rng(42)
        for _ in range(200):
            error = (rng.random(n_qubits) < 0.05).astype(np.uint8)
            syndrome = np.zeros(n_checks, dtype=np.uint8)
            for ci, qubits in enumerate(check_to_qubits):
                syndrome[ci] = int(error[qubits].sum()) % 2

            lt_corr = lt.decode(syndrome)
            uf_corr = uf.decode(syndrome)
            assert np.array_equal(lt_corr, uf_corr), f"Low-weight error: LT={lt_corr.tolist()} != UF={uf_corr.tolist()}"

    def test_surface_code_d5_fallback(self):
        """Code de surface d=5 - table partielle, fallback pour les syndromes non precalcules."""
        check_to_qubits, n_qubits = qd.generate_surface_code_checks(5)
        n_checks = len(check_to_qubits)

        lt = qd.LookupTableDecoder(check_to_qubits, n_qubits)
        lt.build_table(max_entries=10)

        assert lt.table_size <= 100
        assert lt.n_qubits == n_qubits
        assert lt.n_checks == n_checks

        uf = qd.UnionFindDecoder(check_to_qubits, n_qubits)

        rng = np.random.default_rng(123)
        for _ in range(100):
            error = (rng.random(n_qubits) < 0.1).astype(np.uint8)
            syndrome = np.zeros(n_checks, dtype=np.uint8)
            for ci, qubits in enumerate(check_to_qubits):
                syndrome[ci] = int(error[qubits].sum()) % 2

            lt_corr = lt.decode(syndrome)
            uf_corr = uf.decode(syndrome)
            assert np.array_equal(lt_corr, uf_corr), (
                f"Fallback syndrome: LT={lt_corr.tolist()} != UF={uf_corr.tolist()}"
            )

    def test_batch_decode_consistency(self):
        """Verifie que batch_decode est coherent avec decode individuel."""
        check_to_qubits = [[0, 1], [1, 2], [2, 3], [3, 0]]
        n_qubits = 4

        lt = qd.LookupTableDecoder(check_to_qubits, n_qubits)
        lt.build_table(max_entries=10)

        rng = np.random.default_rng(99)
        syndromes = rng.integers(0, 2, size=(20, 4), dtype=np.uint8)
        batch_corr = lt.batch_decode(syndromes)

        assert batch_corr.shape == (20, n_qubits)

        for i in range(20):
            single_corr = lt.decode(syndromes[i])
            assert np.array_equal(batch_corr[i], single_corr)

    def test_table_size_respect_max_entries(self):
        """La taille de la table ne doit pas depasser max_entries."""
        check_to_qubits, n_qubits = qd.generate_surface_code_checks(3)

        lt = qd.LookupTableDecoder(check_to_qubits, n_qubits)
        lt.build_table(max_entries=10)
        assert lt.table_size <= 10

        lt2 = qd.LookupTableDecoder(check_to_qubits, n_qubits)
        lt2.build_table(max_entries=500)
        assert lt2.table_size <= 500
        assert lt2.table_size > 0

        # Pour d=3, n_qubits=9, enumeration exhaustive produit un nombre de syndromes uniques
        # (inferieur ou egal a 2^n_qubits car plusieurs erreurs peuvent avoir le meme syndrome)
        lt3 = qd.LookupTableDecoder(check_to_qubits, n_qubits)
        lt3.build_table(max_entries=10000)
        assert lt3.table_size <= 512  # au plus 512 erreurs possibles
        assert lt3.table_size > 0
