"""Logical equivalence is by coset, not by raw equality.

Section 6 (logical-observable proof). Two corrections with the same syndrome differ by
an element ``st`` of ``ker(H)``. Such an ``st`` is *harmless* (a stabiliser-like move)
iff it flips no logical, i.e. ``(L @ st) & 1 == 0``; otherwise it flips a logical.

We sample ``st`` from a GF(2) basis of ``ker(H)`` and assert:

* if ``(L @ st) & 1 == 0``  then  ``(L @ (c ^ st)) & 1 == (L @ c) & 1``   (coset preserved)
* if ``(L @ st) & 1 == 1``  then  the predicted observable flips:
  ``(L @ (c ^ st)) & 1 != (L @ c) & 1``.

This directly proves that the logical outcome is a function of the coset
``c + ker(H)/stabilisers``, never of the literal correction vector. The repetition code's
only non-trivial ``ker(H)`` element is the logical itself, so it exhibits only the
logical-flipping class; the rotated surface code exhibits both classes.
"""

import numpy as np
import pytest

from qector_decoder_v3 import codes


def _gf2_nullspace_basis(H):
    """Return a GF(2) basis of ``ker(H)`` (vectors ``v`` with ``H @ v == 0`` mod 2)."""
    H = (np.asarray(H) % 2).astype(np.uint8).copy()
    m, n = H.shape
    pivots = []
    row = 0
    for col in range(n):
        piv = None
        for r in range(row, m):
            if H[r, col]:
                piv = r
                break
        if piv is None:
            continue
        H[[row, piv]] = H[[piv, row]]
        for r in range(m):
            if r != row and H[r, col]:
                H[r] ^= H[row]
        pivots.append(col)
        row += 1
        if row == m:
            break
    pivot_set = set(pivots)
    free = [c for c in range(n) if c not in pivot_set]
    basis = []
    for f in free:
        v = np.zeros(n, np.uint8)
        v[f] = 1
        for i, pc in enumerate(pivots):
            if H[i, f]:
                v[pc] = 1
        basis.append(v)
    return basis


def _logical_flips(L, v):
    return (L @ ((np.asarray(v) % 2).astype(np.uint8))) & 1


def _coset_check(code, n_samples=600, seed=0):
    H = code.parity_check_matrix().astype(np.uint8)
    L = code.logicals_matrix().astype(np.uint8)
    assert L is not None and L.shape[1] == code.n_qubits

    basis = _gf2_nullspace_basis(H)
    # Every basis vector must be in ker(H).
    for b in basis:
        assert np.array_equal((H @ b) & 1, np.zeros(H.shape[0], np.uint8))

    rng = np.random.default_rng(seed)
    Bm = np.array(basis, np.uint8) if basis else np.zeros((0, code.n_qubits), np.uint8)

    n_stab_class = 0
    n_logical_class = 0
    for _ in range(n_samples):
        # A random correction (arbitrary qubit set) and a random ker element.
        c = (rng.random(code.n_qubits) < 0.5).astype(np.uint8)
        if Bm.shape[0]:
            coef = rng.integers(0, 2, Bm.shape[0]).astype(np.uint8)
            st = (coef @ Bm) & 1
        else:
            st = np.zeros(code.n_qubits, np.uint8)
        if not st.any():
            continue

        # Same syndrome by construction.
        assert np.array_equal((H @ (c ^ st)) & 1, (H @ c) & 1)

        before = _logical_flips(L, c)
        after = _logical_flips(L, c ^ st)
        if not _logical_flips(L, st).any():
            # Harmless stabiliser-like move: coset (and hence logical outcome) preserved.
            assert np.array_equal(after, before)
            n_stab_class += 1
        else:
            # Logical move: predicted observable must change.
            assert not np.array_equal(after, before)
            n_logical_class += 1

    return n_stab_class, n_logical_class


def test_rotated_surface_code_coset_equivalence():
    code = codes.rotated_surface_code(5)
    n_stab, n_log = _coset_check(code, n_samples=600, seed=1)
    # The rotated surface code has both stabiliser-like and logical ker elements.
    assert n_stab > 0
    assert n_log > 0


@pytest.mark.parametrize("d", [3, 5, 7])
def test_repetition_code_coset_equivalence(d):
    code = codes.repetition_code(d)
    n_stab, n_log = _coset_check(code, n_samples=400, seed=2)
    # The repetition code's only non-trivial ker(H) element is the logical itself,
    # so every non-zero ker sample flips a logical; there is no stabiliser class.
    assert n_log > 0
    assert n_stab == 0
