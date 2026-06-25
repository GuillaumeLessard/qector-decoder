"""Stabiliser-equivalent corrections are not logical failures.

Section 6 (logical-observable proof). When a decoder returns a correction ``c`` that
differs from the planted error ``e`` by a *stabiliser* (an element of ``ker(H)`` that
flips no logical), there is no logical failure -- yet the naive ``c != e`` metric would
wrongly flag one. These tests decode real syndromes with ``BlossomDecoder`` and assert:

* For every decoded shot the residual ``r = c ^ e`` lies in ``ker(H)`` (same syndrome).
* The correct metric ``(L @ r) & 1`` never flags a harmless stabiliser shift.
* There exists at least one concrete shot with ``c != e`` but ``(L @ r) & 1 == 0`` --
  a case where the naive ``c != e`` metric over-counts.
"""

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes


def _gf2_nullspace_basis(H):
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


def _decoder(code):
    return qd.BlossomDecoder(code.check_to_qubits, code.n_qubits)


# ``expect_overcount`` is True for codes with a non-trivial stabiliser group, where
# Blossom can return a correction that differs from the error by a harmless stabiliser.
# The repetition code has no stabiliser-only ker(H) element (its only non-trivial ker
# vector is the logical), so ``c != e`` there always coincides with a logical flip.
@pytest.mark.parametrize(
    "code_factory,expect_overcount",
    [
        (lambda: codes.repetition_code(7), False),
        (lambda: codes.rotated_surface_code(5), True),
    ],
)
def test_residual_always_in_ker_and_metric_consistent(code_factory, expect_overcount):
    code = code_factory()
    H = code.parity_check_matrix().astype(np.uint8)
    L = code.logicals_matrix().astype(np.uint8)
    dec = _decoder(code)
    rng = np.random.default_rng(0)

    naive_failures = 0
    observable_failures = 0
    overcount_examples = 0
    for _ in range(400):
        e = np.asarray(code.random_error(0.08, rng), np.uint8)
        s = np.asarray(code.syndrome(e), np.uint8) & 1
        c = np.asarray(dec.decode(s), np.uint8)
        # Faithful: correction reproduces the syndrome.
        assert np.array_equal((H @ c) & 1, s)
        r = c ^ e
        # Residual is therefore always in ker(H).
        assert np.array_equal((H @ r) & 1, np.zeros(H.shape[0], np.uint8))

        naive = not np.array_equal(c, e)
        logical = bool(((L @ r) & 1).any())
        naive_failures += int(naive)
        observable_failures += int(logical)
        if naive and not logical:
            overcount_examples += 1

    # The observable metric never exceeds the naive one; the gap is exactly the harmless
    # stabiliser shifts the naive metric overcounts.
    assert observable_failures <= naive_failures
    assert overcount_examples == naive_failures - observable_failures
    if expect_overcount:
        assert overcount_examples > 0
    else:
        # No stabiliser group -> the two metrics agree exactly.
        assert overcount_examples == 0
        assert observable_failures == naive_failures


def test_concrete_stabilizer_shift_is_harmless():
    """A planted *stabiliser* error decodes to a different vector with no logical flip."""
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix().astype(np.uint8)
    L = code.logicals_matrix().astype(np.uint8)
    basis = _gf2_nullspace_basis(H)
    Bm = np.array(basis, np.uint8)

    # Find a non-zero, logical-preserving ker element (a stabiliser).
    rng = np.random.default_rng(3)
    st = None
    for _ in range(1000):
        coef = rng.integers(0, 2, Bm.shape[0]).astype(np.uint8)
        v = (coef @ Bm) & 1
        if v.any() and not ((L @ v) & 1).any():
            st = v
            break
    assert st is not None and st.any()

    # Plant e = st: syndrome is zero, so the decoder returns the all-zero correction.
    dec = _decoder(code)
    e = st
    s = np.asarray(code.syndrome(e), np.uint8) & 1
    assert not s.any()
    c = np.asarray(dec.decode(s), np.uint8)
    r = c ^ e

    # Naive metric flags a failure, but it is harmless: residual is a pure stabiliser.
    assert not np.array_equal(c, e)  # naive c != e  -> would (wrongly) flag
    assert np.array_equal(
        (H @ r) & 1, np.zeros(H.shape[0], np.uint8)
    )  # residual in ker(H)
    assert not ((L @ r) & 1).any()  # observable metric -> no logical failure


def test_all_zero_syndrome_is_no_failure():
    for code in (codes.repetition_code(5), codes.rotated_surface_code(3)):
        H = code.parity_check_matrix().astype(np.uint8)
        L = code.logicals_matrix().astype(np.uint8)
        dec = _decoder(code)
        s = np.zeros(code.n_checks, np.uint8)
        c = np.asarray(dec.decode(s), np.uint8)
        e = np.zeros(code.n_qubits, np.uint8)
        r = c ^ e
        assert np.array_equal((H @ c) & 1, s)
        # No syndrome -> no logical flip regardless of any stabiliser the decoder emits.
        assert not ((L @ r) & 1).any()
