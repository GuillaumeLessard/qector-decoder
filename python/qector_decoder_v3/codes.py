"""
qector_decoder_v3.codes — Code-family helpers.

Construct the parity-check structure (``check_to_qubits``) for the common QEC
code families used to benchmark and validate decoders, in the exact format the
QECTOR decoders consume:

    check_to_qubits : list[list[int]]   # one entry per check (row of H)
    n_qubits        : int               # number of data qubits (columns of H)

All *surface-style* generators here return proper **matching graphs**: every
qubit appears in at most two checks, so the Union-Find / Blossom / Sparse-Blossom
/ MWPM decoders apply directly.  The custom-matrix and hypergraph-product helpers
build arbitrary CSS parity checks for the LDPC / BP-OSD path.

Every generator is validated empirically by the syndrome-faithfulness test suite
(``H @ decode(s) == s (mod 2)``) — see ``python/tests/test_codes.py``.

Examples
--------
>>> from qector_decoder_v3 import codes, BlossomDecoder
>>> code = codes.rotated_surface_code(5)
>>> code.n_qubits, code.n_checks
(25, 8)
>>> dec = BlossomDecoder(code.check_to_qubits, code.n_qubits)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Code",
    "repetition_code",
    "ring_code",
    "rotated_surface_code",
    "unrotated_surface_code",
    "toric_code",
    "heavy_hex_code",
    "from_parity_check_matrix",
    "hypergraph_product",
    "bivariate_bicycle_code",
    "bicycle_code",
    "list_codes",
]


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class Code:
    """A decoding problem: a parity-check structure plus optional metadata.

    Attributes
    ----------
    name : str
        Human-readable identifier, e.g. ``"rotated_surface_d5"``.
    check_to_qubits : list[list[int]]
        For each check (row of H), the sorted list of qubit indices it touches.
    n_qubits : int
        Number of data qubits (columns of H).
    distance : int | None
        Code distance ``d`` when well-defined.
    logicals : list[list[int]] | None
        Logical-observable representatives (sets of qubit indices).  ``None``
        when not provided; supply your own (or load from a Stim DEM) for
        logical-error-rate evaluation.
    qubit_weights : numpy.ndarray | None
        Optional per-qubit weights (e.g. ``log((1-p)/p)``) for weighted matching.
    description : str
        Short description of the construction.
    """

    name: str
    check_to_qubits: List[List[int]]
    n_qubits: int
    distance: Optional[int] = None
    logicals: Optional[List[List[int]]] = None
    qubit_weights: Optional[np.ndarray] = None
    description: str = ""
    _meta: dict = field(default_factory=dict, repr=False)

    # -- derived -----------------------------------------------------------
    @property
    def n_checks(self) -> int:
        return len(self.check_to_qubits)

    def parity_check_matrix(self) -> np.ndarray:
        """Dense GF(2) parity-check matrix ``H`` of shape ``(n_checks, n_qubits)``."""
        H = np.zeros((self.n_checks, self.n_qubits), dtype=np.uint8)
        for ci, qs in enumerate(self.check_to_qubits):
            for q in qs:
                H[ci, q] ^= np.uint8(1)
        return H

    # convenient alias
    def H(self) -> np.ndarray:  # noqa: N802 - matches QEC convention
        return self.parity_check_matrix()

    def logicals_matrix(self) -> Optional[np.ndarray]:
        """Logical observables as a ``(n_logicals, n_qubits)`` uint8 matrix, or ``None``."""
        if self.logicals is None:
            return None
        L = np.zeros((len(self.logicals), self.n_qubits), dtype=np.uint8)
        for i, qs in enumerate(self.logicals):
            for q in qs:
                L[i, q] ^= np.uint8(1)
        return L

    def is_matching_graph(self) -> bool:
        """True iff every qubit appears in at most two checks (graphlike code)."""
        deg = np.zeros(self.n_qubits, dtype=np.int64)
        for qs in self.check_to_qubits:
            for q in qs:
                deg[q] += 1
        return bool((deg <= 2).all())

    def max_qubit_degree(self) -> int:
        deg = np.zeros(self.n_qubits, dtype=np.int64)
        for qs in self.check_to_qubits:
            for q in qs:
                deg[q] += 1
        return int(deg.max()) if self.n_qubits else 0

    def random_error(
        self, p: float, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Sample an i.i.d. bit-flip error vector of shape ``(n_qubits,)``."""
        rng = rng or np.random.default_rng()
        return (rng.random(self.n_qubits) < p).astype(np.uint8)

    def syndrome(self, error: Sequence[int]) -> np.ndarray:
        """Compute ``H @ error (mod 2)`` for an error vector."""
        e = np.asarray(error, dtype=np.uint8)
        return (self.parity_check_matrix() @ e) & 1

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        d = f", d={self.distance}" if self.distance is not None else ""
        return (
            f"Code({self.name!r}, n_qubits={self.n_qubits}, "
            f"n_checks={self.n_checks}{d}, matching={self.is_matching_graph()})"
        )


def _sorted_checks(checks: List[List[int]]) -> List[List[int]]:
    return [sorted(set(int(q) for q in c)) for c in checks]


# ---------------------------------------------------------------------------
# 1D codes
# ---------------------------------------------------------------------------
def repetition_code(distance: int) -> Code:
    """Open 1D repetition code: ``d`` qubits in a line, ``d-1`` weight-2 checks.

    The two end qubits are boundary edges (degree 1).  Logical observable is a
    single boundary-crossing edge ``{0}`` — verified valid because every residual
    error after matching lies in ``ker(H) = {0, all-ones}`` and ``{0}``
    distinguishes them.
    """
    if distance < 2:
        raise ValueError("repetition distance must be >= 2")
    d = int(distance)
    checks = [[i, i + 1] for i in range(d - 1)]
    return Code(
        name=f"repetition_d{d}",
        check_to_qubits=checks,
        n_qubits=d,
        distance=d,
        logicals=[[0]],
        description="Open 1D repetition code (matching graph).",
    )


def ring_code(n: int) -> Code:
    """Periodic 1D ring code: ``n`` qubits on a cycle, ``n`` weight-2 checks.

    This is the 1D toric code; checks are linearly dependent (rank ``n-1``) so it
    encodes one logical bit (the full cycle).
    """
    if n < 3:
        raise ValueError("ring size must be >= 3")
    n = int(n)
    checks = [[i, (i + 1) % n] for i in range(n)]
    return Code(
        name=f"ring_n{n}",
        check_to_qubits=checks,
        n_qubits=n,
        distance=n,
        logicals=[[0]],
        description="Periodic 1D ring / 1D-toric code (matching graph).",
    )


# ---------------------------------------------------------------------------
# Surface codes
# ---------------------------------------------------------------------------
def rotated_surface_code(distance: int) -> Code:
    """Rotated surface code, single (Z) sector — a matching graph.

    ``d*d`` data qubits on a square grid; weight-4 plaquette checks on the
    sublattice ``(r+c) even`` plus weight-2 boundary checks so that each interior
    qubit is shared by two checks.  The logical observable is the top row of data
    qubits (a horizontal string crossing the lattice).
    """
    if distance < 2:
        raise ValueError("surface distance must be >= 2")
    d = int(distance)

    def q(r: int, c: int) -> int:
        return r * d + c

    checks: List[List[int]] = []
    # Bulk weight-4 plaquettes on the even sublattice.
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 0:
                checks.append([q(r, c), q(r, c + 1), q(r + 1, c), q(r + 1, c + 1)])
    # Weight-2 boundary checks on the top and bottom rows to bind the dangling
    # qubits, keeping the construction a valid matching graph (degree <= 2).
    for c in range(d - 1):
        if c % 2 == 1:  # top boundary, complementary parity to the bulk
            checks.append([q(0, c), q(0, c + 1)])
        if (d - 2 + c) % 2 == 1:  # bottom boundary
            checks.append([q(d - 1, c), q(d - 1, c + 1)])

    logical = [q(0, c) for c in range(d)]  # top-row horizontal string
    code = Code(
        name=f"rotated_surface_d{d}",
        check_to_qubits=_sorted_checks(checks),
        n_qubits=d * d,
        distance=d,
        logicals=[logical],
        description="Rotated surface code, single sector (matching graph).",
    )
    return code


def unrotated_surface_code(distance: int) -> Code:
    """Unrotated (planar) surface code, single sector — a matching graph.

    Data qubits on the edges of a ``d x d`` vertex lattice; Z-stabilizers on the
    vertex stars (weight 2/3/4, boundary qubits degree 1).  Matches the
    construction validated in the existing faithfulness suite.
    """
    if distance < 2:
        raise ValueError("surface distance must be >= 2")
    d = int(distance)
    nh = d * (d - 1)  # horizontal edges

    def hh(r: int, c: int) -> int:
        return r * (d - 1) + c

    def vv(r: int, c: int) -> int:
        return nh + r * d + c

    n_qubits = nh + (d - 1) * d
    checks: List[List[int]] = []
    for r in range(d):
        for c in range(d):
            star: List[int] = []
            if c - 1 >= 0:
                star.append(hh(r, c - 1))
            if c <= d - 2:
                star.append(hh(r, c))
            if r - 1 >= 0:
                star.append(vv(r - 1, c))
            if r <= d - 2:
                star.append(vv(r, c))
            if len(star) >= 2:
                checks.append(star)
    return Code(
        name=f"unrotated_surface_d{d}",
        check_to_qubits=_sorted_checks(checks),
        n_qubits=n_qubits,
        distance=d,
        description="Unrotated / planar surface code, single sector (matching graph).",
    )


def toric_code(size: int) -> Code:
    """Toric code on an ``L x L`` torus, single (vertex/Z) sector — matching graph.

    ``2*L^2`` qubits live on the edges of the torus; the ``L^2`` vertex checks
    each touch four edges and every edge is shared by exactly two vertices.
    Checks are dependent (rank ``L^2 - 1``); the code encodes two logical qubits.
    """
    if size < 2:
        raise ValueError("toric size must be >= 2")
    L = int(size)

    def he(r: int, c: int) -> int:  # horizontal edge
        return (r % L) * L + (c % L)

    def ve(r: int, c: int) -> int:  # vertical edge
        return L * L + (r % L) * L + (c % L)

    checks: List[List[int]] = []
    for r in range(L):
        for c in range(L):
            checks.append([he(r, c), he(r, c - 1), ve(r, c), ve(r - 1, c)])
    return Code(
        name=f"toric_L{L}",
        check_to_qubits=_sorted_checks(checks),
        n_qubits=2 * L * L,
        distance=L,
        description="Toric code on an L x L torus, vertex sector (matching graph).",
    )


def heavy_hex_code(distance: int) -> Code:
    """Heavy-hexagon-style distance-``d`` matching graph (single sector).

    A heavy-hex layout interleaves data qubits with flag qubits on the bonds of a
    hexagonal lattice.  This helper builds the *graphlike* Z-sector: a brick-wall
    of weight-2 and weight-4 checks in which every data qubit is shared by at most
    two checks, so the matching decoders apply.  It is intended for layout/scaling
    studies, not as a bit-exact replica of any specific device.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError("heavy_hex distance must be an odd integer >= 3")
    d = int(distance)

    def q(r: int, c: int) -> int:
        return r * d + c

    checks: List[List[int]] = []
    # Vertical weight-2 bonds (the "heavy" links) on alternating columns/rows.
    for r in range(d - 1):
        for c in range(d):
            if (r + c) % 2 == 0:
                checks.append([q(r, c), q(r + 1, c)])
    # Horizontal weight-2 bonds on the complementary sublattice, capped at
    # degree two so the result stays a matching graph.
    deg = np.zeros(d * d, dtype=np.int64)
    for ch in checks:
        for x in ch:
            deg[x] += 1
    for r in range(d):
        for c in range(d - 1):
            if (r + c) % 2 == 1 and deg[q(r, c)] < 2 and deg[q(r, c + 1)] < 2:
                checks.append([q(r, c), q(r, c + 1)])
                deg[q(r, c)] += 1
                deg[q(r, c + 1)] += 1
    logical = [q(0, c) for c in range(d)]
    return Code(
        name=f"heavy_hex_d{d}",
        check_to_qubits=_sorted_checks(checks),
        n_qubits=d * d,
        distance=d,
        logicals=[logical],
        description="Heavy-hexagon-style matching graph (single sector).",
    )


# ---------------------------------------------------------------------------
# Arbitrary parity-check matrices (LDPC / hypergraph-product / bicycle / custom)
# ---------------------------------------------------------------------------
def from_parity_check_matrix(
    H: Any, name: str = "custom", distance: Optional[int] = None
) -> Code:
    """Build a :class:`Code` from an arbitrary GF(2) parity-check matrix ``H``.

    Accepts a dense ``numpy`` array or any ``scipy.sparse`` matrix.  Rows are
    checks, columns are qubits.  Use this for LDPC, hypergraph-product, bicycle,
    or any custom code (decode with :class:`BPOSDDecoder` when not graphlike).
    """
    arr = _to_dense_binary(H)
    if arr.ndim != 2:
        raise ValueError(f"H must be 2D, got shape {arr.shape}")
    n_checks, n_qubits = arr.shape
    check_to_qubits = [
        sorted(int(c) for c in np.nonzero(arr[r])[0]) for r in range(n_checks)
    ]
    return Code(
        name=name,
        check_to_qubits=check_to_qubits,
        n_qubits=int(n_qubits),
        distance=distance,
        description="User-supplied parity-check matrix.",
    )


def _to_dense_binary(H: Any) -> np.ndarray:
    """Convert dense/scipy-sparse input to a dense uint8 GF(2) array."""
    if hasattr(H, "toarray"):  # scipy.sparse
        arr = H.toarray()
    else:
        arr = np.asarray(H)
    arr = (np.asarray(arr) % 2).astype(np.uint8)
    return arr


def _cyclic_shift(n: int) -> np.ndarray:
    """The n x n cyclic shift permutation matrix S (S[i, (i+1)%n] = 1)."""
    S = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        S[i, (i + 1) % n] = 1
    return S


def bivariate_bicycle_code(
    ell: int,
    m: int,
    a_terms: Sequence[Tuple[str, int]],
    b_terms: Sequence[Tuple[str, int]],
) -> Tuple[Code, Code]:
    """Bivariate-bicycle (BB) CSS LDPC code (Bravyi et al., 2024).

    Qubits live on a torus ``Z_ell x Z_m`` (``n = 2*ell*m`` physical qubits). With
    ``x = S_ell ⊗ I_m`` and ``y = I_ell ⊗ S_m`` (cyclic shifts), polynomials
    ``A = Σ a_terms`` and ``B = Σ b_terms`` give

        Hx = [A | B],   Hz = [B^T | A^T].

    ``a_terms`` / ``b_terms`` are lists of ``(var, power)`` with ``var in {'x','y'}``.
    Returns ``(code_x, code_z)`` (decode with BP-OSD). The famous ``[[144,12,12]]``
    code is ``ell=12, m=6, A=x^3+y+y^2, B=y^3+x+x^2``.
    """
    Ix, Iy = np.eye(ell, dtype=np.uint8), np.eye(m, dtype=np.uint8)
    Sx, Sy = _cyclic_shift(ell), _cyclic_shift(m)
    x = np.kron(Sx, Iy)
    y = np.kron(Ix, Sy)

    def poly(terms: Sequence[Tuple[str, int]]) -> np.ndarray:
        dim = ell * m
        M = np.zeros((dim, dim), dtype=np.uint8)
        for var, power in terms:
            base = x if var == "x" else y
            term = np.linalg.matrix_power(base.astype(np.int64), power) % 2
            M = (M + term.astype(np.uint8)) % 2
        return M

    A = poly(a_terms)
    B = poly(b_terms)
    Hx = np.hstack([A, B]).astype(np.uint8) % 2
    Hz = np.hstack([B.T, A.T]).astype(np.uint8) % 2

    code_x = from_parity_check_matrix(Hx, name=f"bb_x_{ell}_{m}")
    code_z = from_parity_check_matrix(Hz, name=f"bb_z_{ell}_{m}")
    code_x.description = "Bivariate-bicycle CSS code, X sector."
    code_z.description = "Bivariate-bicycle CSS code, Z sector."
    return code_x, code_z


def bicycle_code(n_circulant: int, weight: int = 4, seed: int = 0) -> Tuple[Code, Code]:
    """Bicycle CSS LDPC code from two random circulants.

    Builds two sparse circulants ``A`` and ``B`` of size ``n_circulant`` (each with
    ``weight//2`` ones per row) and forms ``Hx = [A | B]``, ``Hz = [B^T | A^T]``.
    Circulants commute, so ``Hx Hz^T = AB + BA = 2AB = 0 (mod 2)`` — a valid CSS
    code on ``2*n_circulant`` qubits. Decode with BP-OSD.
    """
    n = int(n_circulant)
    rng = np.random.default_rng(seed)

    def circulant() -> np.ndarray:
        row = np.zeros(n, dtype=np.uint8)
        row[rng.choice(n, size=max(1, weight // 2), replace=False)] = 1
        M = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            M[i] = np.roll(row, i)
        return M

    A, B = circulant(), circulant()
    Hx = np.hstack([A, B]).astype(np.uint8) % 2
    Hz = np.hstack([B.T, A.T]).astype(np.uint8) % 2
    code_x = from_parity_check_matrix(Hx, name=f"bicycle_x_{n}")
    code_z = from_parity_check_matrix(Hz, name=f"bicycle_z_{n}")
    code_x.description = "Bicycle CSS LDPC code, X sector."
    code_z.description = "Bicycle CSS LDPC code, Z sector."
    return code_x, code_z


def hypergraph_product(H1: Any, H2: Optional[Any] = None) -> Tuple[Code, Code]:
    """Tillich–Zémor hypergraph-product CSS code from seed matrix/matrices.

    Given a single seed ``H1`` (then ``H2 = H1``) or two seeds, returns
    ``(code_x, code_z)`` — the X- and Z-sector :class:`Code` objects of the
    resulting CSS code.  The classic GF(2) construction:

        Hx = [ H1 ⊗ I_{n2} | I_{r1} ⊗ H2^T ]
        Hz = [ I_{n1} ⊗ H2 | H1^T ⊗ I_{r2} ]

    These sectors are generally **not** graphlike — decode with BP-OSD.
    """
    A = _to_dense_binary(H1)
    B = _to_dense_binary(H2) if H2 is not None else A
    r1, n1 = A.shape
    r2, n2 = B.shape

    In1, In2 = np.eye(n1, dtype=np.uint8), np.eye(n2, dtype=np.uint8)
    Ir1, Ir2 = np.eye(r1, dtype=np.uint8), np.eye(r2, dtype=np.uint8)

    Hx = np.hstack([np.kron(A, In2), np.kron(Ir1, B.T)]).astype(np.uint8) % 2
    Hz = np.hstack([np.kron(In1, B), np.kron(A.T, Ir2)]).astype(np.uint8) % 2

    code_x = from_parity_check_matrix(Hx, name="hgp_x")
    code_z = from_parity_check_matrix(Hz, name="hgp_z")
    code_x.description = "Hypergraph-product CSS code, X sector."
    code_z.description = "Hypergraph-product CSS code, Z sector."
    return code_x, code_z


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------
def list_codes() -> List[str]:
    """Names of the built-in parametric code families."""
    return [
        "repetition_code",
        "ring_code",
        "rotated_surface_code",
        "unrotated_surface_code",
        "toric_code",
        "heavy_hex_code",
        "from_parity_check_matrix",
        "hypergraph_product",
    ]
