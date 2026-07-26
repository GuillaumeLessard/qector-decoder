"""
qector_decoder_v3.codes - Code-family helpers.

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
(``H @ decode(s) == s (mod 2)``) - see ``python/tests/test_codes.py``.

Examples
--------
>>> from qector_decoder_v3 import codes, BlossomDecoder
>>> code = codes.rotated_surface_code(5)
>>> code.n_qubits, code.n_checks
(25, 8)
>>> dec = BlossomDecoder(code.check_to_qubits, code.n_qubits)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Code",
    "bicycle_code",
    "bivariate_bicycle_code",
    "color_code",
    "css_logicals",
    "from_parity_check_matrix",
    "gf2_kernel",
    "gf2_rank",
    "heavy_hex_code",
    "hypergraph_product",
    "list_codes",
    "repetition_code",
    "ring_code",
    "rotated_surface_code",
    "toric_code",
    "unrotated_surface_code",
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
    check_to_qubits: list[list[int]]
    n_qubits: int
    distance: int | None = None
    logicals: list[list[int]] | None = None
    qubit_weights: np.ndarray | None = None
    description: str = ""
    _meta: dict = field(default_factory=dict, repr=False)

    # -- derived -----------------------------------------------------------
    @property
    def n_checks(self) -> int:
        return len(self.check_to_qubits)

    @property
    def num_logical_qubits(self) -> int:
        """Number of logical qubits ``k`` (0 when no logicals are defined)."""
        return 0 if self.logicals is None else len(self.logicals)

    def parity_check_matrix(self) -> np.ndarray:
        """Dense GF(2) parity-check matrix ``H`` of shape ``(n_checks, n_qubits)``."""
        H = np.zeros((self.n_checks, self.n_qubits), dtype=np.uint8)
        for ci, qs in enumerate(self.check_to_qubits):
            for q in qs:
                H[ci, q] ^= np.uint8(1)
        return H

    # convenient alias
    def H(self) -> np.ndarray:
        return self.parity_check_matrix()

    def logicals_matrix(self) -> np.ndarray | None:
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

    def random_error(self, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
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


def _sorted_checks(checks: list[list[int]]) -> list[list[int]]:
    return [sorted({int(q) for q in c}) for c in checks]


# ---------------------------------------------------------------------------
# 1D codes
# ---------------------------------------------------------------------------
def repetition_code(distance: int) -> Code:
    """Open 1D repetition code: ``d`` qubits in a line, ``d-1`` weight-2 checks.

    The two end qubits are boundary edges (degree 1).  Logical observable is a
    single boundary-crossing edge ``{0}`` - verified valid because every residual
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
    """Rotated surface code, single (Z) sector - a matching graph.

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

    checks: list[list[int]] = []
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
    """Unrotated (planar) surface code, single sector - a matching graph.

    Data qubits on the edges of a ``d x d`` vertex lattice; Z-stabilizers on the
    vertex stars (weight 2/3/4, boundary qubits degree 1).  Matches the
    construction validated in the existing faithfulness suite.

    .. note::
        As built (every vertex star with >= 2 edges included), this
        single-sector matching graph has ``k = 0`` in the CSS sense — there is
        no logical operator, so ``logicals`` is ``None`` and logical error
        rates are not defined on it.  For LER studies use
        :func:`rotated_surface_code`, :func:`toric_code`, or the circuit-level
        DEM path (:mod:`qector_decoder_v3.dem`).
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
    checks: list[list[int]] = []
    for r in range(d):
        for c in range(d):
            star: list[int] = []
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
    """Toric code on an ``L x L`` torus, single (vertex/Z) sector - matching graph.

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

    checks: list[list[int]] = []
    for r in range(L):
        for c in range(L):
            checks.append([he(r, c), he(r, c - 1), ve(r, c), ve(r - 1, c)])
    code = Code(
        name=f"toric_L{L}",
        check_to_qubits=_sorted_checks(checks),
        n_qubits=2 * L * L,
        distance=L,
        description="Toric code on an L x L torus, vertex sector (matching graph).",
    )
    # Derive the two logical operators from the plaquette (Z-stabiliser)
    # sector. The returned Code stays the vertex (matching) sector; the
    # plaquettes are only used here to compute ker(H_stars)/row(H_plaquettes).
    n_q = 2 * L * L
    Hz = np.zeros((L * L, n_q), dtype=np.uint8)
    for r in range(L):
        for c in range(L):
            row = r * L + c
            for q in (he(r, c), he(r + 1, c), ve(r, c), ve(r, c + 1)):
                Hz[row, q] ^= np.uint8(1)
    Lx, Lz = css_logicals(code.parity_check_matrix(), Hz)
    code.logicals = _matrix_to_logicals(Lx)
    code._meta["n_logical_qubits"] = int(Lx.shape[0])
    code._meta["logical_x_ops"] = Lx
    code._meta["logical_z_ops"] = Lz
    return code


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

    checks: list[list[int]] = []
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

def color_code(distance: int) -> Code:
    """Triangular 6.6.6 color code (C2).  Qubits on vertices of a honeycomb
    lattice tiling of side length ``distance``.  X-checks are the "red" faces
    (hexagons); Z-checks are derived from the complementary "green+blue" faces
    to compute logical operators via the CSS quotient.  Returns a graphlike
    single-sector ``Code`` with ``logicals`` populated (``k=2`` for the planar
    code).  Distances < 2 are unsupported.
    """
    if distance < 2:
        raise ValueError("color code distance must be >= 2")
    d = int(distance)
    # Qubits are vertices of a triangular lattice of width d.
    # Each vertex is indexed by row r [0..d) and column c [0..d-r).
    # A vertex's neighbours in the six directions are used only to define
    # the faces; the actual parity checks are the hexagonal faces themselves.
    def qidx(r: int, c: int) -> int | None:
        if 0 <= r < d and 0 <= c < d - r:
            # Cumulative count before row r: sum_{i=0}^{r-1} (d-i) = r*d - r*(r-1)//2
            return r * d - r * (r - 1) // 2 + c
        return None

    n_q = (d * (d + 1)) // 2
    # Each interior vertex belongs to exactly 3 hexagons (one per colour).
    # Build the "red" faces: centres at positions (r+1/3, c+2/3) in skewed
    # coordinates — these are the faces whose top vertex is at (r,c) with
    # r >= 1 and c >= 1.
    red_faces: list[list[int]] = []
    for r in range(1, d):
        for c in range(1, d - r):
            vs = [
                qidx(r, c), qidx(r, c - 1), qidx(r - 1, c - 1),
                qidx(r - 1, c), qidx(r - 1, c + 1), qidx(r, c + 1),
            ]
            all_v = [v for v in vs if v is not None]
            if len(all_v) >= 3:
                red_faces.append(sorted(set(all_v)))

    # Build the dual "green" faces: shifted by one lattice unit from red faces.
    # Centres at (r+2/3, c+1/3) — the faces whose top-left vertex is (r,c-1).
    green_faces: list[list[int]] = []
    for r in range(1, d):
        for c in range(1, d - r + 1):
            vs = [
                qidx(r, c - 1), qidx(r - 1, c - 1), qidx(r - 1, c),
                qidx(r, c), qidx(r + 1, c), qidx(r + 1, c - 1),
            ]
            all_v = [v for v in vs if v is not None]
            if len(all_v) >= 3:
                green_faces.append(sorted(set(all_v)))

    # Use red faces as the primary checks (Hx). If too few, fall back.
    checks = red_faces if red_faces else (green_faces if green_faces else [[0]])
    code = Code(
        name=f"color_code_d{d}",
        check_to_qubits=_sorted_checks(checks),
        n_qubits=n_q,
        distance=d,
        description=f"Triangular 6.6.6 color code (dist {d}, planar, k=2).",
    )
    # Derive logicals from the two face colours.
    Hx = code.parity_check_matrix()
    n = code.n_qubits
    if green_faces:
        Hz = np.zeros((len(green_faces), n), dtype=np.uint8)
        for ci, qs in enumerate(green_faces):
            for qb in qs:
                if qb < n:
                    Hz[ci, qb] ^= 1
        if not np.any((Hx @ Hz.T) % 2):
            Lx, Lz = css_logicals(Hx, Hz)
            code.logicals = _matrix_to_logicals(Lx)
            code._meta["n_logical_qubits"] = int(Lx.shape[0])
            code._meta["logical_x_ops"] = Lx
            code._meta["logical_z_ops"] = Lz
    return code


def from_parity_check_matrix(H: Any, name: str = "custom", distance: int | None = None) -> Code:
    """Build a :class:`Code` from an arbitrary GF(2) parity-check matrix ``H``.

    Accepts a dense ``numpy`` array or any ``scipy.sparse`` matrix.  Rows are
    checks, columns are qubits.  Use this for LDPC, hypergraph-product, bicycle,
    or any custom code (decode with :class:`BPOSDDecoder` when not graphlike).
    """
    arr = _to_dense_binary(H)
    if arr.ndim != 2:
        raise ValueError(f"H must be 2D, got shape {arr.shape}")
    n_checks, n_qubits = arr.shape
    check_to_qubits = [sorted(int(c) for c in np.nonzero(arr[r])[0]) for r in range(n_checks)]
    return Code(
        name=name,
        check_to_qubits=check_to_qubits,
        n_qubits=int(n_qubits),
        distance=distance,
        description="User-supplied parity-check matrix.",
    )


def _to_dense_binary(H: Any) -> np.ndarray:
    """Convert dense/scipy-sparse input to a dense uint8 GF(2) array.

    Raises ``ValueError`` rather than silently mis-converting when ``H``
    clearly isn't a parity-check matrix. The most common misuse: passing a
    check-to-qubits adjacency list (rows of *qubit indices*, as returned by
    ``generate_repetition_code_checks`` / ``generate_surface_code_checks``
    and consumed by ``BPOSDDecoder`` / ``BlossomDecoder`` /
    ``UnionFindDecoder``) where this function instead expects a dense
    ``(n_checks, n_qubits)`` 0/1 matrix. Left unchecked, that mistake
    silently "succeeds": the adjacency list is read as if it were already a
    dense matrix, so ``n_qubits`` becomes the check *weight* (e.g. 2)
    instead of the true qubit count, and every decode call then returns a
    correction of the wrong length with no error at all.
    """
    if hasattr(H, "toarray"):  # scipy.sparse
        arr = H.toarray()
    else:
        try:
            arr = np.asarray(H)
        except ValueError as e:
            raise ValueError(
                "Could not interpret H as a dense array (ragged rows?). If you "
                "have a check-to-qubits adjacency list (rows of qubit indices, "
                "e.g. from generate_surface_code_checks), this function expects "
                "a dense 0/1 parity-check matrix instead -- use "
                "BPOSDDecoder(check_to_qubits, n_qubits, error_rate) for the "
                f"adjacency-list format. Original error: {e}"
            ) from e
    if arr.size > 0 and arr.dtype != bool:
        max_abs = float(np.abs(arr).max())
        if max_abs > 1:
            raise ValueError(
                f"H has entries with magnitude up to {int(max_abs)}, but a GF(2) "
                "parity-check matrix must contain only 0/1. This usually means a "
                "check-to-qubits adjacency list (rows of qubit indices) was "
                "passed where a dense matrix was expected. For that format, use "
                "BPOSDDecoder(check_to_qubits, n_qubits, error_rate) instead; for "
                "a dense matrix, pass shape (n_checks, n_qubits) with only 0s "
                "and 1s (e.g. code.parity_check_matrix())."
            )
    arr = (np.asarray(arr) % 2).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# GF(2) linear algebra — logical-operator computation
# ---------------------------------------------------------------------------
def _gf2_rref(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Row-reduced echelon form over GF(2). Returns ``(RREF, pivot_columns)``."""
    M = (np.asarray(M, dtype=np.uint8) % 2).copy()
    rows, cols = M.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + int(nz[0])
        if p != r:
            M[[r, p]] = M[[p, r]]
        for i in np.nonzero(M[:, c])[0]:
            if i != r:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
    return M, pivots


def gf2_rank(M: Any) -> int:
    """Rank of a GF(2) matrix."""
    _, pivots = _gf2_rref(np.asarray(M, dtype=np.uint8) % 2)
    return len(pivots)


def gf2_kernel(H: Any) -> np.ndarray:
    """Basis of ``ker(H)`` over GF(2), shape ``(n - rank(H), n)``.

    Every row ``v`` satisfies ``H @ v == 0 (mod 2)``.  Rows are in reduced
    form: one free variable per row, in increasing column order.
    """
    H = _to_dense_binary(H)
    _m, n = H.shape
    R, pivots = _gf2_rref(H)
    pivset = set(pivots)
    free = [c for c in range(n) if c not in pivset]
    K = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        K[i, f] = 1
        for row, p in enumerate(pivots):
            if R[row, f]:
                K[i, p] = 1
    return K


def _gf2_row_basis(M: np.ndarray) -> np.ndarray:
    """A basis of the row space of ``M`` (the nonzero rows of its RREF)."""
    M = np.asarray(M, dtype=np.uint8) % 2
    if M.ndim != 2 or M.shape[0] == 0:
        width = M.shape[1] if M.ndim == 2 else 0
        return np.zeros((0, width), dtype=np.uint8)
    R, pivots = _gf2_rref(M)
    return R[: len(pivots)].copy()


def _gf2_quotient_basis(K: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Rows of ``K`` that stay independent modulo ``rowspan(S)``.

    Grows a basis starting from a row basis of ``S``; every vector of ``K``
    that raises the rank is kept.  For CSS codes (``K = ker(Hx)``,
    ``S = rowspan(Hz) ⊆ ker(Hx)``) the result spans exactly the
    logical-operator space ``ker(Hx) / rowspan(Hz)``.
    """
    K = np.asarray(K, dtype=np.uint8) % 2
    n = K.shape[1]
    cur = _gf2_row_basis(S) if len(S) else np.zeros((0, n), dtype=np.uint8)
    cur_rank = gf2_rank(cur) if len(cur) else 0
    keep: list[np.ndarray] = []
    for v in K:
        trial = np.vstack([cur, v]) if len(cur) else v.reshape(1, -1)
        if gf2_rank(trial) > cur_rank:
            cur = trial
            cur_rank += 1
            keep.append(v.copy())
    if not keep:
        return np.zeros((0, n), dtype=np.uint8)
    return np.asarray(keep, dtype=np.uint8)


def _gf2_inv(M: np.ndarray) -> np.ndarray:
    """Inverse of a square GF(2) matrix (Gauss-Jordan). Raises if singular."""
    M = (np.asarray(M, dtype=np.uint8) % 2).copy()
    n = M.shape[0]
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("matrix must be square")
    A = np.concatenate([M, np.eye(n, dtype=np.uint8)], axis=1)
    # Every column must pivot (else the matrix is singular and we raise), so
    # the pivot row for column c is always c itself.
    for c in range(n):
        nz = np.nonzero(A[c:, c])[0]
        if nz.size == 0:
            raise ValueError("matrix is singular over GF(2)")
        p = c + int(nz[0])
        if p != c:
            A[[c, p]] = A[[p, c]]
        for i in np.nonzero(A[:, c])[0]:
            if i != c:
                A[i] ^= A[c]
    return A[:, n:]


def css_logicals(Hx: Any, Hz: Any) -> tuple[np.ndarray, np.ndarray]:
    """Logical X and Z operators of a CSS code as biorthogonal GF(2) bases.

    Given the X-check matrix ``Hx`` and Z-check matrix ``Hz`` (both
    ``(n_checks, n_qubits)``, requiring ``Hx @ Hz.T == 0 (mod 2)``), returns
    ``(Lx, Lz)`` where

    * ``Lz`` rows span ``ker(Hx) / rowspan(Hz)`` — Z-type logical operators;
    * ``Lx`` rows span ``ker(Hz) / rowspan(Hx)`` — X-type logical operators;
    * ``Lx @ Lz.T == I_k (mod 2)`` — biorthogonal, so the logical-flip vector
      of any residual ``r ∈ ker(Hx)`` is exactly ``(Lx @ r) mod 2``.

    ``k = n - rank(Hx) - rank(Hz)`` is the number of logical qubits.
    Representatives are valid but not minimum-weight.
    """
    Hx = _to_dense_binary(Hx)
    Hz = _to_dense_binary(Hz)
    if Hx.ndim != 2 or Hz.ndim != 2:
        raise ValueError("Hx and Hz must be 2D parity-check matrices")
    if Hx.shape[1] != Hz.shape[1]:
        raise ValueError(f"Hx has {Hx.shape[1]} columns but Hz has {Hz.shape[1]}")
    n = Hx.shape[1]
    if np.any((Hx @ Hz.T) % 2):
        raise ValueError("not a CSS pair: Hx @ Hz.T != 0 (mod 2)")
    Kz = gf2_kernel(Hx)
    Kx = gf2_kernel(Hz)
    Lz = _gf2_quotient_basis(Kz, Hz)
    Lx = _gf2_quotient_basis(Kx, Hx)
    k = n - gf2_rank(Hx) - gf2_rank(Hz)
    if Lz.shape[0] != k or Lx.shape[0] != k:
        raise RuntimeError(
            f"internal error: quotient basis size ({Lz.shape[0]}, {Lx.shape[0]}) != k ({k})"
        )
    if k == 0:
        empty = np.zeros((0, n), dtype=np.uint8)
        return empty, empty.copy()
    pairing = (Lx @ Lz.T) % 2
    Lx = (_gf2_inv(pairing) @ Lx) % 2
    return Lx.astype(np.uint8), Lz.astype(np.uint8)


def _matrix_to_logicals(L: np.ndarray) -> list[list[int]]:
    """Dense GF(2) rows -> the ``Code.logicals`` list-of-sorted-index-lists form."""
    return [sorted(int(q) for q in np.nonzero(row)[0]) for row in np.asarray(L, dtype=np.uint8)]


def _attach_css_logicals(code_x: Code, code_z: Code, Hx: np.ndarray, Hz: np.ndarray) -> None:
    """Compute CSS logical operators and attach them to both sector codes.

    ``code_x`` (``H = Hx``) corrects Z-type errors; its measurable observables
    are the X-type duals ``Lx`` — a residual ``r ∈ ker(Hx)`` is a logical
    failure iff ``(Lx @ r) % 2`` has any nonzero entry.  Symmetrically for
    ``code_z``.  Both operator matrices are also stored in ``_meta`` as
    ``logical_x_ops`` / ``logical_z_ops`` (dense uint8), with
    ``n_logical_qubits = k``.
    """
    Lx, Lz = css_logicals(Hx, Hz)
    k = int(Lx.shape[0])
    code_x.logicals = _matrix_to_logicals(Lx)
    code_z.logicals = _matrix_to_logicals(Lz)
    for code in (code_x, code_z):
        code._meta["n_logical_qubits"] = k
        code._meta["logical_x_ops"] = Lx
        code._meta["logical_z_ops"] = Lz


def _cyclic_shift(n: int) -> np.ndarray:
    """The n x n cyclic shift permutation matrix S (S[i, (i+1)%n] = 1)."""
    S = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        S[i, (i + 1) % n] = 1
    return S


def bivariate_bicycle_code(
    ell: int,
    m: int,
    a_terms: Sequence[tuple[str, int]],
    b_terms: Sequence[tuple[str, int]],
) -> tuple[Code, Code]:
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

    def poly(terms: Sequence[tuple[str, int]]) -> np.ndarray:
        dim = ell * m
        M: np.ndarray = np.zeros((dim, dim), dtype=np.uint8)
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
    _attach_css_logicals(code_x, code_z, Hx, Hz)
    return code_x, code_z


def bicycle_code(n_circulant: int, weight: int = 4, seed: int = 0) -> tuple[Code, Code]:
    """Bicycle CSS LDPC code from two random circulants.

    Builds two sparse circulants ``A`` and ``B`` of size ``n_circulant`` (each with
    ``weight//2`` ones per row) and forms ``Hx = [A | B]``, ``Hz = [B^T | A^T]``.
    Circulants commute, so ``Hx Hz^T = AB + BA = 2AB = 0 (mod 2)`` - a valid CSS
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
    _attach_css_logicals(code_x, code_z, Hx, Hz)
    return code_x, code_z


def hypergraph_product(H1: Any, H2: Any | None = None) -> tuple[Code, Code]:
    """Tillich-Zémor hypergraph-product CSS code from seed matrix/matrices.

    Given a single seed ``H1`` (then ``H2 = H1``) or two seeds, returns
    ``(code_x, code_z)`` - the X- and Z-sector :class:`Code` objects of the
    resulting CSS code.  The classic GF(2) construction:

        Hx = [ H1 ⊗ I_{n2} | I_{r1} ⊗ H2^T ]
        Hz = [ I_{n1} ⊗ H2 | H1^T ⊗ I_{r2} ]

    These sectors are generally **not** graphlike - decode with BP-OSD.
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
    _attach_css_logicals(code_x, code_z, Hx, Hz)
    return code_x, code_z


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------
def list_codes() -> list[str]:
    """Names of the built-in parametric code families."""
    return [
        "repetition_code",
        "ring_code",
        "rotated_surface_code",
        "unrotated_surface_code",
        "toric_code",
        "heavy_hex_code",
        "color_code",
        "from_parity_check_matrix",
        "hypergraph_product",
    ]
