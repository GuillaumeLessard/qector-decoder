"""Rigorous syndrome-faithfulness probe for the Union-Find decoders.

Generates random errors -> syndromes (guaranteed reachable) on several code
families (rotated surface, unrotated/planar surface, toric ring, repetition),
decodes, and checks that H @ correction == syndrome (mod 2).

A "syndrome failure" is a correction whose syndrome does NOT match the input
syndrome -- i.e. the decoder produced an invalid correction. This is distinct
from a *logical* failure (valid correction but wrong coset).
"""
import numpy as np
from qector_decoder_v3 import UnionFindDecoder, FastUnionFindDecoder


# ---------------------------------------------------------------------------
# Code constructions
# ---------------------------------------------------------------------------
def repetition_code(d):
    """1D distance-d repetition code (the simplest surface code with a boundary).
    d data qubits, d-1 weight-2 checks. Qubits 0 and d-1 are boundary qubits."""
    return [[i, i + 1] for i in range(d - 1)], d


def ring_code(n):
    """1D periodic ring (toric, no boundary). n qubits, n weight-2 checks."""
    return [[i, (i + 1) % n] for i in range(n)], n


def rotated_surface_code(d):
    """Rotated surface code distance d. Returns Z-checks (detect X errors).

    d*d data qubits on a grid, index = r*d + c.
    Bulk: weight-4 checkerboard plaquettes. Boundary: weight-2 plaquettes on
    the left/right edges, giving the rotated code its characteristic boundary.
    """
    checks = []
    # Bulk weight-4 plaquettes, checkerboard colouring (Z where (r+c) even)
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 0:
                checks.append([r * d + c, r * d + c + 1,
                               (r + 1) * d + c, (r + 1) * d + c + 1])
    # Weight-2 boundary plaquettes on top/bottom rows (rotated-code boundaries)
    for c in range(d - 1):
        if c % 2 == 1:                      # top boundary
            checks.append([c, c + 1])
        # bottom boundary, complementary parity
        if (d - 2 + c) % 2 == 0:
            checks.append([(d - 1) * d + c, (d - 1) * d + c + 1])
    return checks, d * d


def unrotated_surface_code(d):
    """Unrotated (standard) planar surface code, distance d. Returns Z-checks.

    Data qubits live on the edges of a d x (d-1)... we use the standard
    construction: vertices of an L x L lattice carry star (Z) stabilizers.
    Here we build the planar code via vertex stars with open boundaries, which
    yields weight-3/weight-4 checks and boundary qubits of degree 1.
    """
    # Data qubits on horizontal and vertical edges of a (d) x (d) vertex lattice.
    # Horizontal edges: h[r][c], r in 0..d, c in 0..d-1
    # Vertical edges:   v[r][c], r in 0..d-1, c in 0..d
    nh = d * (d - 1)
    def h_id(r, c):       # r in 0..d-1? we use r in 0..d, c in 0..d-2
        return r * (d - 1) + c
    def v_id(r, c):
        return nh + r * d + c
    n_qubits = nh + (d - 1) * d
    checks = []
    # Star (vertex) Z-stabilizers at interior vertices (r,c), r in 1..d-1, c in 1..d-1
    for r in range(d):
        for c in range(d):
            star = []
            # left/right horizontal edges
            if c - 1 >= 0:
                star.append(h_id(r, c - 1))
            if c <= d - 2:
                star.append(h_id(r, c))
            # up/down vertical edges
            if r - 1 >= 0:
                star.append(v_id(r - 1, c))
            if r <= d - 2:
                star.append(v_id(r, c))
            if len(star) >= 2:
                checks.append(star)
    return checks, n_qubits


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def syndrome_of(check_to_qubits, vec, n_checks):
    syn = np.zeros(n_checks, dtype=np.uint8)
    for ci, qubits in enumerate(check_to_qubits):
        p = 0
        for q in qubits:
            p ^= int(vec[q])
        syn[ci] = p
    return syn


def run_family(name, check_to_qubits, n_qubits, n_trials=2000, p=0.10, seed=12345):
    rng = np.random.default_rng(seed)
    n_checks = len(check_to_qubits)
    uf = UnionFindDecoder(check_to_qubits, n_qubits)
    fuf = FastUnionFindDecoder(check_to_qubits, n_qubits)
    uf_fail = 0
    fuf_fail = 0
    nontrivial = 0
    for _ in range(n_trials):
        error = (rng.random(n_qubits) < p).astype(np.uint8)
        syn = syndrome_of(check_to_qubits, error, n_checks)
        if syn.sum() > 0:
            nontrivial += 1
        # UnionFind (accurate)
        corr = np.asarray(uf.decode(syn))
        if not np.array_equal(syndrome_of(check_to_qubits, corr, n_checks), syn):
            uf_fail += 1
        # FastUnionFind (heuristic)
        corr_f = np.asarray(fuf.decode(syn))
        if not np.array_equal(syndrome_of(check_to_qubits, corr_f, n_checks), syn):
            fuf_fail += 1
    print(f"{name:38s} q={n_qubits:4d} checks={n_checks:4d} "
          f"nontrivial={nontrivial:4d}/{n_trials}  "
          f"UnionFind_synfail={uf_fail:4d} ({100*uf_fail/n_trials:5.1f}%)  "
          f"FastUF_synfail={fuf_fail:4d} ({100*fuf_fail/n_trials:5.1f}%)")
    return uf_fail, fuf_fail


if __name__ == "__main__":
    print("=" * 120)
    families = []
    for d in (3, 5, 7, 9):
        c, n = repetition_code(d);          families.append((f"repetition d={d} (1D, boundary)", c, n))
    for n in (8, 12, 24):
        c, nn = ring_code(n);               families.append((f"ring/toric n={n} (no boundary)", c, nn))
    for d in (3, 5, 7):
        c, n = rotated_surface_code(d);     families.append((f"rotated surface d={d}", c, n))
    for d in (3, 4, 5):
        c, n = unrotated_surface_code(d);   families.append((f"unrotated/planar surface d={d}", c, n))

    tot_uf = tot_fuf = 0
    for name, c, n in families:
        a, b = run_family(name, c, n)
        tot_uf += a; tot_fuf += b
    print("=" * 120)
    print(f"TOTAL  UnionFind syndrome-failures = {tot_uf}   FastUnionFind syndrome-failures = {tot_fuf}")
