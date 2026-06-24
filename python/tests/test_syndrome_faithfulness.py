"""Syndrome-faithfulness regression suite.

Every decoder must satisfy ``H @ correction == syndrome (mod 2)`` for any
reachable syndrome (one generated from a real error). The absence of this exact
check is why the broken Union-Find / batch / GPU decoders went unnoticed — the
older tests only asserted output *shape*. These tests assert correctness itself,
across rotated-surface, unrotated/planar-surface, ring (toric) and repetition
codes, so the regression cannot recur silently.
"""
import numpy as np
import pytest
import qector_decoder_v3 as qd


# ---------------------------------------------------------------------------
# Code families (all proper matching graphs: each qubit in <= 2 checks)
# ---------------------------------------------------------------------------
def repetition_code(d):
    return [[i, i + 1] for i in range(d - 1)], d


def ring_code(n):
    return [[i, (i + 1) % n] for i in range(n)], n


def rotated_surface_code(d):
    checks = []
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 0:
                checks.append([r * d + c, r * d + c + 1,
                               (r + 1) * d + c, (r + 1) * d + c + 1])
    return checks, d * d


def unrotated_surface_code(d):
    """Planar surface code: data qubits on edges of a d x d vertex lattice,
    Z-stabilizers on vertex stars (weight 2/3/4, boundary qubits degree 1)."""
    nh = d * (d - 1)
    h = lambda r, c: r * (d - 1) + c
    v = lambda r, c: nh + r * d + c
    n_qubits = nh + (d - 1) * d
    checks = []
    for r in range(d):
        for c in range(d):
            star = []
            if c - 1 >= 0:
                star.append(h(r, c - 1))
            if c <= d - 2:
                star.append(h(r, c))
            if r - 1 >= 0:
                star.append(v(r - 1, c))
            if r <= d - 2:
                star.append(v(r, c))
            if len(star) >= 2:
                checks.append(star)
    return checks, n_qubits


CODES = [
    ("repetition d=7", *repetition_code(7)),
    ("repetition d=15", *repetition_code(15)),
    ("ring n=12", *ring_code(12)),
    ("ring n=24", *ring_code(24)),
    ("rotated surface d=5", *rotated_surface_code(5)),
    ("rotated surface d=7", *rotated_surface_code(7)),
    ("unrotated surface d=4", *unrotated_surface_code(4)),
    ("unrotated surface d=5", *unrotated_surface_code(5)),
]


def _H(check_to_qubits, n_qubits):
    H = np.zeros((len(check_to_qubits), n_qubits), dtype=np.uint8)
    for ci, qs in enumerate(check_to_qubits):
        for q in qs:
            H[ci, q] ^= 1
    return H


def _reachable_syndromes(H, n_qubits, n_shots, p, seed):
    rng = np.random.default_rng(seed)
    errors = (rng.random((n_shots, n_qubits)) < p).astype(np.uint8)
    return errors, (errors @ H.T) & 1


SINGLE_DECODERS = ["UnionFind", "FastUnionFind", "Blossom", "SparseBlossom",
                   "BPOSD", "CPUBatch"]


def _make(name, c2q, n):
    if name == "UnionFind":
        return qd.UnionFindDecoder(c2q, n)
    if name == "FastUnionFind":
        return qd.FastUnionFindDecoder(c2q, n)
    if name == "Blossom":
        return qd.BlossomDecoder(c2q, n)
    if name == "SparseBlossom":
        return qd.SparseBlossomDecoder(c2q, n)
    if name == "BPOSD":
        return qd.BPOSDDecoder(c2q, n, 0.08)
    if name == "CPUBatch":
        return qd.CPUBatchDecoder(c2q, n)
    raise ValueError(name)


@pytest.mark.parametrize("decoder_name", SINGLE_DECODERS)
@pytest.mark.parametrize("code", CODES, ids=[c[0] for c in CODES])
def test_single_decode_is_syndrome_faithful(decoder_name, code):
    name, c2q, n = code
    H = _H(c2q, len(c2q) and n)
    dec = _make(decoder_name, c2q, n)
    _, syndromes = _reachable_syndromes(H, n, 400, p=0.08, seed=1234)
    for s in syndromes:
        corr = np.asarray(dec.decode(s.astype(np.uint8))).astype(np.uint8)
        assert corr.shape == (n,)
        assert np.array_equal((H @ corr) & 1, s), (
            f"{decoder_name} on {name}: H@c != s")


@pytest.mark.parametrize("code", CODES, ids=[c[0] for c in CODES])
def test_batch_decoders_are_syndrome_faithful(code):
    name, c2q, n = code
    H = _H(c2q, n)
    _, syndromes = _reachable_syndromes(H, n, 256, p=0.08, seed=99)
    syndromes = syndromes.astype(np.uint8)
    for dname, dec in [("BatchDecoder", qd.BatchDecoder(c2q, n)),
                       ("CPUBatch", qd.CPUBatchDecoder(c2q, n))]:
        out = np.asarray(dec.batch_decode(syndromes))
        assert out.shape == (len(syndromes), n)
        for i in range(len(syndromes)):
            assert np.array_equal((H @ out[i]) & 1, syndromes[i]), (
                f"{dname} on {name}: row {i} H@c != s")


@pytest.mark.parametrize("code", CODES, ids=[c[0] for c in CODES])
def test_hybrid_modes_are_syndrome_faithful(code):
    name, c2q, n = code
    H = _H(c2q, n)
    dec = qd.HybridDecoder(c2q, n)
    _, syndromes = _reachable_syndromes(H, n, 200, p=0.06, seed=7)
    for s in syndromes:
        s = s.astype(np.uint8)
        for mode in (dec.decode_standard, dec.decode_heuristic, dec.decode_hybrid):
            corr = np.asarray(mode(s)).astype(np.uint8)
            assert np.array_equal((H @ corr) & 1, s), f"Hybrid on {name}: H@c != s"


def test_lookup_table_is_syndrome_faithful():
    c2q, n = rotated_surface_code(5)
    H = _H(c2q, n)
    dec = qd.LookupTableDecoder(c2q, n)
    dec.build_table(200000)
    _, syndromes = _reachable_syndromes(H, n, 400, p=0.08, seed=3)
    for s in syndromes:
        corr = np.asarray(dec.decode(s.astype(np.uint8))).astype(np.uint8)
        assert np.array_equal((H @ corr) & 1, s), "LookupTable: H@c != s"


@pytest.mark.skipif(not qd.CUDABatchDecoder.is_available(),
                    reason="CUDA not available")
@pytest.mark.parametrize("code", CODES, ids=[c[0] for c in CODES])
def test_cuda_bit_identical_and_faithful(code):
    name, c2q, n = code
    H = _H(c2q, n)
    cpu = qd.UnionFindDecoder(c2q, n)
    cuda = qd.CUDABatchDecoder(c2q, n)
    _, syndromes = _reachable_syndromes(H, n, 256, p=0.08, seed=42)
    syndromes = syndromes.astype(np.uint8)
    ref = cpu.batch_decode(syndromes)
    got = cuda.batch_decode(syndromes)
    assert np.array_equal(got, ref), f"CUDA != CPU on {name}"
    for i in range(len(syndromes)):
        assert np.array_equal((H @ got[i]) & 1, syndromes[i])


@pytest.mark.skipif(not qd.OpenCLBatchDecoder.is_available(),
                    reason="OpenCL not available")
@pytest.mark.parametrize("code", CODES, ids=[c[0] for c in CODES])
def test_opencl_bit_identical_and_faithful(code):
    name, c2q, n = code
    H = _H(c2q, n)
    cpu = qd.UnionFindDecoder(c2q, n)
    ocl = qd.OpenCLBatchDecoder(c2q, n)
    _, syndromes = _reachable_syndromes(H, n, 256, p=0.08, seed=42)
    syndromes = syndromes.astype(np.uint8)
    ref = cpu.batch_decode(syndromes)
    got = ocl.batch_decode(syndromes)
    assert np.array_equal(got, ref), f"OpenCL != CPU on {name}"
    for i in range(len(syndromes)):
        assert np.array_equal((H @ got[i]) & 1, syndromes[i])
