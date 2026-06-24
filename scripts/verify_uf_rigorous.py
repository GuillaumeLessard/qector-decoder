import numpy as np
import time
from qector_decoder_v3 import UnionFindDecoder, FastUnionFindDecoder

# Code constructions
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
    for c in range(d - 1):
        if c % 2 == 1:
            checks.append([c, c + 1])
        if (d - 2 + c) % 2 == 0:
            checks.append([(d - 1) * d + c, (d - 1) * d + c + 1])
    return checks, d * d

def unrotated_surface_code(d):
    nh = d * (d - 1)
    def h_id(r, c):
        return r * (d - 1) + c
    def v_id(r, c):
        return nh + r * d + c
    n_qubits = nh + (d - 1) * d
    checks = []
    for r in range(d):
        for c in range(d):
            star = []
            if c - 1 >= 0:
                star.append(h_id(r, c - 1))
            if c <= d - 2:
                star.append(h_id(r, c))
            if r - 1 >= 0:
                star.append(v_id(r - 1, c))
            if r <= d - 2:
                star.append(v_id(r, c))
            if len(star) >= 2:
                checks.append(star)
    return checks, n_qubits

def syndrome_of(check_to_qubits, vec, n_checks):
    syn = np.zeros(n_checks, dtype=np.uint8)
    for ci, qubits in enumerate(check_to_qubits):
        p = 0
        for q in qubits:
            p ^= int(vec[q])
        syn[ci] = p
    return syn

def run_rigorous_test(name, check_to_qubits, n_qubits, n_trials=50000, p=0.10, seed=12345):
    rng = np.random.default_rng(seed)
    n_checks = len(check_to_qubits)
    uf = UnionFindDecoder(check_to_qubits, n_qubits)
    fuf = FastUnionFindDecoder(check_to_qubits, n_qubits)
    uf_fail = 0
    fuf_fail = 0
    nontrivial = 0
    
    t0 = time.perf_counter()
    for _ in range(n_trials):
        error = (rng.random(n_qubits) < p).astype(np.uint8)
        syn = syndrome_of(check_to_qubits, error, n_checks)
        if syn.sum() > 0:
            nontrivial += 1
            
        # UnionFind
        corr = np.asarray(uf.decode(syn))
        if not np.array_equal(syndrome_of(check_to_qubits, corr, n_checks), syn):
            uf_fail += 1
            
        # FastUnionFind
        corr_f = np.asarray(fuf.decode(syn))
        if not np.array_equal(syndrome_of(check_to_qubits, corr_f, n_checks), syn):
            fuf_fail += 1
            
    elapsed = time.perf_counter() - t0
    print(f"{name:38s} q={n_qubits:4d} checks={n_checks:4d} "
          f"nontrivial={nontrivial:5d}/{n_trials:5d}  "
          f"UF_synfail={uf_fail:4d}  FF_synfail={fuf_fail:4d} ({elapsed:.1f}s)")
    return uf_fail, fuf_fail

if __name__ == "__main__":
    print("=" * 120)
    print("RUNNING RIGOROUS LARGE-SCALE SYNDROME FAITHFULNESS AUDIT (Phenomenological IID Bit-Flip Noise)")
    print("=" * 120)
    
    # We run 50,000 trials on smaller codes, and 10,000 trials on larger codes.
    tests = [
        # Repetition (1D boundary)
        ("repetition d=5 (50k)", *repetition_code(5), 50000),
        ("repetition d=11 (50k)", *repetition_code(11), 50000),
        ("repetition d=31 (10k)", *repetition_code(31), 10000),
        ("repetition d=51 (10k)", *repetition_code(51), 10000),
        
        # Ring/Toric (1D no boundary)
        ("ring n=12 (50k)", *ring_code(12), 50000),
        ("ring n=24 (50k)", *ring_code(24), 50000),
        ("ring n=50 (10k)", *ring_code(50), 10000),
        ("ring n=100 (10k)", *ring_code(100), 10000),
        
        # Rotated Surface
        ("rotated surface d=3 (50k)", *rotated_surface_code(3), 50000),
        ("rotated surface d=5 (50k)", *rotated_surface_code(5), 50000),
        ("rotated surface d=7 (10k)", *rotated_surface_code(7), 10000),
        ("rotated surface d=9 (10k)", *rotated_surface_code(9), 10000),
        ("rotated surface d=11 (10k)", *rotated_surface_code(11), 10000),
        ("rotated surface d=13 (10k)", *rotated_surface_code(13), 10000),
        
        # Unrotated/Planar Surface
        ("unrotated surface d=3 (50k)", *unrotated_surface_code(3), 50000),
        ("unrotated surface d=5 (50k)", *unrotated_surface_code(5), 50000),
        ("unrotated surface d=7 (10k)", *unrotated_surface_code(7), 10000),
    ]
    
    tot_uf = tot_fuf = 0
    for name, c, n, trials in tests:
        a, b = run_rigorous_test(name, c, n, n_trials=trials, p=0.08)
        tot_uf += a
        tot_fuf += b
    print("=" * 120)
    print(f"Rigorous Audit Total: UnionFind failures = {tot_uf}   FastUnionFind failures = {tot_fuf}")
    print("=" * 120)
