import time
import tracemalloc
import numpy as np
from pymatching import Matching
from qector_decoder_v3 import BlossomDecoder

def rotated_surface_code(d):
    checks = []
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 0:
                checks.append([r * d + c, r * d + c + 1,
                               (r + 1) * d + c, (r + 1) * d + c + 1])
    logical = [r * d for r in range(d)]
    return checks, d * d, logical

def build_H(check_to_qubits, n_qubits):
    H = np.zeros((len(check_to_qubits), n_qubits), dtype=np.uint8)
    for ci, qs in enumerate(check_to_qubits):
        for q in qs:
            H[ci, q] ^= 1
    return H

def syndrome_of(H, vec):
    return (H @ vec) & 1

def run_scaling_benchmark(d, shots=1000, p=0.06, seed=42):
    rng = np.random.default_rng(seed)
    check_to_qubits, n_qubits, logical = rotated_surface_code(d)
    H = build_H(check_to_qubits, n_qubits)
    n_checks = len(check_to_qubits)
    
    # Generate syndromes
    errors = (rng.random((shots, n_qubits)) < p).astype(np.uint8)
    syndromes = (errors @ H.T) & 1
    
    # 1. Profile QECTOR Blossom
    tracemalloc.start()
    t0 = time.perf_counter()
    qector = BlossomDecoder(check_to_qubits, n_qubits)
    q_init_time = time.perf_counter() - t0
    
    q_dec_times = []
    q_corrections = []
    q_valids = []
    
    for s in range(shots):
        syn = syndromes[s].astype(np.uint8)
        
        t_start = time.perf_counter()
        qc = np.asarray(qector.decode(syn)).astype(np.uint8)
        q_dec_times.append(time.perf_counter() - t_start)
        
        q_ok = np.array_equal(syndrome_of(H, qc), syn)
        q_corrections.append(qc)
        q_valids.append(q_ok)
        
    _, q_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 2. Profile PyMatching
    tracemalloc.start()
    t0 = time.perf_counter()
    pm = Matching.from_check_matrix(H)
    pm_init_time = time.perf_counter() - t0
    
    pm_dec_times = []
    weight_match = 0
    
    for s in range(shots):
        syn = syndromes[s].astype(np.uint8)
        qc = q_corrections[s]
        q_ok = q_valids[s]
        
        t_start = time.perf_counter()
        pc = pm.decode(syn).astype(np.uint8)
        pm_dec_times.append(time.perf_counter() - t_start)
        
        pm_ok = np.array_equal(syndrome_of(H, pc), syn)
        
        # Verify matching weight agreement (on uniform weights, weight = sum of flipped qubits)
        if q_ok and pm_ok and qc.sum() == pc.sum():
            weight_match += 1
            
    _, pm_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    avg_q_us = np.mean(q_dec_times) * 1e6
    avg_pm_us = np.mean(pm_dec_times) * 1e6
    ratio = avg_q_us / avg_pm_us
    
    print(f"d={d:2d} | Qubits={n_qubits:3d} | Checks={n_checks:3d} | "
          f"Qector={avg_q_us:6.2f} us (Init={q_init_time*1000:5.2f}ms, Mem={q_peak_mem/1024:6.1f}KB) | "
          f"PyMatching={avg_pm_us:6.2f} us (Init={pm_init_time*1000:5.2f}ms, Mem={pm_peak_mem/1024:6.1f}KB) | "
          f"Ratio={ratio:5.2f}x | Match={100*weight_match/shots:5.1f}%")
          
    return {
        "d": d, "qubits": n_qubits, "checks": n_checks,
        "q_avg_us": avg_q_us, "q_init_ms": q_init_time * 1000, "q_mem_kb": q_peak_mem / 1024,
        "pm_avg_us": avg_pm_us, "pm_init_ms": pm_init_time * 1000, "pm_mem_kb": pm_peak_mem / 1024,
        "ratio": ratio, "match_pct": 100 * weight_match / shots
    }

if __name__ == "__main__":
    print("=" * 120)
    print("RUNNING SCALING AND PROFILING BENCHMARK: QECTOR Blossom vs PyMatching (d=3..25)")
    print("=" * 120)
    
    distances = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    results = []
    for d in distances:
        shots = 1000 if d < 17 else 500
        res = run_scaling_benchmark(d, shots=shots)
        results.append(res)
        
    print("\nMarkdown Table for Report:\n")
    print("| Distance $d$ | Qubits | Checks | QECTOR Latency (µs) | PyMatching Latency (µs) | Latency Ratio | QECTOR Memory (KB) | PyMatching Memory (KB) | MWPM Agreement |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['d']} | {r['qubits']} | {r['checks']} | {r['q_avg_us']:.2f} µs | {r['pm_avg_us']:.2f} µs | {r['ratio']:.2f}× | {r['q_mem_kb']:.1f} KB | {r['pm_mem_kb']:.1f} KB | {r['match_pct']:.1f}% |")
    print("=" * 120)
