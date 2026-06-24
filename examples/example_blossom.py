#!/usr/bin/env python3
"""
example_blossom.py — QECTOR Decoder v3.4 Blossom & Sparse Blossom Demo

Demonstrates exact MWPM vs sparse region-growing Blossom on challenging cases.
"""

import sys

# Force UTF-8 stdout/stderr so emoji output doesn't crash on legacy consoles
# (e.g. Windows cp1252). No-op where reconfigure is unavailable.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import numpy as np
from qector_decoder_v3 import (
    BlossomDecoder,
    SparseBlossomDecoder,
    generate_ring_code_checks,
)

def main():
    # Use distance 5 ring code
    checks, n_qubits = generate_ring_code_checks(5)
    n_checks = len(checks)
    
    rng = np.random.default_rng(42)
    
    # Test cases with increasing difficulty
    test_cases = [
        ("Empty syndrome", np.zeros(n_checks, dtype=np.uint8)),
        ("Single error", np.array([0, 0, 1] + [0] * (n_checks - 3), dtype=np.uint8)),
        ("Two errors", np.array([0, 0, 1, 0, 0, 0, 0, 1] + [0] * (n_checks - 8), dtype=np.uint8)),
        ("Four errors", np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] + [0] * (n_checks - 18), dtype=np.uint8)),
    ]
    
    # Add random syndrome
    random_syndrome = rng.integers(0, 2, size=n_checks, dtype=np.uint8)
    test_cases.append(("Random syndrome", random_syndrome))
    
    print("=" * 60)
    print("QECTOR v3.4 — Blossom vs Sparse Blossom Demo")
    print("=" * 60)
    print(f"Code: Ring Code d=5 ({n_qubits} qubits, {n_checks} checks)")
    
    blossom = BlossomDecoder(checks, n_qubits)
    sparse = SparseBlossomDecoder(checks, n_qubits)
    
    for name, syndrome in test_cases:
        print(f"\n{name}:")
        print(f"   Syndrome: {syndrome}")
        
        t0 = time.perf_counter()
        result_blossom = blossom.decode(syndrome)
        t1 = time.perf_counter()
        time_blossom = (t1 - t0) * 1e6
        
        t0 = time.perf_counter()
        result_sparse = sparse.decode(syndrome)
        t1 = time.perf_counter()
        time_sparse = (t1 - t0) * 1e6
        
        match = np.array_equal(result_blossom, result_sparse)
        print(f"   Blossom:   {result_blossom} ({time_blossom:.1f} µs)")
        print(f"   Sparse:      {result_sparse} ({time_sparse:.1f} µs)")
        print(f"   Match:       {'yes' if match else 'no'}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
