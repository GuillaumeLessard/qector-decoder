#!/usr/bin/env python3
"""
example_basic.py — QECTOR Decoder v3.4 Basic Usage

Demonstrates UnionFind, Blossom, SparseBlossom, BP-OSD, and Hybrid decoders.
"""

import sys

# Force UTF-8 stdout/stderr so emoji output doesn't crash on legacy consoles
# (e.g. Windows cp1252). No-op where reconfigure is unavailable.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from qector_decoder_v3 import (
    UnionFindDecoder,
    BlossomDecoder,
    SparseBlossomDecoder,
    BPOSDDecoder,
    HybridDecoder,
    generate_ring_code_checks,
)

def main():
    # Generate checks and number of qubits for a ring code of distance 5
    checks, n_qubits = generate_ring_code_checks(5)
    n_checks = len(checks)
    
    # Create syndrome with active checks at indices 2 and 7
    syndrome = np.zeros(n_checks, dtype=np.uint8)
    syndrome[2] = 1
    syndrome[7] = 1
    
    print("=" * 60)
    print("QECTOR v3.4 — Basic Decoder Demo")
    print("=" * 60)
    print(f"Code: Ring Code d=5 ({n_qubits} qubits, {n_checks} checks)")
    print(f"Syndrome: {syndrome}")
    
    # 1. UnionFind (fastest, approximate)
    uf = UnionFindDecoder(checks, n_qubits)
    result_uf = uf.decode(syndrome)
    print("\n1. UnionFindDecoder:")
    print(f"   Correction: {result_uf}")
    
    # 2. Blossom (exact MWPM)
    blossom = BlossomDecoder(checks, n_qubits)
    result_blossom = blossom.decode(syndrome)
    print("\n2. BlossomDecoder:")
    print(f"   Correction: {result_blossom}")
    
    # 3. SparseBlossom (region-growing, approximate)
    sparse = SparseBlossomDecoder(checks, n_qubits)
    result_sparse = sparse.decode(syndrome)
    print("\n3. SparseBlossomDecoder:")
    print(f"   Correction: {result_sparse}")
    
    # 4. BP-OSD (best LER, slower)
    bposd = BPOSDDecoder(checks, n_qubits)
    result_bposd = bposd.decode(syndrome)
    print("\n4. BPOSDDecoder:")
    print(f"   Correction: {result_bposd}")
    
    # 5. Hybrid Decoder (GNN + SparseBlossom)
    hybrid = HybridDecoder(checks, n_qubits)
    result_hybrid = hybrid.decode_hybrid(syndrome)
    print("\n5. HybridDecoder (GNN + SparseBlossom):")
    print(f"   Correction: {result_hybrid}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
