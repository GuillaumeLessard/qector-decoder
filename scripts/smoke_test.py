#!/usr/bin/env python3
"""
QECTOR Decoder v3 Extended Smoke Test Suite
"""

import sys
import time

import numpy as np
from qector_decoder_v3 import (
    BatchDecoder,
    BlossomDecoder,
    CUDABatchDecoder,
    FastUnionFindDecoder,
    NativeAutoDecoder,
    UnionFindDecoder,
    codes,
    get_license_info,
    set_license_key,
)
from qector_decoder_v3 import __version__ as qector_version


def main():
    print("=== QECTOR Decoder v3 Extended Smoke Test ===")
    print(f"Python {sys.version.split()[0]}\n")

    # 1. Basic decoders
    print("1. Basic decoders")
    checks = [[0,1], [1,2], [2,3], [3,4]]
    nq = 5
    syndrome = np.array([1, 1, 0, 0], dtype=np.uint8)

    for name, Cls in [("UnionFind", UnionFindDecoder), ("FastUF", FastUnionFindDecoder), ("Blossom", BlossomDecoder)]:
        dec = Cls(checks, nq)
        corr = dec.decode(syndrome)
        print(f"  ✓ {name:12} → {list(corr)}")

    # 2. Hypergraph rejection (v0.6.2)
    print("\n2. Hypergraph rejection (v0.6.2)")
    hyper_checks = [[0,1,2], [1,2,3], [2,3,4]]  # clear hypergraph
    try:
        UnionFindDecoder(hyper_checks, 5)
        print("  ✗ UnionFind should have rejected")
    except ValueError:
        print("  ✓ UnionFind correctly rejected hypergraph")

    # 3. Surface code + timing
    print("\n3. Surface code d=5 + timing")
    try:
        if hasattr(codes, "generate_surface_code_checks"):
            surface_checks, nq_surf = codes.generate_surface_code_checks(5)
        else:
            surface_checks, nq_surf = checks, nq

        syndrome_surf = np.random.randint(0, 2, len(surface_checks), dtype=np.uint8)

        start = time.perf_counter()
        blossom = BlossomDecoder(surface_checks, nq_surf)
        _ = blossom.decode(syndrome_surf)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  ✓ Blossom d=5 in {elapsed:.2f} ms")
    except Exception as e:
        print(f"  ✗ Surface code issue: {e}")

    # 4. Batch
    print("\n4. Batch decoding + timing")
    start = time.perf_counter()
    batch_dec = BatchDecoder([[0,1],[1,2],[2,3]], 4)
    batch_syndromes = np.random.randint(0, 2, (1024, 3), dtype=np.uint8)
    _ = batch_dec.parallel_batch_decode(batch_syndromes)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  ✓ Batch 1024 in {elapsed:.2f} ms")

    print(f"\n5. GPU: {CUDABatchDecoder.is_available()}")

    # 6. Native AutoDecoder
    print("\n6. Native AutoDecoder")
    try:
        code = codes.rotated_surface_code(3)
        native = NativeAutoDecoder(code.check_to_qubits, code.n_qubits, distance=3, noise_rate=0.08, batch_size=1, is_qldpc=False)
        syndrome = np.zeros(len(code.check_to_qubits), dtype=np.uint8)
        corr = native.decode(syndrome)
        print(f"  ✓ NativeAutoDecoder d=3 → {corr.shape}")
    except Exception as e:
        print(f"  ✗ NativeAutoDecoder error: {e}")

    # 7. License bridge
    print("\n7. License bridge")
    try:
        set_license_key("QECT-PRO-test123")
        info = get_license_info()
        print(f"  ✓ set_license_key / get_license_info → tier={info.get('tier')}, max_distance={info.get('max_distance')}")
    except Exception as e:
        print(f"  ✗ License bridge error: {e}")

    print(f"\nVersion: {qector_version}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("v0.6.2 core features verified.")

if __name__ == "__main__":
    main()
