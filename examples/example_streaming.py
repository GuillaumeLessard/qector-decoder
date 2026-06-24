#!/usr/bin/env python3
"""
example_streaming.py — QECTOR Decoder v3.4 Streaming Demo

Demonstrates real-time streaming decoder with sliding window.
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
    StreamingDecoder,
    SlidingWindowDecoder,
    generate_ring_code_checks,
)

def main():
    # Use distance 5 ring code
    checks, n_qubits = generate_ring_code_checks(5)
    n_checks = len(checks)
    
    # Streaming parameters
    window_size = 10  # number of rounds in sliding window
    n_rounds = 500    # total rounds to simulate
    
    rng = np.random.default_rng(42)
    
    print("=" * 60)
    print("QECTOR v3.4 — Streaming & Sliding Window Demo")
    print("=" * 60)
    print(f"Code: Ring Code d=5 ({n_qubits} qubits, {n_checks} checks)")
    print(f"Simulating {n_rounds} rounds of syndrome extraction...")
    
    # 1. Basic streaming decoder
    print("\n1. StreamingDecoder (accumulates syndromes over time):")
    stream = StreamingDecoder(checks, n_qubits, history_size=window_size)
    
    t0 = time.perf_counter()
    for round_idx in range(n_rounds):
        syndrome = rng.integers(0, 2, size=n_checks, dtype=np.uint8)
        # update() inserts the new round's syndrome and returns the corrected state
        result = stream.update(syndrome)
        if round_idx % 100 == 0:
            stream.flush()  # periodically flush history in a real environment
    t1 = time.perf_counter()
    total_time = (t1 - t0) * 1000
    print(f"   Rounds: {n_rounds}")
    print(f"   Total time: {total_time:.2f} ms")
    print(f"   Latency per round: {total_time / n_rounds * 1000:.2f} µs")
    print(f"   Throughput: {n_rounds / (total_time / 1000):.0f} rounds/s")
    
    # 2. Sliding window decoder
    print(f"\n2. SlidingWindowDecoder (sliding window with decay factor 0.85):")
    window = SlidingWindowDecoder(checks, n_qubits, window_size=window_size, decay_factor=0.85)
    
    t0 = time.perf_counter()
    for round_idx in range(n_rounds):
        syndrome = rng.integers(0, 2, size=n_checks, dtype=np.uint8)
        result = window.update(syndrome)
        if round_idx % 100 == 0:
            window.flush()
    t1 = time.perf_counter()
    total_time = (t1 - t0) * 1000
    print(f"   Rounds: {n_rounds}")
    print(f"   Total time: {total_time:.2f} ms")
    print(f"   Latency per round: {total_time / n_rounds * 1000:.2f} µs")
    print(f"   Throughput: {n_rounds / (total_time / 1000):.0f} rounds/s")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
