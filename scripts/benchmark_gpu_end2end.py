#!/usr/bin/env python3
"""End-to-end native CUDA/OpenCL benchmark with resilience monitoring.

Legacy end-to-end GPU benchmark. The canonical reproducible harness is
``scripts/run_competitive_benchmark.py`` + ``qector_decoder_v3.benchmarking``
(hot/cold split, tail-latency percentiles, environment capture); see
``docs/METHODOLOGY.md``. Automatic backend selection (CPU/Rayon/CUDA/OpenCL with
calibration) is available via ``qector_decoder_v3.backend.AutoDecoder``.
"""

import time

import numpy as np
import qector_decoder_v3 as qec


def build_surface_code_checks(distance):
    """Build check-to-qubits for a compact surface-code layout."""
    checks = []
    for row in range(distance - 1):
        for col in range(distance - 1):
            checks.append(
                [
                    row * distance + col,
                    row * distance + col + 1,
                    (row + 1) * distance + col,
                    (row + 1) * distance + col + 1,
                ]
            )
    for row in range(1, distance):
        for col in range(1, distance):
            checks.append(
                [
                    (row - 1) * distance + col - 1,
                    (row - 1) * distance + col,
                    row * distance + col - 1,
                    row * distance + col,
                ]
            )
    return checks


def available_backends():
    backends = []
    if qec.CUDABatchDecoder.is_available():
        backends.append(("CUDA", qec.CUDABatchDecoder))
    if qec.OpenCLBatchDecoder.is_available():
        backends.append(("OpenCL", qec.OpenCLBatchDecoder))
    return backends


def measure_e2e_latency(decoder, syndromes, warmup=5, repeats=20):
    """Measure transfer, kernel, synchronization, and return latency."""
    for _ in range(warmup):
        decoder.batch_decode(syndromes)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        decoder.batch_decode(syndromes)
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2], times[min(len(times) - 1, int(len(times) * 0.99))]


def main():
    print("=" * 70)
    print("QECTOR v3 GPU CUDA/OpenCL End-to-End Benchmark")
    print("=" * 70)
    backends = available_backends()
    print(f"Native CUDA available: {qec.CUDABatchDecoder.is_available()}")
    print(f"GPU OpenCL available: {qec.OpenCLBatchDecoder.is_available()}")
    print()

    print("--- Section 1: Single-shot decode latency ---")
    checks = build_surface_code_checks(5)
    n_qubits = 25
    n_checks = len(checks)
    cpu = qec.UnionFindDecoder(checks, n_qubits)
    cpu_times = []
    for _ in range(1000):
        syndrome = np.random.randint(0, 2, size=n_checks, dtype=np.uint8)
        start = time.perf_counter()
        cpu.decode(syndrome)
        cpu_times.append(time.perf_counter() - start)
    cpu_times.sort()
    print(
        f"  CPU decode:       p50={cpu_times[500]*1e6:6.2f}us "
        f"p99={cpu_times[990]*1e6:6.2f}us"
    )
    for name, decoder_cls in backends:
        decoder = decoder_cls(checks, n_qubits)
        timings = []
        for _ in range(1000):
            syndrome = np.random.randint(0, 2, size=(1, n_checks), dtype=np.uint8)
            start = time.perf_counter()
            decoder.batch_decode(syndrome)
            timings.append(time.perf_counter() - start)
        timings.sort()
        print(
            f"  {name:6s} batch=1: p50={timings[500]*1e6:6.2f}us "
            f"p99={timings[990]*1e6:6.2f}us"
        )
    print()

    print("--- Section 2: End-to-end batch latency ---")
    for distance in [5, 9]:
        checks = build_surface_code_checks(distance)
        n_qubits = distance * distance
        n_checks = len(checks)
        for name, decoder_cls in backends:
            decoder = decoder_cls(checks, n_qubits)
            for batch_size in [1024, 4096]:
                syndromes = np.random.randint(
                    0, 2, size=(batch_size, n_checks), dtype=np.uint8
                )
                p50, p99 = measure_e2e_latency(decoder, syndromes)
                print(
                    f"  {name:6s} d={distance} bs={batch_size:4d} | "
                    f"p50={p50*1000:6.2f}ms p99={p99*1000:6.2f}ms "
                    f"throughput={batch_size/p50:9.0f} dec/s"
                )
    print()

    print("--- Section 3: Resilience monitoring (1000 calls) ---")
    checks = build_surface_code_checks(5)
    n_qubits = 25
    n_checks = len(checks)
    for name, decoder_cls in backends:
        decoder = decoder_cls(checks, n_qubits)
        for batch_size in [64, 1024, 4096]:
            syndromes = np.random.randint(
                0, 2, size=(batch_size, n_checks), dtype=np.uint8
            )
            for _ in range(1000):
                decoder.batch_decode(syndromes)
            print(
                f"  {name:6s} bs={batch_size:4d} | "
                f"consecutive_failures={decoder.consecutive_failures} "
                f"total_failures={decoder.total_failures} "
                f"degraded={decoder.is_degraded}"
            )
    print()

    print("--- Section 4: Kernel architecture ---")
    print("  CUDA: NVRTC-compiled native kernel, Driver API, reusable device workspace")
    print("  OpenCL: local/global-memory kernels selected by code size")
    print("  Both: one syndrome per work item/thread and transparent CPU fallback")
    print("  Both: degraded mode after 3 failures with periodic recovery attempts")
    print()
    print("=" * 70)
    print("End-to-end benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
