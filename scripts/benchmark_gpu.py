#!/usr/bin/env python3
"""Benchmark native CUDA and OpenCL GPU decoders against the CPU decoder.

Legacy GPU micro-benchmark (raw batch latency, ad-hoc timing). For the
reproducible, statistically-reported, environment-stamped harness — including
hot/cold split, p50/p90/p95/p99 and PyMatching cross-checks — use
``scripts/run_competitive_benchmark.py`` and ``qector_decoder_v3.benchmarking``
(see ``docs/METHODOLOGY.md`` and ``docs/REPRODUCE.md``). For circuit-level
logical-error-rate head-to-head vs PyMatching, use
``scripts/competitive_stim_ler.py``.
"""
import time
import numpy as np
import qector_decoder_v3 as qec


def build_surface_code_checks(d):
    """Build check-to-qubits for d×d surface code."""
    checks = []
    # X-checks (plaquettes) + Z-checks (stars)
    for r in range(d - 1):
        for c in range(d - 1):
            q0 = r * d + c
            q1 = r * d + (c + 1)
            q2 = (r + 1) * d + c
            q3 = (r + 1) * d + (c + 1)
            checks.append([q0, q1, q2, q3])
    for r in range(1, d):
        for c in range(1, d):
            q0 = (r - 1) * d + (c - 1)
            q1 = (r - 1) * d + c
            q2 = r * d + (c - 1)
            q3 = r * d + c
            checks.append([q0, q1, q2, q3])
    return checks


def benchmark_decoder(name, decoder, syndromes, warmup=3, repeats=10):
    """Return median steady-state latency and the final corrections."""
    for _ in range(warmup):
        decoder.batch_decode(syndromes)
    times = []
    corrections = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        corrections = decoder.batch_decode(syndromes)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), corrections


def main():
    print("=" * 60)
    print("QECTOR v3 GPU Benchmark: Native CUDA / OpenCL / CPU")
    print("=" * 60)

    # GPU availability
    opencl_available = qec.OpenCLBatchDecoder.is_available()
    cuda_available = qec.CUDABatchDecoder.is_available()
    print(f"GPU OpenCL available: {opencl_available}")
    print(f"Native CUDA available: {cuda_available}")
    print()

    distances = [3, 5, 7, 9]
    batch_sizes = [1, 8, 64, 256, 1024, 4096]

    for d in distances:
        checks = build_surface_code_checks(d)
        n_qubits = d * d
        n_checks = len(checks)
        print(f"--- Surface code d={d}, n_qubits={n_qubits}, n_checks={n_checks} ---")

        # CPU decoder
        cpu_dec = qec.CPUBatchDecoder(checks, n_qubits)

        # GPU decoder
        opencl_dec = (
            qec.OpenCLBatchDecoder(checks, n_qubits) if opencl_available else None
        )
        cuda_dec = qec.CUDABatchDecoder(checks, n_qubits) if cuda_available else None
        if cuda_dec is not None:
            print(
                f"  CUDA device: {cuda_dec.device_name} "
                f"(compute {cuda_dec.compute_capability[0]}.{cuda_dec.compute_capability[1]})"
            )

        for bs in batch_sizes:
            # Generate random syndromes
            syndromes = np.random.randint(0, 2, size=(bs, n_checks), dtype=np.uint8)

            # CPU benchmark
            cpu_t, cpu_corr = benchmark_decoder("cpu", cpu_dec, syndromes)
            cpu_tp = bs / cpu_t

            print(
                f"  bs={bs:4d} | CPU    {cpu_t*1000:7.2f}ms "
                f"({cpu_tp:10.1f} dec/s)"
            )
            for label, decoder in (("OpenCL", opencl_dec), ("CUDA", cuda_dec)):
                if decoder is None:
                    print(f"          | {label:7s} N/A")
                    continue
                gpu_t, gpu_corr = benchmark_decoder(label.lower(), decoder, syndromes)
                print(
                    f"          | {label:7s} {gpu_t*1000:7.2f}ms "
                    f"({bs/gpu_t:10.1f} dec/s) | speedup={cpu_t/gpu_t:.2f}x "
                    f"| match={np.array_equal(cpu_corr, gpu_corr)}"
                )
        print()

    print("=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
