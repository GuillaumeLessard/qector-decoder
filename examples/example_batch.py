#!/usr/bin/env python3
"""
example_batch.py — QECTOR Decoder v3.4 Batch & GPU Demo

Demonstrates CPU and GPU parallel batch decoding (OpenCL & CUDA if available).
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
    CPUBatchDecoder,
    OpenCLBatchDecoder,
    CUDABatchDecoder,
    cuda_is_available,
    opencl_is_available,
    generate_surface_code_checks,
)

def main():
    # Use distance 5 surface code
    checks, n_qubits = generate_surface_code_checks(5)
    n_checks = len(checks)
    batch_size = 10000  # larger batch size to show GPU speedup
    
    # Generate random syndromes using NumPy
    rng = np.random.default_rng(42)
    syndromes = rng.integers(0, 2, size=(batch_size, n_checks), dtype=np.uint8)
    
    print("=" * 60)
    print("QECTOR v3.4 — Batch & GPU Demo")
    print("=" * 60)
    print(f"Code: Surface Code d=5 ({n_qubits} qubits, {n_checks} checks)")
    print(f"Batch size: {batch_size}")
    
    # 1. CPU Batch decoder
    print(f"\n1. CPUBatchDecoder (CPU parallel execution):")
    cpu_batch = CPUBatchDecoder(checks, n_qubits)
    # Warm up
    _ = cpu_batch.batch_decode(syndromes[:10])
    
    t0 = time.perf_counter()
    results_cpu = cpu_batch.batch_decode(syndromes)
    t1 = time.perf_counter()
    cpu_time = (t1 - t0) * 1000
    print(f"   Time: {cpu_time:.2f} ms")
    print(f"   Throughput: {batch_size / (cpu_time / 1000):.0f} dec/s")
    
    # 2. GPU OpenCL Batch decoder (if available)
    cl_avail = opencl_is_available()
    print(f"\n2. OpenCLBatchDecoder (GPU OpenCL):")
    print(f"   OpenCL available: {cl_avail}")
    
    if cl_avail:
        gpu_cl = OpenCLBatchDecoder(checks, n_qubits)
        # Warm up
        _ = gpu_cl.batch_decode(syndromes[:10])
        
        t0 = time.perf_counter()
        results_gpu_cl = gpu_cl.batch_decode(syndromes)
        t1 = time.perf_counter()
        cl_time = (t1 - t0) * 1000
        print(f"   Time: {cl_time:.2f} ms")
        print(f"   Throughput: {batch_size / (cl_time / 1000):.0f} dec/s")
        print(f"   Speedup vs CPU: {cpu_time / cl_time:.2f}x")
        
        # Verify correctness
        matches = np.array_equal(results_cpu, results_gpu_cl)
        print(f"   Output matches CPU: {'yes' if matches else 'no'}")
    else:
        print("   (Skipped — OpenCL GPU not available)")
        
    # 3. GPU CUDA Batch decoder (if available)
    cuda_avail = cuda_is_available()
    print(f"\n3. CUDABatchDecoder (GPU Native CUDA):")
    print(f"   CUDA available: {cuda_avail}")
    
    if cuda_avail and CUDABatchDecoder is not None:
        gpu_cuda = CUDABatchDecoder(checks, n_qubits)
        # Warm up
        _ = gpu_cuda.batch_decode(syndromes[:10])
        
        t0 = time.perf_counter()
        results_gpu_cuda = gpu_cuda.batch_decode(syndromes)
        t1 = time.perf_counter()
        cuda_time = (t1 - t0) * 1000
        print(f"   Device Name: {gpu_cuda.device_name}")
        print(f"   Compute Capability: {gpu_cuda.compute_capability}")
        print(f"   Time: {cuda_time:.2f} ms")
        print(f"   Throughput: {batch_size / (cuda_time / 1000):.0f} dec/s")
        print(f"   Speedup vs CPU: {cpu_time / cuda_time:.2f}x")
        
        # Verify correctness
        matches = np.array_equal(results_cpu, results_gpu_cuda)
        print(f"   Output matches CPU: {'yes' if matches else 'no'}")
    else:
        print("   (Skipped — CUDA not available or not built)")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
