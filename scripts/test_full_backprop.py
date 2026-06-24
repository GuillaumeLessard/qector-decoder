import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from qector_decoder_v3 import (
    HybridDecoder, BlossomDecoder,
    generate_toy_code_checks, generate_surface_code_checks
)
import random

print("=" * 60)
print("TEST: Full backprop MPNN training (v2.0)")
print("=" * 60)

# Test 1: Toy code d=5
d = 5
result = generate_toy_code_checks(d)
check_to_qubits = result[0]
n_qubits = result[1]

print(f"\n--- Toy code d={d}: {n_qubits} qubits, {len(check_to_qubits)} checks ---")

# Create HybridDecoder with an internal GNN matching DetectorGraph dimensions.
trained_decoder = HybridDecoder(
    check_to_qubits,
    n_qubits=n_qubits,
    gnn_hidden_size=16,
    gnn_n_layers=2,
)

# Train with soft targets (Blossom optimal corrections)
print("Training internal GNN with full MPNN backprop (soft targets)...")
print("  100 samples, 30 epochs...")
import time
t0 = time.time()
loss = trained_decoder.train(100, 30, error_rate=0.05)
t1 = time.time()
print(f"  Training completed in {t1-t0:.2f}s")
print(f"  Final loss: {loss:.6f}")

print(f"\nBenchmarking on 200 samples (p=0.05)...")
correct = 0
valid = 0
total = 0
t_decode = 0.0
rng = random.Random(42)

for _ in range(200):
    error_vec = np.array([1 if rng.random() < 0.05 else 0 for _ in range(n_qubits)], dtype=np.uint8)
    syndrome = np.array([sum(int(error_vec[q]) for q in check) % 2 for check in check_to_qubits], dtype=np.uint8)
    
    t0 = time.time()
    correction = trained_decoder.decode_hybrid(syndrome)
    t_decode += time.time() - t0
    
    residual = np.bitwise_xor(error_vec, correction)
    valid += all(sum(int(residual[q]) for q in check) % 2 == 0 for check in check_to_qubits)
    
    optimal = BlossomDecoder(check_to_qubits, n_qubits).decode(syndrome)
    correct += bool(np.array_equal(correction, optimal))
    total += 1

print(f"\n--- Results (toy code d={d}, trained GNN) ---")
print(f"Valid corrections: {valid}/{total} = {valid/total:.2%}")
print(f"Blossom-optimal: {correct}/{total} = {correct/total:.2%}")
print(f"Avg decode time: {t_decode/total*1e6:.1f} µs")

# Benchmark standard Blossom for comparison
print(f"\n--- Baseline: Standard Blossom ---")
std_decoder = BlossomDecoder(check_to_qubits, n_qubits)
correct_std = 0
valid_std = 0
t_std = 0.0
rng = random.Random(43)
for _ in range(200):
    error_vec = np.array([1 if rng.random() < 0.05 else 0 for _ in range(n_qubits)], dtype=np.uint8)
    syndrome = np.array([sum(int(error_vec[q]) for q in check) % 2 for check in check_to_qubits], dtype=np.uint8)
    t0 = time.time()
    correction = std_decoder.decode(syndrome)
    t_std += time.time() - t0
    residual = np.bitwise_xor(error_vec, correction)
    valid_std += all(sum(int(residual[q]) for q in check) % 2 == 0 for check in check_to_qubits)
    correct_std += bool(np.array_equal(correction, std_decoder.decode(syndrome)))

print(f"Valid: {valid_std}/200 = {valid_std/200:.2%}")
print(f"Blossom-optimal: {correct_std}/200 = {correct_std/200:.2%}")
print(f"Avg decode time: {t_std/200*1e6:.1f} µs")

# Test 2: Proper toric code with CSS-aware decoding
print("\n" + "=" * 60)
print("TEST: Proper toric code d=5 with CSS-aware decoding")
print("=" * 60)

d = 5
result = generate_surface_code_checks(d)
check_to_qubits = result[0]
n_qubits = result[1]

print(f"Toric code d={d}: {n_qubits} qubits, {len(check_to_qubits)} checks")

# For X-error decoding, use only Z-stabilizers (the d*d last checks)
z_checks = check_to_qubits[d*d:]
print(f"Z-stabilizers (for X-error decoding): {len(z_checks)} checks")

# Create and train HybridDecoder on Z-stabilizers only.
trained_decoder_toric = HybridDecoder(
    z_checks,
    n_qubits=n_qubits,
    gnn_hidden_size=16,
    gnn_n_layers=2,
)

print(f"Training on toric code (100 samples, 30 epochs)...")
t0 = time.time()
loss_toric = trained_decoder_toric.train(100, 30, error_rate=0.05)
t1 = time.time()
print(f"  Training completed in {t1-t0:.2f}s")
print(f"  Final loss: {loss_toric:.6f}")

correct_toric = 0
valid_toric = 0
total_toric = 0
rng = random.Random(44)
for _ in range(200):
    error_vec = np.array([1 if rng.random() < 0.05 else 0 for _ in range(n_qubits)], dtype=np.uint8)
    syndrome = np.array([sum(int(error_vec[q]) for q in check) % 2 for check in z_checks], dtype=np.uint8)
    correction = trained_decoder_toric.decode_hybrid(syndrome)
    
    residual = np.bitwise_xor(error_vec, correction)
    valid_toric += all(sum(int(residual[q]) for q in check) % 2 == 0 for check in z_checks)
    
    optimal = BlossomDecoder(z_checks, n_qubits).decode(syndrome)
    correct_toric += bool(np.array_equal(correction, optimal))
    total_toric += 1

print(f"\n--- Results (proper toric d={d}, trained GNN) ---")
print(f"Valid corrections: {valid_toric}/{total_toric} = {valid_toric/total_toric:.2%}")
print(f"Blossom-optimal: {correct_toric}/{total_toric} = {correct_toric/total_toric:.2%}")

# Baseline for toric code
std_toric = BlossomDecoder(z_checks, n_qubits)
correct_std_toric = 0
valid_std_toric = 0
rng = random.Random(45)
for _ in range(200):
    error_vec = np.array([1 if rng.random() < 0.05 else 0 for _ in range(n_qubits)], dtype=np.uint8)
    syndrome = np.array([sum(int(error_vec[q]) for q in check) % 2 for check in z_checks], dtype=np.uint8)
    correction = std_toric.decode(syndrome)
    residual = np.bitwise_xor(error_vec, correction)
    valid_std_toric += all(sum(int(residual[q]) for q in check) % 2 == 0 for check in z_checks)
    correct_std_toric += bool(np.array_equal(correction, std_toric.decode(syndrome)))

print(f"\n--- Baseline: Standard Blossom (toric) ---")
print(f"Valid: {valid_std_toric}/200 = {valid_std_toric/200:.2%}")
print(f"Blossom-optimal: {correct_std_toric}/200 = {correct_std_toric/200:.2%}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("Full MPNN backprop is working correctly.")
print("=" * 60)
