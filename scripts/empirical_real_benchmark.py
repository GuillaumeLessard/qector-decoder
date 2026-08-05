"""
Real Empirical Benchmark & LER Verification Harness for QECTOR v0.9.0 vs PyMatching.

Generates real Stim quantum error correction detector error models (DEMs) across
surface and repetition codes, samples real detector event shots, decodes with
QECTOR decoders, and outputs real empirical LER and throughput metrics.
"""

import time

import numpy as np
import pymatching
import qector_decoder_v3 as qd
import stim
from qector_decoder_v3 import FastUnionFindDecoder


def run_empirical_benchmark():
    print("==========================================================================")
    print("         EMPIRICAL QECTOR v0.9.0 vs PYMATCHING REAL BENCHMARK             ")
    print("==========================================================================")
    
    # Real Stim circuits: surface code and repetition code across distances
    configs = [
        ("surface_code:rotated_memory_z", 3, 0.005, 10000),
        ("surface_code:rotated_memory_z", 5, 0.005, 10000),
        ("surface_code:rotated_memory_z", 7, 0.005, 10000),
        ("repetition_code:memory", 5, 0.01, 20000),
        ("repetition_code:memory", 7, 0.01, 20000),
    ]
    
    results = []
    
    for circuit_type, distance, noise_p, num_shots in configs:
        if "surface" in circuit_type:
            circuit = stim.Circuit.generated(
                circuit_type,
                distance=distance,
                rounds=distance,
                after_clifford_depolarization=noise_p,
                before_measure_flip_probability=noise_p,
                after_reset_flip_probability=noise_p,
            )
        else:
            circuit = stim.Circuit.generated(
                circuit_type,
                distance=distance,
                rounds=distance,
                after_clifford_depolarization=noise_p,
            )
            
        dem = circuit.detector_error_model(decompose_errors=True)
        model = qd.dem.from_stim(dem)
        
        # Sample real detector shots and observable flips from Stim
        sampler = circuit.compile_detector_sampler()
        detectors, observables = sampler.sample(num_shots, separate_observables=True)
        detectors = np.ascontiguousarray(detectors, dtype=np.uint8)
        observables = np.ascontiguousarray(observables, dtype=np.uint8)
        
        # 1. PyMatching 2.x baseline
        pm = pymatching.Matching.from_detector_error_model(dem)
        t0 = time.perf_counter()
        pm_predictions = pm.decode_batch(detectors)
        t_pm = time.perf_counter() - t0
        pm_ler = np.mean(pm_predictions[:, 0] != observables[:, 0])
        pm_throughput = num_shots / t_pm
        
        # 2. QECTOR Blossom (MWPM) via PyMatching-compatible Matching
        q_blossom = qd.pymatching_compat.Matching.from_detector_error_model(dem)
        t0 = time.perf_counter()
        blossom_predictions = q_blossom.decode_batch(detectors)
        t_blossom = time.perf_counter() - t0
        blossom_ler = np.mean(blossom_predictions[:, 0] != observables[:, 0])
        blossom_throughput = num_shots / t_blossom
        
        # 3. QECTOR Fast Union-Find
        c2q = model.check_to_qubits()
        nq = model.num_errors
        weights = model.weights().tolist()
        uf = FastUnionFindDecoder(c2q, n_qubits=nq, edge_weights=weights)
        L = model.observables_matrix()
        
        t0 = time.perf_counter()
        uf_corrections = uf.batch_decode(detectors)
        t_uf = time.perf_counter() - t0
        
        # Batch predict observables L @ c.T % 2
        uf_predictions = ((L @ uf_corrections.T) % 2).T
        uf_ler = np.mean(uf_predictions[:, 0] != observables[:, 0])
        uf_throughput = num_shots / t_uf
        
        print(f"\n--- Code: {circuit_type} (d={distance}, p={noise_p}, shots={num_shots}) ---")
        print(f"PyMatching 2.x   : LER = {pm_ler:.5f} | Throughput = {pm_throughput:10.1f} shots/sec")
        print(f"QECTOR Blossom   : LER = {blossom_ler:.5f} | Throughput = {blossom_throughput:10.1f} shots/sec")
        print(f"QECTOR Fast UF   : LER = {uf_ler:.5f} | Throughput = {uf_throughput:10.1f} shots/sec")
        
        results.append({
            "code": f"{circuit_type} d={distance}",
            "shots": num_shots,
            "pm_ler": pm_ler,
            "pm_fps": pm_throughput,
            "q_blossom_ler": blossom_ler,
            "q_blossom_fps": blossom_throughput,
            "q_uf_ler": uf_ler,
            "q_uf_fps": uf_throughput,
        })
        
    print("\n==========================================================================")
    print("                     REAL EMPIRICAL SUMMARY TABLE                         ")
    print("==========================================================================")
    print(f"{'Code Topology':<30} | {'PyMatching LER':<15} | {'QECTOR Blossom LER':<18} | {'QECTOR UF LER':<15}")
    print("-" * 88)
    for r in results:
        print(f"{r['code']:<30} | {r['pm_ler']:<15.5f} | {r['q_blossom_ler']:<18.5f} | {r['q_uf_ler']:<15.5f}")
    print("==========================================================================")


if __name__ == "__main__":
    run_empirical_benchmark()
