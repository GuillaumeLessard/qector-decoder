import time

import numpy as np
import qector_decoder_v3 as qd
import stim

print('=== CPU vs GPU Benchmark (Enterprise, dev.bat) ===')
print(f'GPU available: {qd.cuda_is_available()}')

configs = [
    ('d=5 surface', 5, 0.01, 5000),
    ('d=9 surface', 9, 0.005, 2000),
    ('d=13 surface', 13, 0.003, 1000),
]

for name, d, p, shots in configs:
    print(f'\n--- {name} (p={p}, shots={shots}) ---')
    circuit = stim.Circuit.generated(
        'surface_code:rotated_memory_x',
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    model = qd.dem.from_stim(dem).collapse_to_graph()
    c2q = model.check_to_qubits()
    nq = model.num_errors
    weights = model.weights().tolist()
    det, obs = circuit.compile_detector_sampler(seed=42).sample(shots, separate_observables=True)
    det = np.ascontiguousarray(det.astype(np.uint8))

    # CPU (FastUnionFind weighted)
    cpu = qd.FastUnionFindDecoder(c2q, nq, edge_weights=weights)
    t0 = time.perf_counter()
    cpu_corr = cpu.batch_decode(det)
    t_cpu = time.perf_counter() - t0
    cpu_pred = ((model.observables_matrix() @ cpu_corr.T) % 2).T
    cpu_ler = float(np.mean(cpu_pred[:, 0] != obs[:, 0]))

    # GPU (CUDABatchDecoder weighted f64)
    try:
        gpu = qd.CUDABatchDecoder(c2q, nq, weights, precision='f64')
        t0 = time.perf_counter()
        gpu_corr = gpu.batch_decode(det)
        t_gpu = time.perf_counter() - t0
        gpu_pred = ((model.observables_matrix() @ gpu_corr.T) % 2).T
        gpu_ler = float(np.mean(gpu_pred[:, 0] != obs[:, 0]))
        speedup = t_cpu / t_gpu
        print(f'  CPU: {t_cpu*1000:.1f} ms ({shots/t_cpu:.0f} fps) LER={cpu_ler:.5f}')
        print(f'  GPU: {t_gpu*1000:.1f} ms ({shots/t_gpu:.0f} fps) LER={gpu_ler:.5f}')
        print(f'  Speedup: {speedup:.2f}x')
    except Exception as e:
        print(f'  CPU: {t_cpu*1000:.1f} ms LER={cpu_ler:.5f}')
        print(f'  GPU: FAILED ({type(e).__name__}: {e})')
