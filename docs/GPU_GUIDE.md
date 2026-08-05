# QECTOR Decoder v3 — GPU Acceleration Guide

QECTOR provides GPU acceleration via CUDA (`CUDABatchDecoder`) and CuPy (`gpu_backend`).

---

## CPU vs GPU Crossover Surface

Launching a GPU kernel incurs fixed host-to-device memory copy and kernel launch overhead. For small batch sizes ($N < \text{crossover}$), multi-core CPU decoding (`CPUBatchDecoder` / `BatchDecoder`) is faster. Above the crossover, GPU batching achieves millions of shots per second.

| Code Distance ($d$) | Data Qubits ($n$) | Measured GPU Crossover ($N_\text{shots}$) |
|---|---|---|
| $d=3$ | 5 | 8,000 |
| $d=5$ | 25 | 40,000 |
| $d=9$ | 81 | 150,000 |
| $d=13$ | 169 | 450,000 |

---

## Weighted vs Unweighted GPU Decoding

> [!WARNING]
> **Always pass the DEM per-mechanism weights $\log((1-p)/p)$ to the GPU for production use.**
>
> Without `edge_weights`, `CUDABatchDecoder` falls back to unweighted cluster growth. On circuit-level noise, unweighted decoding costs ~3x higher logical error rate ($0.059$ vs $0.012$ at $d=5$).

### Example: Weighted GPU Batch Decoding

```python
from qector_decoder_v3 import dem, CUDABatchDecoder

# Parse Stim Detector Error Model
model = dem.from_stim(circuit.detector_error_model(decompose_errors=True))
if model.is_graphlike:
    model = model.collapse_to_graph()

# Pass check-to-qubits and edge weights
c2q = model.check_to_qubits()
weights = model.weights().tolist()

if CUDABatchDecoder.is_available():
    # Weights are attached at construction; batch_decode then runs weighted growth.
    decoder = CUDABatchDecoder(c2q, n_qubits=model.num_errors, edge_weights=weights)
    corrections = decoder.batch_decode(syndromes)
```

---

## Growth-Accumulator Precision: f32 vs f64

The weighted kernel's default `precision="f32"` reproduces the historical build
bit-for-bit. `precision="f64"` runs `uf_decode_batch_cuda_f64`: identical control
flow, but the adaptive time-step growth accumulates in double precision against
the exact (pre-cast) f64 edge lengths — the fusion schedule stays faithful when
DEM weights span orders of magnitude (circuit-level noise).

```python
decoder = CUDABatchDecoder(c2q, n_qubits=model.num_errors, edge_weights=weights, precision="f64")
print(decoder.precision)  # "f64"
```

Costs: 2x the support-scratch VRAM and the device's FP64 throughput (consumer
cards are FP64-limited). Measured acceptance: GPU-f64 weighted LER matches the
weighted CPU LER within sampling noise at d=13 (see
`python/tests/test_cuda_f64_precision.py`). On an unweighted graph the flag is a
no-op (the unweighted branch is integer).
