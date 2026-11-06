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

## Growth-Accumulator Precision and CPU Identity

The weighted kernel accumulates cluster growth in f32 against the same f32 edge
lengths as the CPU `UnionFindDecoder` (`uf_core::grow_weighted`), and since
1.0.1 it also peels the same minimum-weight spanning forest (Kruskal over the
grown edges, cheapest first, ties by edge id). Weighted GPU corrections are
therefore bit-identical to the weighted CPU decoder, not merely equal in LER;
1.0.0's kernel peeled every grown edge and differed on a few percent of shots.
`python/tests/test_gpu_kernel_sources.py` checks this on any machine with a C++
compiler, by compiling the kernel source for the host.

An f64 accumulation mode (`precision="f64"`) was described in an earlier draft
of this guide but is not part of the 1.x kernels.
