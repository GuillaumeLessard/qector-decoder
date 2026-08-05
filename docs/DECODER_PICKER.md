# QECTOR Decoder v3 — Decoder Picker Guide

QECTOR ships multiple specialized decoder families. The policy functions `recommend_decoder` and `recommend_backend` automatically map your code topology, distance, and shot count to the optimal decoder.

---

## Automatic Recommendation API

```python
from qector_decoder_v3 import recommend_decoder, recommend_backend

# 1. Recommend optimal decoder class name
decoder_name = recommend_decoder(code_family="surface", distance=5, priority="accuracy")
print("Recommended Decoder:", decoder_name)
# -> 'BlossomDecoder'

# 2. Recommend execution backend based on measured crossover
backend_name = recommend_backend(n_qubits=25, distance=5, batch_size=50000, hardware="all")
print("Recommended Backend:", backend_name)
# -> 'cuda'
```

---

## Decoder Family Selection Matrix

| Problem Type | Recommended Decoder | Priority Goal | Key Asset |
|---|---|---|---|
| **Graphlike (Surface / Repetition)** | `BlossomDecoder` | Accuracy | Edmonds MWPM ($H \cdot c == s$) |
| **Graphlike (Large distance / scale)** | `SparseBlossomDecoder` | Accuracy + Speed | RadixHeap region growth |
| **Graphlike (High Throughput / Realtime)** | `FastUnionFindDecoder` | Speed | Zero-allocation $O(n \alpha(n))$ |
| **qLDPC / Hypergraph (BB72, BB144)** | `MatrixBPOSDDecoder` | Validity | Sum-product BP + OSD post-process |
| **Circuit-level noise / Correlated X-Z** | `BeliefMatching` | LER Parity | Hyperedge BP + MWPM edge mapping |
| **Large GPU Batches ($N \ge 40\text{k}$)** | `CUDABatchDecoder` | Throughput | Parallel GPU cluster growth |
