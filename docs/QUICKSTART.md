# QECTOR Decoder v3 — Quickstart Guide

Get started with high-performance quantum error correction decoding in Python and Rust.

---

## Installation

```bash
pip install qector-decoder-v3
```

Optional ecosystem extras:
```bash
pip install "qector-decoder-v3[stim]"   # Stim, Sinter, PyMatching, LDPC
pip install "qector-decoder-v3[bench]"  # Benchmarking & telemetry
```

---

## 1-Line PyMatching Drop-in Replacement

QECTOR provides a 100% compatible PyMatching shim:

```python
# Change this import:
# import pymatching
import qector_decoder_v3.pymatching_compat as pymatching

# Existing PyMatching code runs unmodified with QECTOR performance:
matching = pymatching.Matching.from_detector_error_model(dem)
prediction = matching.decode_batch(syndromes)
```

---

## Direct Decoder Usage

```python
import numpy as np
from qector_decoder_v3 import BlossomDecoder, FastUnionFindDecoder

# Repetition code parity checks (n_checks = 4, n_qubits = 5)
check_to_qubits = [[0, 1], [1, 2], [2, 3], [3, 4]]
syndrome = np.array([0, 1, 0, 0], dtype=np.uint8)

# Minimum-Weight Perfect Matching (Edmonds Blossom)
blossom = BlossomDecoder(check_to_qubits, n_qubits=5)
correction = blossom.decode(syndrome)
print("Blossom Correction:", correction)

# Fast Union-Find (throughput optimized)
uf = FastUnionFindDecoder(check_to_qubits, n_qubits=5)
correction_uf = uf.decode(syndrome)
print("Union-Find Correction:", correction_uf)
```

---

## Batch Decoding

```python
import numpy as np
from qector_decoder_v3 import BatchDecoder, CUDABatchDecoder

checks = [[0, 1], [1, 2], [2, 3], [3, 4]]
batch_syndromes = np.random.randint(0, 2, size=(10000, 4), dtype=np.uint8)

# Multi-core CPU Rayon batching
cpu_batch = BatchDecoder(checks, n_qubits=5)
corrections_cpu = cpu_batch.batch_decode(batch_syndromes)

# GPU acceleration (requires CUDA driver + Pro/Enterprise key)
if CUDABatchDecoder.is_available():
    gpu_batch = CUDABatchDecoder(checks, n_qubits=5)
    corrections_gpu = gpu_batch.batch_decode(batch_syndromes)
```
