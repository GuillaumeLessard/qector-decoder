# QECTOR Decoder v3

[![PyPI version](https://img.shields.io/pypi/v/qector-decoder-v3)](https://pypi.org/project/qector-decoder-v3/)
[![Python versions](https://img.shields.io/pypi/pyversions/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![License](https://img.shields.io/badge/License-Source_Available-blue)](LICENSE)

**Production-grade quantum error correction decoding library — Rust core with Python bindings.**

*Copyright © 2026 Guillaume Lessard / iD01t Productions. All Rights Reserved.*

**Key Capabilities**
- PyMatching-compatible MWPM with exact parity validation
- Union-Find, Sparse Blossom, BP-OSD, and hybrid decoders
- CPU/GPU batch decoding with intelligent hardware routing
- 7-tier self-debugging fallback engine
- Ed25519 cryptographic license verification
- Comprehensive Stim/Sinter and ecosystem integration
- Artifact-backed, reproducible benchmarks

[Website](https://www.qector.store) · [PyPI](https://pypi.org/project/qector-decoder-v3/) · [Commercial Licensing](mailto:admin@qector.store)

---

### Installation

```bash
pip install qector-decoder-v3
```

**Supported Platforms**: Python 3.9–3.13 on Linux x86_64, Windows x64, and macOS arm64.

**Optional Extras**:
```bash
pip install "qector-decoder-v3[stim]"   # Stim, Sinter, PyMatching, LDPC integration
pip install "qector-decoder-v3[bench]" # Benchmarking tools
pip install "qector-decoder-v3[all]"   # Full development environment
```

---

### Quick Start

```python
import numpy as np
from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder

checks = [[0, 1], [1, 2], [2, 3], [3, 4]]
n_qubits = 5
syndrome = np.array([0, 1, 0, 0], dtype=np.uint8)

# Fast approximate decoding
uf = UnionFindDecoder(checks, n_qubits)
print(uf.decode(syndrome))

# Exact MWPM decoding
blossom = BlossomDecoder(checks, n_qubits)
print(blossom.decode(syndrome))
```

#### Batch & GPU Decoding
```python
from qector_decoder_v3 import BatchDecoder, CUDABatchDecoder

syndromes = np.random.randint(0, 2, size=(4096, 4), dtype=np.uint8)
cpu_decoder = BatchDecoder(checks, n_qubits)
corrections = cpu_decoder.parallel_batch_decode(syndromes)

if CUDABatchDecoder.is_available():
    gpu_decoder = CUDABatchDecoder(checks, n_qubits)
    corrections = gpu_decoder.batch_decode(syndromes)
```

#### AutoDecoder (Intelligent Routing + Self-Debug)
```python
from qector_decoder_v3 import AutoDecoder

decoder = AutoDecoder(checks, n_qubits)
corrections = decoder.batch_decode(syndromes)

# Diagnostics
print(decoder.diagnostics())
```

---

### Ecosystem Integration

**Stim / Sinter**
```python
import stim
from qector_decoder_v3.stim_compat import from_stim_detector_error_model
from qector_decoder_v3.sinter_compat import qector_sinter_decoders
```

**BP-OSD for LDPC/qLDPC**
```python
from qector_decoder_v3 import BpOsdDecoder, codes
```

**License Verification**
```python
from qector_decoder_v3.license import verify_license_token
```

---

### Decoder Overview

| Decoder Family              | Primary Use Case                     | Status      |
|-----------------------------|--------------------------------------|-------------|
| UnionFind / Fast UF         | Low-latency approximate decoding     | Stable      |
| BlossomDecoder              | Exact MWPM (PyMatching parity)       | Stable      |
| SparseBlossomDecoder        | High-performance near-optimal MWPM   | Stable      |
| BpOsdDecoder                | LDPC / qLDPC codes                   | Stable      |
| Batch / CPUBatchDecoder     | High-throughput CPU batch            | Stable      |
| CUDABatchDecoder            | GPU-accelerated batch                | Stable      |
| AutoDecoder                 | Intelligent 7-tier fallback routing | Stable      |
| Neural / GNN Predecoders    | Learned correction assistance        | Research    |

---

### Architecture Highlights (v0.6.8)
- **7-Tier Self-Debugging Backend**: Automatic hardware failure detection, health monitoring, and seamless fallback.
- **Cryptographic Licensing**: Offline Ed25519 token verification with Stripe fulfillment.
- **Reproducibility**: Full benchmark artifacts and cross-decoder validation suites.

**Full documentation, licensing details, and commercial pricing** are available at [qector.store](https://www.qector.store).

---

**QECTOR Decoder v3** is source-available for personal, academic, and non-commercial research use. Commercial, institutional, or production deployment requires a license.

Contact: [admin@qector.store](mailto:admin@qector.store)

*Copyright © 2026 Guillaume Lessard / iD01t Productions. All Rights Reserved.*
```

This version removes the broken CI badge while keeping everything else professional and functional. You can copy-paste it directly. 

If you make the repo public later and add a working CI workflow, you can re-add the badge. Let me know if you need more changes.
