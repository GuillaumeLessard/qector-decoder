# QECTOR Decoder v3

<div align="center">

[![CI](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/CI.yml/badge.svg)](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/CI.yml)
[![PyPI](https://img.shields.io/pypi/v/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![Python](https://img.shields.io/pypi/pyversions/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![License](https://img.shields.io/badge/License-Source_Available-blue)](LICENSE)
[![PyPI Downloads](https://img.shields.io/pypi/dm/qector-decoder-v3)](https://pypi.org/project/qector-decoder-v3/)

**Source-available Rust/Python quantum error correction decoding platform.**

**v0.6.2**: Hardened Union-Find family with explicit hypergraph rejection (no more silent wrong results on unsupported codes) + comprehensive validation.

PyMatching-compatible MWPM validation • Belief-matching accuracy mode • BP-OSD for LDPC/qLDPC • CPU/GPU batch decoding • Artifact-backed evidence

[Website](https://www.qector.store) — [PyPI](https://pypi.org/project/qector-decoder-v3/) — [Commercial licensing](mailto:admin@qector.store)

</div>

---

## Installation

```bash
pip install qector-decoder-v3
```

**Supported platforms**: Python 3.9–3.13 on Linux (manylinux), Windows x64, macOS arm64. Source builds available.

**Optional extras**:
```bash
pip install "qector-decoder-v3[stim]"   # Stim, Sinter, PyMatching, LDPC ecosystem
pip install "qector-decoder-v3[bench]"  # Benchmarking tools
pip install "qector-decoder-v3[all]"    # Full validation suite
```

**GPU check** (always verify on target machine):
```python
from qector_decoder_v3 import CUDABatchDecoder, OpenCLBatchDecoder
print("CUDA:", CUDABatchDecoder.is_available())
print("OpenCL:", OpenCLBatchDecoder.is_available())
```

---

## Quick Start (v0.6.2)

```python
import numpy as np
from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder, BatchDecoder, CUDABatchDecoder

# Example code (graph-like)
checks = [[0,1], [1,2], [2,3], [3,4]]
nq = 5
syndrome = np.array([0,1,0,0], dtype=np.uint8)

uf = UnionFindDecoder(checks, nq)
print(uf.decode(syndrome))

blossom = BlossomDecoder(checks, nq)
print(blossom.decode(syndrome))

# Batch decoding
batch = BatchDecoder(checks, nq)
syndromes = np.random.randint(0, 2, (4096, 4), dtype=np.uint8)
corrections = batch.parallel_batch_decode(syndromes)

if CUDABatchDecoder.is_available():
    gpu = CUDABatchDecoder(checks, nq)
    corrections = gpu.batch_decode(syndromes)
```

> **Note**: `UnionFindDecoder` / `FastUnionFindDecoder` explicitly reject hypergraph codes (e.g. surface codes with weight-4 checks) since v0.6.2. Use `BlossomDecoder`, `SparseBlossomDecoder` or `BpOsdDecoder` for general cases.

See `examples/` folder for more.

---

## Core Decoders

| Decoder | Best For | Status |
|---------|----------|--------|
| `UnionFindDecoder` / `FastUnionFindDecoder` | Low-latency approximate (graph-like codes) | Stable (hypergraph rejection) |
| `BlossomDecoder` | Exact MWPM, PyMatching validation | Stable |
| `SparseBlossomDecoder` | Near-optimal matching | Experimental |
| `BeliefMatching` | Correlated noise accuracy | Research mode |
| `BpOsdDecoder` | LDPC / qLDPC | Experimental |
| `BatchDecoder` / `CPUBatchDecoder` | High-throughput CPU | Stable |
| `CUDABatchDecoder` | GPU acceleration | Runtime dependent |

---

## v0.6.2 Highlights & Evidence

- Critical fix: Union-Find family now hard-errors on unsupported hypergraphs.
- 919 tests (832 Python + 87 Rust), syndrome faithfulness verified.
- Competitive LER parity with PyMatching on tested surface codes.
- Honest latency positioning (Blossom slower than PyMatching on standard workloads).
- Full reproduction scripts and benchmark artifacts included.

**Independent evaluation**: 9.7/10 (core correctness 10/10).

Full audit, methodology, and artifacts in [`docs/`](docs/) and `benchmark_results/`.

---

## Reproduce Benchmarks

```bash
# Competitive MWPM
python scripts/competitive_stim_ler.py --distances 3 5 7 9 11 13 15 --shots 40000

# Belief-matching
python scripts/competitive_belief_matching.py --distances 3 5 7 --shots 3000

# Full validation bundle
python scripts/run_due_diligence_bundle.py
```

Always regenerate artifacts on your hardware before quoting performance numbers.

---

## Commercial & Licensing

Source-available.  
Personal, academic, and non-commercial research use is free.  
Commercial use, redistribution, SaaS, or integration requires a paid license.

Contact: admin@qector.store

---

## Citation

```bibtex
@software{lessard2026qector,
  author  = {Guillaume Lessard},
  title   = {{QECTOR Decoder v3}: Rust/Python Quantum Error Correction Decoding Platform},
  year    = {2026},
  version = {0.6.2},
  url     = {https://github.com/GuillaumeLessard/qector-decoder}
}
```

---

**Full documentation**, examples, and security notes are in the repository.

**Copyright © 2026 Guillaume Lessard / iD01t Productions. All rights reserved.**
