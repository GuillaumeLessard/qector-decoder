# QECTOR Decoder v3

<div align="center">

[![CI](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/CI.yml/badge.svg)](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/CI.yml)
[![tests](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/tests.yml/badge.svg)](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![Python](https://img.shields.io/pypi/pyversions/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![License](https://img.shields.io/badge/license-Source--Available-blue.svg)](LICENSE)

**Rust/Python quantum error correction decoding platform.**  
Exact-MWPM parity with PyMatching · Belief-matching accuracy mode · BP-OSD for LDPC/qLDPC · CPU/GPU batch decoding · Artifact-backed benchmarks

[🌐 qector.store](https://www.qector.store) · [📦 PyPI](https://pypi.org/project/qector-decoder-v3/) · [📧 Commercial licensing](mailto:admin@qector.store)

</div>

---

## Install

```bash
pip install qector-decoder-v3
```

Prebuilt wheels for **Python 3.9–3.13** on **Linux**, **Windows**, and **macOS** (Intel + Apple Silicon).  
No Rust toolchain required. No CUDA required for the CPU build.

### Optional extras

```bash
# Stim/Sinter/PyMatching ecosystem
pip install "qector-decoder-v3[stim]"

# Benchmark harness (psutil, matplotlib, scipy, tabulate)
pip install "qector-decoder-v3[bench]"

# All extras
pip install "qector-decoder-v3[all]"
```

### GPU builds

CUDA and OpenCL backends are compiled into the wheel and activated at runtime when drivers are present — no rebuild needed.

```python
from qector_decoder_v3 import CUDABatchDecoder, OpenCLBatchDecoder
print(CUDABatchDecoder.is_available())   # True if CUDA driver found
print(OpenCLBatchDecoder.is_available()) # True if OpenCL platform found
```

---

## Quick start

```python
import numpy as np
from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder

# Define a simple repetition code (5 qubits, 4 checks)
check_to_qubits = [[0, 1], [1, 2], [2, 3], [3, 4]]
n_qubits = 5

# Fast approximate decoder
decoder = UnionFindDecoder(check_to_qubits, n_qubits)
syndrome = np.array([0, 1, 0, 0], dtype=np.uint8)  # one error
correction = decoder.decode(syndrome)
print(correction)  # [0 1 0 0 0]

# Exact MWPM decoder (PyMatching-parity)
blossom = BlossomDecoder(check_to_qubits, n_qubits)
correction = blossom.decode(syndrome)
print(correction)  # [0 1 0 0 0]
```

### Batch decoding (CPU / GPU)

```python
from qector_decoder_v3 import BatchDecoder, CUDABatchDecoder
import numpy as np

checks = [[0,1],[1,2],[2,3],[3,4]]
syndromes = np.random.randint(0, 2, size=(4096, 4), dtype=np.uint8)

# CPU parallel (Rayon)
dec = BatchDecoder(checks, n_qubits=5)
corrections = dec.parallel_batch_decode(syndromes)

# GPU (falls back to CPU if no driver)
if CUDABatchDecoder.is_available():
    gpu = CUDABatchDecoder(checks, n_qubits=5)
    corrections = gpu.batch_decode(syndromes)
```

### Stim / DEM workflow

```python
import stim
from qector_decoder_v3.stim_compat import stim_circuit_to_check_matrix
from qector_decoder_v3 import BlossomDecoder

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=5, rounds=5, after_clifford_depolarization=0.005
)
checks, n_qubits = stim_circuit_to_check_matrix(circuit)
decoder = BlossomDecoder(checks, n_qubits)
```

### Belief-matching (accuracy mode)

```python
from qector_decoder_v3.belief_matching import BeliefMatching

bm = BeliefMatching(check_to_qubits, n_qubits, error_rate=0.005)
correction = bm.decode(syndrome)
```

### BP-OSD for LDPC / qLDPC codes

```python
from qector_decoder_v3.bposd import BpOsdDecoder
from qector_decoder_v3 import codes

cx, cz = codes.bivariate_bicycle_code(
    6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]
)
decoder = BpOsdDecoder(cx.parity_check_matrix(), error_rate=0.05, osd_order=0)
correction = decoder.decode(syndrome)
```

---

## Decoder reference

| Decoder | Best for | Notes |
|---|---|---|
| `UnionFindDecoder` | Ultra-low latency, surface/repetition codes | Fast approximate; not guaranteed on arbitrary graphs |
| `FastUnionFindDecoder` | Same, optimized hot path | Tightest latency budget |
| `BlossomDecoder` | Exact MWPM, PyMatching parity | Validated to d=15 vs PyMatching |
| `SparseBlossomDecoder` | Near-optimal, faster than Blossom | Region-growing; not exact MWPM |
| `BeliefMatching` | Correlated noise, accuracy mode | Lower LER on tested workloads; slower per-shot |
| `BpOsdDecoder` | LDPC / qLDPC (bivariate bicycle, HGP) | BP + ordered-statistics post-process |
| `BatchDecoder` / `CPUBatchDecoder` | High-throughput CPU batch | Rayon data-parallel |
| `CUDABatchDecoder` | GPU batch, CUDA | Bit-identical to CPU; auto CPU fallback |
| `OpenCLBatchDecoder` | GPU batch, OpenCL (AMD/Intel/NVIDIA) | Bit-identical to CPU; auto CPU fallback |
| `AutoDecoder` | Calibrated CPU/GPU dispatch | Measures crossover, picks best backend |
| `PredecodedDecoder` | Pre-filter easy syndromes before full decode | Pairs with Blossom/UnionFind residual |
| `SlidingWindowDecoder` | Multi-round / real-time workflows | Streaming capable |

See [`docs/API_STABILITY.md`](docs/API_STABILITY.md) for stable vs experimental surface boundaries.

---

## Validated claims (v0.5.0)

All claims below are backed by checked-in benchmark artifacts and reproducible scripts.

| Claim | Evidence |
|---|---|
| Exact-MWPM LER parity with PyMatching through **d=15** | `benchmark_results/stim_ler_d13_d15.json` |
| Belief-matching lower observed LER on tested correlated workloads | `benchmark_results/competitive_belief.json` |
| GPU (CUDA/OpenCL) output **bit-identical** to CPU across batch sizes 1–65 536 | `benchmark_results/gpu_extensive.json` |
| BP-OSD within tolerance of reference `ldpc` package on tested LDPC families | `benchmark_results/competitive_belief.json` |
| Native RSS memory stays bounded; no Python GC leak in hot path | `benchmark_results/native_memory.json` |

Artifacts are **reference evidence**, not universal hardware claims.  
Regenerate locally before citing throughput, latency, or GPU speedup numbers.

```bash
# Reproduce competitive LER benchmark locally
python scripts/competitive_stim_ler.py --distances 3 5 7 9 11 13 15 --shots 40000

# Reproduce belief-matching comparison
python scripts/competitive_belief_matching.py --distances 3 5 7 9 --shots 20000

# Full due-diligence bundle (all artifacts, hashes, environment)
python scripts/run_due_diligence_bundle.py --out qector_evidence_bundle
```

---

## Architecture

```
qector_decoder_v3/
├── Rust core (proprietary, injected at CI build time)
│   ├── Union-Find / Blossom / SparseBlossom matchers
│   ├── Rayon CPU batch engine
│   ├── CUDA + OpenCL GPU batch engines
│   └── DEM collapse + Stim interface
│
└── Python layer (open in this repo)
    ├── __init__.py          — public API surface
    ├── belief_matching.py   — belief-matching accuracy mode
    ├── bposd.py             — BP-OSD for LDPC/qLDPC
    ├── predecoder.py        — local-matching predecoder
    ├── backend.py           — AutoDecoder, calibration
    ├── dem.py               — DEM parser
    ├── stim_compat.py       — Stim circuit → check matrix
    ├── sinter_compat.py     — Sinter Decoder interface
    ├── qiskit_plugin.py     — Qiskit result adapter
    ├── rest_api.py          — optional FastAPI/Flask service
    ├── workbench.py         — benchmark harness
    └── codes.py             — built-in code families
```

The Rust core is compiled into the wheel binary. It is proprietary and not distributed in source form. The `src/lib.rs` in this repo is a stub — see [`COMMERCIAL.md`](COMMERCIAL.md) for licensing.

---

## Sinter integration

```python
import sinter
from qector_decoder_v3.sinter_compat import qector_sinter_decoders

tasks = [...]  # list of sinter.Task
samples = sinter.collect(
    num_workers=4,
    tasks=tasks,
    decoders=["qector_belief", "qector_blossom", "qector_unionfind"],
    custom_decoders=qector_sinter_decoders(),
)
```

---

## REST API (optional, local only)

```bash
pip install "qector-decoder-v3[stim]" fastapi uvicorn
python -m qector_decoder_v3.rest_api
```

```bash
curl -X POST http://localhost:8000/decode \
  -H "Content-Type: application/json" \
  -d '{"check_to_qubits":[[0,1],[1,2],[2,3],[3,4]],"syndrome":[0,1,0,0]}'
```

> ⚠️ The REST API is for local / partner-review use only.  
> Do not expose it publicly without authentication, TLS, and rate limiting.  
> See [`docs/SECURITY_DEPLOYMENT.md`](docs/SECURITY_DEPLOYMENT.md).

---

## Honest limitations

- **PyMatching is the latency leader** for exact MWPM on standard surface-code workloads. QECTOR's value is parity correctness, workflow packaging, GPU batch paths, belief-matching, and LDPC support.
- **Belief-matching is an accuracy mode**, not a fast path. It rebuilds weights per shot.
- **SparseBlossom is near-optimal**, not exact MWPM. Use `BlossomDecoder` when minimum weight is required.
- **GPU wins only at suitable batch sizes** (typically ≥ 4096) and hardware configurations. Measure on your hardware before quoting speedup.
- **UnionFind can fail on adversarial non-surface-code graphs.** Use Blossom for general correctness guarantees.
- **REST/gRPC/MCP surfaces** are not hardened for production SaaS without a separate commercial deployment review.

---

## Docs

| Document | Contents |
|---|---|
| [`docs/API_STABILITY.md`](docs/API_STABILITY.md) | Stable vs experimental API surface |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Validation methodology |
| [`docs/CORRECTNESS_AUDIT.md`](docs/CORRECTNESS_AUDIT.md) | Correctness invariants |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | Reproduction commands |
| [`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md) | Pre-publication checklist |
| [`docs/SCALING.md`](docs/SCALING.md) | Scaling notes |
| [`docs/BEYOND_PYMATCHING.md`](docs/BEYOND_PYMATCHING.md) | Belief-matching and BP-OSD positioning |
| [`docs/BENCHMARK_COMPETITIVE.md`](docs/BENCHMARK_COMPETITIVE.md) | Competitive benchmark methodology |
| [`docs/SECURITY_DEPLOYMENT.md`](docs/SECURITY_DEPLOYMENT.md) | REST/gRPC/Docker hardening |
| [`docs/SERVICE_API_SCHEMA.md`](docs/SERVICE_API_SCHEMA.md) | REST schema and limitations |
| [`docs/PLATFORM_ARTIFACT_ROADMAP.md`](docs/PLATFORM_ARTIFACT_ROADMAP.md) | CI, SBOM, wheel roadmap |
| [`INSTALL.md`](INSTALL.md) | Source build (requires licensed Rust source) |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |
| [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | v0.5.0 release notes |
| [`SECURITY.md`](SECURITY.md) | Vulnerability reporting |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guidelines |

---

## CI / CD

| Workflow | Triggers | Jobs |
|---|---|---|
| `tests.yml` | Every push / PR | ruff lint · mypy · docker build · smoke import (py3.9–3.13) |
| `CI.yml` | Every push / `v*` tag | Wheels: Linux · Windows · macOS Intel · macOS ARM · sdist · PyPI publish (OIDC, tag-only) |

PyPI publish uses [OIDC Trusted Publisher](https://docs.pypi.org/trusted-publishers/) — no stored token.

---

## Commercial model

QECTOR Decoder v3 is **source-available**.

| Use | License required |
|---|---|
| Personal, academic, educational, non-commercial research | Free — see [`LICENSE`](LICENSE) |
| Company use, commercial R&D, government / institutional funded work | Paid commercial license |
| SaaS, hosted API, OEM embedding, product integration, redistribution | Paid commercial license |
| Commercial benchmarking, revenue-linked use, paid consulting | Paid commercial license |

**Commercial contact:** [admin@qector.store](mailto:admin@qector.store)  
**Website:** [https://www.qector.store](https://www.qector.store)

See [`COMMERCIAL.md`](COMMERCIAL.md) and [`LICENSE`](LICENSE) for full terms.

---

## Citation

```bibtex
@software{lessard2026qector,
  author  = {Guillaume Lessard},
  title   = {{QECTOR Decoder v3}: Rust/Python Quantum Error Correction Decoding Platform},
  year    = {2026},
  version = {0.5.0},
  url     = {https://www.qector.store},
  note    = {Source-available. Commercial license required for commercial use.}
}
```

---

<div align="center">

**Copyright © 2026 Guillaume Lessard / iD01t Productions. All rights reserved.**  
[https://www.qector.store](https://www.qector.store) · [admin@qector.store](mailto:admin@qector.store)

</div>
