# QECTOR Decoder v3

[![CI](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/CI.yml/badge.svg)](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/CI.yml)
[![PyPI](https://img.shields.io/pypi/v/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![Python](https://img.shields.io/pypi/pyversions/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![License](https://img.shields.io/badge/License-Source_Available-blue)](LICENSE)
**Source-available Rust/Python quantum error correction decoding platform.**

PyMatching-compatible MWPM validation - Belief-matching accuracy mode - BP-OSD for LDPC/qLDPC - CPU/GPU batch decoding - Artifact-backed benchmark evidence

**Companion tooling**: the public package snapshot focuses on the decoder library, Python API, validation suite, and benchmark evidence. Companion desktop and automation surfaces are distributed separately from this checkout.

[Website](https://www.qector.store) - [PyPI](https://pypi.org/project/qector-decoder-v3/) - [Commercial licensing](mailto:admin@qector.store)

---

## Install

### Stable package

```bash
pip install qector-decoder-v3
```

Supported public wheel target: **Python 3.9 to 3.13** on Linux, Windows, and macOS where wheels are published for the release.

### Optional Python extras

```bash
# Stim, Sinter, PyMatching, LDPC and belief-matching ecosystem
pip install "qector-decoder-v3[stim]"

# Benchmark harness: psutil, matplotlib, scipy, tabulate
pip install "qector-decoder-v3[bench]"

# Everything needed for validation and benchmark scripts
pip install "qector-decoder-v3[all]"
```

### GPU runtime check

GPU support is runtime and build dependent. The public CI wheel workflow currently builds the CUDA feature path; OpenCL-capable builds are validated in checked-in benchmark artifacts but may require a licensed/custom build. Always detect support on the target machine before quoting performance.

```python
from qector_decoder_v3 import CUDABatchDecoder, OpenCLBatchDecoder

print("CUDA:", CUDABatchDecoder.is_available())
print("OpenCL:", OpenCLBatchDecoder.is_available())
```

### Licensed source build

The public repository contains the Python layer and a Rust source stub. The proprietary Rust core is injected during trusted CI/release builds or provided under commercial license.

```bash
git clone https://github.com/GuillaumeLessard/qector-decoder.git
cd qector-decoder
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip maturin
python -m pip install -e "[stim,bench]"
```

For a full native Rust extension build, use the licensed Rust source bundle alongside the packaging metadata in [pyproject.toml](pyproject.toml) and the checked-in Rust sources in [src](src).

---

## Quick start

```python
import numpy as np
from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder

check_to_qubits = [[0, 1], [1, 2], [2, 3], [3, 4]]
n_qubits = 5
syndrome = np.array([0, 1, 0, 0], dtype=np.uint8)

fast = UnionFindDecoder(check_to_qubits, n_qubits)
print(fast.decode(syndrome))

mwpm = BlossomDecoder(check_to_qubits, n_qubits)
print(mwpm.decode(syndrome))
```

### Batch decoding

```python
import numpy as np
from qector_decoder_v3 import BatchDecoder, CUDABatchDecoder

checks = [[0, 1], [1, 2], [2, 3], [3, 4]]
syndromes = np.random.randint(0, 2, size=(4096, 4), dtype=np.uint8)

cpu = BatchDecoder(checks, n_qubits=5)
corrections = cpu.parallel_batch_decode(syndromes)

if CUDABatchDecoder.is_available():
    gpu = CUDABatchDecoder(checks, n_qubits=5)
    corrections = gpu.batch_decode(syndromes)
```

### Stim detector-error-model workflow

```python
import stim
from qector_decoder_v3 import BlossomDecoder
from qector_decoder_v3.stim_compat import from_stim_detector_error_model

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=5,
    rounds=5,
    after_clifford_depolarization=0.005,
)

dem = circuit.detector_error_model(decompose_errors=True)
checks, n_qubits = from_stim_detector_error_model(dem)
decoder = BlossomDecoder(checks, n_qubits)
```

### Belief-matching accuracy mode

```python
import stim
from qector_decoder_v3.belief_matching import BeliefMatching

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=5,
    rounds=5,
    after_clifford_depolarization=0.005,
)

bm = BeliefMatching.from_stim_circuit(circuit)
sampler = circuit.compile_detector_sampler()
(syndrome,) = sampler.sample(shots=1)
correction = bm.decode(syndrome.astype("uint8"))
```

### BP-OSD for LDPC / qLDPC codes

```python
from qector_decoder_v3 import codes
from qector_decoder_v3.bposd import BpOsdDecoder

cx, cz = codes.bivariate_bicycle_code(
    6,
    6,
    [("x", 3), ("y", 1), ("y", 2)],
    [("y", 3), ("x", 1), ("x", 2)],
)

decoder = BpOsdDecoder(cx.parity_check_matrix(), error_rate=0.05, osd_order=0)
correction = decoder.decode(syndrome)
```

---

## What it includes

| Decoder / module | Best use | Status |
| --- | --- | --- |
| `UnionFindDecoder` | Low-latency approximate decoding | Stable public API |
| `FastUnionFindDecoder` | Optimized Union-Find hot path | Stable public API |
| `BlossomDecoder` | Exact MWPM / PyMatching-parity validation | Stable public API |
| `SparseBlossomDecoder` | Faster near-optimal matching | Experimental correctness envelope |
| `BeliefMatching` | Correlated-noise accuracy experiments | Accuracy mode, not latency mode |
| `BpOsdDecoder` | LDPC / qLDPC decoding workflows | Experimental / research |
| `BatchDecoder` / `CPUBatchDecoder` | CPU batch Monte Carlo sweeps | Stable public API |
| `CUDABatchDecoder` | CUDA batch decoding | Runtime/build dependent |
| `OpenCLBatchDecoder` | OpenCL batch decoding | Runtime/build dependent |
| `AutoDecoder` | CPU/GPU backend calibration | Experimental |
| `PredecodedDecoder` | Easy-syndrome prefiltering | Experimental |
| `DecoderPool` | Multi-process batch decoding (auto-Rayon on Windows) | Stable public API |
| `get_decoder` | Cached decoder factory (zero construction cost after first call) | Stable public API |
| `clear_decoder_cache` | Clear the decoder cache | Stable public API |
| `get_decoder_pool` | Cached DecoderPool factory | Stable public API |
| `decode_mmap` | Out-of-core decoding via memory-mapped arrays | Stable public API |
| `DecodeResult` | Structured decode result with diagnostics | Stable public API |
| `decode_with_diagnostics` | Decode with detailed diagnostic info | Stable public API |
| `Workbench` | High-level workbench orchestration | Stable public API |
| `SlidingWindowDecoder` | Multi-round streaming workflows | Experimental |
| `stim_compat` | Stim circuit / DEM conversion | Stable utility |
| `sinter_compat` | Sinter custom decoder integration | Stable utility |
| `rest_api` | Local service endpoint | Local/partner review only |

See the public API regression coverage in [python/tests](python/tests) before building production code on experimental modules.

---

## New in v0.6.4

| Feature | Description |
| --- | --- |
| BP-OSD `decode_timed` | Wall-clock deadline for BP iterations; falls back to hard-decision on timeout |
| AVX2 runtime dispatch | CPU batch decoder auto-detects AVX2 support and uses SIMD transpose for 1.1M shots/s |
| Blossom intra-decode parallelism | Rayon-parallelized Blossom matching for multi-shot batches |
| DecoderPool Windows fix | Auto-Rayon fallback on Windows when multi-process pool is unavailable |
| `DecoderPool` | Multi-process batch decoding with automatic worker management |
| `get_decoder` / `clear_decoder_cache` / `get_decoder_pool` | Cached decoder factory — zero construction cost after first call |
| `decode_mmap` | Out-of-core decoding via memory-mapped NumPy arrays |
| `DecodeResult` / `decode_with_diagnostics` | Structured decode results with per-shot diagnostic metadata |
| `Workbench` | High-level orchestration for multi-decoder comparison and benchmarking |

---

## Companion tooling and partner distribution

The public repository snapshot focuses on the decoder package, its Python API, validation suite, and reproducible benchmark evidence. Companion desktop, automation, and documentation tooling may be provided separately through partner or commercial distribution channels and are not bundled in this checkout.

If you need the full desktop GUI, hosted automation stack, or additional documentation-generation tooling, contact the project team through the commercial site listed above.

---

## Validated evidence snapshot

All public claims should cite an artifact, commit, command, machine, and version. The current package release is **v0.6.4**; checked-in evidence should be regenerated before making new performance claims.

> **v0.6.4 additions**: AVX2 SIMD transpose (CPU batch 1.1M shots/s), BP-OSD convergence cap (`decode_timed`), Blossom intra-decode Rayon parallelism, DecoderPool auto-Rayon on Windows.

### MWPM parity against PyMatching

Artifact: `benchmark_results/stim_ler_d13_d15.json`

Environment: Windows 10/11 class x64 machine, Python 3.11+, QECTOR v0.6.4 + v3.3 Workbench, PyMatching 2.4+, Stim 1.16+, 20,000 shots per distance.

| Distance | QECTOR Blossom LER | PyMatching LER | QECTOR us/shot | PyMatching us/shot |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 0.00075 | 0.00075 | 820.46 | 81.12 |
| 15 | 0.00050 | 0.00050 | 1965.15 | 203.20 |

Interpretation: QECTOR Blossom matched PyMatching logical-error counts on this artifact. PyMatching remains much faster for standard MWPM latency on these workloads.

### Belief-matching accuracy experiment

Artifact: `benchmark_results/competitive_belief.json`

Environment: Windows x64, Python 3.11, QECTOR 0.5.7, PyMatching 2.4.0, Stim 1.16.0, 3,000 shots per distance.

| Distance | PyMatching LER | QECTOR MWPM LER | QECTOR Belief LER | Belief us/shot |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.01167 | 0.01167 | 0.01233 | 2331.07 |
| 5 | 0.00767 | 0.00767 | 0.00500 | 12125.38 |
| 7 | 0.00600 | 0.00600 | 0.00300 | 54323.56 |

Interpretation: belief-matching improved observed LER at d=5 and d=7 in this artifact but was dramatically slower. It should be positioned as an accuracy/research mode, not a production latency path.

### GPU bit-identity artifact

Artifact: `benchmark_results/gpu_extensive.json`

Environment: NVIDIA GeForce GTX 1660 Ti, Python 3.11, CUDA and OpenCL available, distances 3 to 13, batch sizes 1 to 65,536.

| Claim | Artifact result |
| --- | --- |
| Number of tested configurations | 36 |
| CUDA bit-identical to CPU | true |
| OpenCL bit-identical to CPU | true |
| All tested GPU paths faithful | true |

Interpretation: this is a correctness and reproducibility artifact for one machine. It is not a universal GPU speed claim.

### Native memory artifact

Artifact: `benchmark_results/native_memory.json`

Distance 13, batch 16,384:

| Decoder | RSS base MiB | RSS peak MiB | Native delta MiB |
| --- | ---: | ---: | ---: |
| `cpu_batch` | 120.98 | 130.39 | 9.41 |
| `blossom` | 123.64 | 129.52 | 5.88 |
| `fast_union_find` | 121.98 | 122.00 | 0.02 |
| `cuda_batch` | 211.57 | 214.24 | 2.67 |

---

## Reproduce locally

```bash
# MWPM / PyMatching comparison
python scripts/competitive_stim_ler.py --distances 3 5 7 9 11 13 15 --shots 40000

# Belief-matching comparison
python scripts/competitive_belief_matching.py --distances 3 5 7 --shots 3000 --no-ref

# GPU correctness and crossover checks
python scripts/gpu_extensive_test.py --distances 3 5 7 9 11 13 --batches 1 64 1024 4096 16384 65536 --error-rate 0.05

# Native memory profile
python scripts/native_memory_profile.py --distances 5 9 13 --batch 16384 --out benchmark_results/native_memory

# Full due-diligence bundle
python scripts/run_due_diligence_bundle.py --out qector_evidence_bundle
```

Benchmark results are hardware, driver, compiler, seed, and workload dependent. Regenerate before quoting throughput, latency, GPU speedup, or buyer-facing performance numbers.

---

## Architecture

```text
qector_decoder_v3/
+-- Rust core, proprietary
|   +-- Union-Find / Blossom / SparseBlossom engines
|   +-- CPU batch engine
|   +-- CUDA / OpenCL batch paths where enabled
|   +-- DEM collapse and Stim integration support
|   +-- Native Python extension
|
+-- Python layer, public in this repository
    +-- __init__.py
    +-- belief_matching.py
    +-- bposd.py
    +-- predecoder.py
    +-- backend.py
    +-- dem.py
    +-- stim_compat.py
    +-- sinter_compat.py
    +-- qiskit_plugin.py
    +-- rest_api.py
    +-- workbench.py
    +-- codes.py
```

The Rust core is built into the native extension during local development and packaging. The checked-in sources in [src](src) are the implementation used for this workspace build, while release-wheel packaging relies on the project’s selected build environment.

---

## Sinter integration

```python
import sinter
from qector_decoder_v3.sinter_compat import qector_sinter_decoders

tasks = [...]  # list[sinter.Task]

samples = sinter.collect(
    num_workers=4,
    tasks=tasks,
    decoders=["qector_belief", "qector_blossom", "qector_unionfind"],
    custom_decoders=qector_sinter_decoders(),
)
```

---

## REST API, local only

```bash
pip install "qector-decoder-v3[stim]" fastapi uvicorn
python -m qector_decoder_v3.rest_api
```

```bash
curl -X POST http://localhost:8000/decode \
  -H "Content-Type: application/json" \
  -d '{"check_to_qubits":[[0,1],[1,2],[2,3],[3,4]],"syndrome":[0,1,0,0]}'
```

The REST API is for local experiments, partner review, or controlled internal deployments. Do not expose it publicly without authentication, TLS, authorization, logging, input limits, and rate limiting.

---

## Limits and claim boundaries

| Area | Boundary |
| --- | --- |
| MWPM latency | PyMatching remains the speed leader on standard surface-code MWPM workloads in the provided artifacts. |
| Belief-matching | Accuracy/research mode. It can improve observed LER on selected workloads but is much slower. |
| GPU performance | Correctness is artifact-backed for tested machines. Speedup is not universal. |
| OpenCL wheels | OpenCL support depends on build configuration and target environment. Confirm locally. |
| SparseBlossom | Near-optimal, not exact MWPM. Use `BlossomDecoder` for exact minimum-weight matching. |
| UnionFind | Fast approximate path; not a universal decoder for arbitrary graphs. |
| REST/gRPC/MCP surfaces | Not hardened as public SaaS without a separate deployment/security review. |

---

## Documentation available in this checkout

The public checkout currently includes the following primary reference material:

- [README.md](README.md) — package overview, install steps, and quick-start usage
- [PYPI_README.md](PYPI_README.md) — PyPI-facing package summary
- [AGENTS.md](AGENTS.md) — repository-specific development and validation notes
- [LICENSE](LICENSE) — repository license terms
- [pyproject.toml](pyproject.toml) — packaging metadata and optional dependency groups
- [python/tests](python/tests) — public API and regression tests that validate the current behavior

---

## CI / CD

The current checkout is validated through the Python regression suite under [python/tests](python/tests) and the packaging metadata in [pyproject.toml](pyproject.toml). Release automation and wheel publication are managed through the upstream project workflow outside this local snapshot.

---

## Repository cleanup status

The public checkout now emphasizes the package sources, Python API, regression tests, packaging metadata, and the validated evidence captured in the repository history and release metadata.

---

## Commercial model

QECTOR Decoder v3 is source-available.

| Use | License required |
| --- | --- |
| Personal, academic, educational, non-commercial research | Free under the repository license |
| Company use, commercial R&D, institutional funded work | Paid commercial license |
| SaaS, hosted API, OEM embedding, product integration, redistribution | Paid commercial license |
| Commercial benchmarking, paid consulting, revenue-linked work | Paid commercial license |

Commercial contact: [admin@qector.store](mailto:admin@qector.store)

Website: [https://www.qector.store](https://www.qector.store)

See [LICENSE](LICENSE) for the repository terms and contact the commercial team above for separate licensing.

---

## Citation

```bibtex
@software{lessard2026qector,
  author  = {Guillaume Lessard},
  title   = {{QECTOR Decoder v3}: Rust/Python Quantum Error Correction Decoding Platform},
  year    = {2026},
  version = {0.6.4},
  url     = {https://www.qector.store},
  note    = {Source-available. Commercial license required for commercial use.}
}
```

---

**Copyright (c) 2026 Guillaume Lessard / iD01t Productions. All rights reserved.**

[https://www.qector.store](https://www.qector.store) - [admin@qector.store](mailto:admin@qector.store)
