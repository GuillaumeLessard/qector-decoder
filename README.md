# QECTOR Decoder v3

<div align="center">

[![CI](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/CI.yml/badge.svg)](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/CI.yml)
[![tests](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/tests.yml/badge.svg)](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![Python](https://img.shields.io/pypi/pyversions/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![License](https://img.shields.io/badge/license-Source--Available-blue.svg)](LICENSE)

**Source-available Rust/Python quantum error correction decoding platform.**

PyMatching-compatible MWPM validation · Belief-matching accuracy mode · BP-OSD for LDPC/qLDPC · CPU/GPU batch decoding · Artifact-backed benchmark evidence

[Website](https://www.qector.store) · [PyPI](https://pypi.org/project/qector-decoder-v3/) · [Repository](https://github.com/GuillaumeLessard/qector-decoder) · [Commercial licensing](https://www.qector.store)

</div>

---

## Install

### Recommended PyPI command

```bash
pip install qector-decoder-v3
```

This is the command shown on PyPI and the safest command for normal users because it uses the `pip` bound to the active Python environment.

Verify the install with the same Python environment:

```bash
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; print('QECTOR OK')"
```

Check which Python and pip are being used:

```bash
python --version
pip --version
```

### Windows note

Do not force `py -m pip` unless you have checked which interpreter the Windows launcher selected. On some systems, `py` can select a free-threaded interpreter such as `python3.13t.exe`. QECTOR v0.5.x publishes standard CPython wheels, not `cp313t` free-threaded wheels. If pip cannot find a matching wheel, it may fall back to a source build and fail because the public repository does not ship the proprietary Rust core.

Inspect installed Python launchers with:

```powershell
py -0p
```

Use the working PyPI command from the active standard Python environment:

```powershell
pip install qector-decoder-v3
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; print('QECTOR OK')"
```

### Supported public wheel targets

QECTOR v0.5.x public releases target standard CPython wheels for:

| Platform | Wheel status |
|---|---|
| Linux x86_64 | Published |
| Windows x64 | Published |
| macOS arm64 / Apple Silicon | Published |
| macOS Intel x86_64 | Not published in v0.5.x public CI |
| CPython free-threaded builds such as `cp313t` | Not published in v0.5.x |

Supported Python version classifiers are standard CPython **3.9 to 3.13**.

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

The public repository contains the Python layer and a Rust source stub. The proprietary Rust core is injected during trusted CI/release builds or provided under commercial license. A full native source build requires the licensed Rust source bundle.

```bash
git clone https://github.com/GuillaumeLessard/qector-decoder.git
cd qector-decoder
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin
pip install -e ".[stim,bench]"
```

For a full native Rust extension build, unpack the licensed Rust source bundle first, then follow [`INSTALL.md`](INSTALL.md).

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
from qector_decoder_v3.stim_compat import stim_circuit_to_check_matrix

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=5,
    rounds=5,
    after_clifford_depolarization=0.005,
)

checks, n_qubits = stim_circuit_to_check_matrix(circuit)
decoder = BlossomDecoder(checks, n_qubits)
```

### Belief-matching accuracy mode

```python
from qector_decoder_v3.belief_matching import BeliefMatching

bm = BeliefMatching(check_to_qubits, n_qubits, error_rate=0.005)
correction = bm.decode(syndrome)
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
|---|---|---|
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
| `SlidingWindowDecoder` | Multi-round streaming workflows | Experimental |
| `stim_compat` | Stim circuit / DEM conversion | Stable utility |
| `sinter_compat` | Sinter custom decoder integration | Stable utility |
| `rest_api` | Local service endpoint | Local/partner review only |

See [`docs/API_STABILITY.md`](docs/API_STABILITY.md) before building production code on experimental modules.

---

## Validated evidence snapshot

All public claims should cite an artifact, commit, command, machine, and version. The checked-in evidence supports the following scoped claims for **v0.5.0**.

### MWPM parity against PyMatching

Artifact: `benchmark_results/stim_ler_d13_d15.json`

Environment: Windows 10/11 class x64 machine, Python 3.11, QECTOR 0.5.0, PyMatching 2.4.0, Stim 1.16.0, 20,000 shots per distance.

| Distance | QECTOR Blossom LER | PyMatching LER | QECTOR us/shot | PyMatching us/shot |
|---:|---:|---:|---:|---:|
| 13 | 0.00075 | 0.00075 | 820.46 | 81.12 |
| 15 | 0.00050 | 0.00050 | 1965.15 | 203.20 |

Interpretation: QECTOR Blossom matched PyMatching logical-error counts on this artifact. PyMatching remains much faster for standard MWPM latency on these workloads.

### Belief-matching accuracy experiment

Artifact: `benchmark_results/competitive_belief.json`

Environment: Windows x64, Python 3.11, QECTOR 0.5.0, PyMatching 2.4.0, Stim 1.16.0, 3,000 shots per distance.

| Distance | PyMatching LER | QECTOR MWPM LER | QECTOR Belief LER | Belief us/shot |
|---:|---:|---:|---:|---:|
| 3 | 0.01167 | 0.01167 | 0.01233 | 2331.07 |
| 5 | 0.00767 | 0.00767 | 0.00500 | 12125.38 |
| 7 | 0.00600 | 0.00600 | 0.00300 | 54323.56 |

Interpretation: belief-matching improved observed LER at d=5 and d=7 in this artifact but was dramatically slower. It should be positioned as an accuracy/research mode, not a production latency path.

### GPU bit-identity artifact

Artifact: `benchmark_results/gpu_extensive.json`

Environment: NVIDIA GeForce GTX 1660 Ti, Python 3.11, CUDA and OpenCL available, distances 3 to 13, batch sizes 1 to 65,536.

| Claim | Artifact result |
|---|---|
| Number of tested configurations | 36 |
| CUDA bit-identical to CPU | true |
| OpenCL bit-identical to CPU | true |
| All tested GPU paths faithful | true |

Interpretation: this is a correctness and reproducibility artifact for one machine. It is not a universal GPU speed claim.

### Native memory artifact

Artifact: `benchmark_results/native_memory.json`

Distance 13, batch 16,384:

| Decoder | RSS base MiB | RSS peak MiB | Native delta MiB |
|---|---:|---:|---:|
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
├── Rust core, proprietary
│   ├── Union-Find / Blossom / SparseBlossom engines
│   ├── CPU batch engine
│   ├── CUDA / OpenCL batch paths where enabled
│   ├── DEM collapse and Stim integration support
│   └── Native Python extension
│
└── Python layer, public in this repository
    ├── __init__.py
    ├── belief_matching.py
    ├── bposd.py
    ├── predecoder.py
    ├── backend.py
    ├── dem.py
    ├── stim_compat.py
    ├── sinter_compat.py
    ├── qiskit_plugin.py
    ├── rest_api.py
    ├── workbench.py
    └── codes.py
```

The Rust core is compiled into release wheels. The public `src/` directory is a placeholder/stub and is not enough for a standalone native source build. Commercial builds use the licensed source bundle or trusted CI injection.

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

The REST API is for local experiments, partner review, or controlled internal deployments. Do not expose it publicly without authentication, TLS, authorization, logging, input limits, and rate limiting. See [`docs/SECURITY_DEPLOYMENT.md`](docs/SECURITY_DEPLOYMENT.md).

---

## Limits and claim boundaries

| Area | Boundary |
|---|---|
| MWPM latency | PyMatching remains the speed leader on standard surface-code MWPM workloads in the provided artifacts. |
| Belief-matching | Accuracy/research mode. It can improve observed LER on selected workloads but is much slower. |
| GPU performance | Correctness is artifact-backed for tested machines. Speedup is not universal. |
| OpenCL wheels | OpenCL support depends on build configuration and target environment. Confirm locally. |
| SparseBlossom | Near-optimal, not exact MWPM. Use `BlossomDecoder` for exact minimum-weight matching. |
| UnionFind | Fast approximate path; not a universal decoder for arbitrary graphs. |
| REST/gRPC/MCP surfaces | Not hardened as public SaaS without a separate deployment/security review. |

---

## Documentation map

| Document | Contents |
|---|---|
| [`docs/API_STABILITY.md`](docs/API_STABILITY.md) | Stable vs experimental API surface |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Benchmark and validation methodology |
| [`docs/CORRECTNESS_AUDIT.md`](docs/CORRECTNESS_AUDIT.md) | Correctness invariants |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | Reproduction commands |
| [`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md) | Claim checklist before publication |
| [`docs/SCALING.md`](docs/SCALING.md) | Scaling and backend notes |
| [`docs/BEYOND_PYMATCHING.md`](docs/BEYOND_PYMATCHING.md) | Positioning beyond PyMatching |
| [`docs/BENCHMARK_COMPETITIVE.md`](docs/BENCHMARK_COMPETITIVE.md) | Competitive benchmark methodology |
| [`docs/SECURITY_DEPLOYMENT.md`](docs/SECURITY_DEPLOYMENT.md) | Service deployment hardening |
| [`docs/SERVICE_API_SCHEMA.md`](docs/SERVICE_API_SCHEMA.md) | REST schema and limitations |
| [`docs/PLATFORM_ARTIFACT_ROADMAP.md`](docs/PLATFORM_ARTIFACT_ROADMAP.md) | CI, SBOM, wheel and artifact roadmap |
| [`INSTALL.md`](INSTALL.md) | Source build notes |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |
| [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | Current release notes |
| [`SECURITY.md`](SECURITY.md) | Vulnerability reporting |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guidelines |

---

## CI / CD

| Workflow | Purpose | Notes |
|---|---|---|
| `tests.yml` | Ruff, format check, advisory MyPy, Docker smoke, import smoke | Rust source stub means Docker/native build checks can be informational in this workflow. |
| `CI.yml` | Linux x86_64, Windows x64, macOS arm64 wheels, PyPI publish on `v*` tags | Rust core is injected from GitHub Actions secrets for trusted release builds. Public sdist upload is intentionally avoided for wheel-only releases. |

PyPI release uses OIDC Trusted Publisher and does not require a stored PyPI API token.

---

## Repository cleanup status

The public repository has been cleaned for external review. Internal roadmaps, raw debug dumps, stale GPU notes, old CI setup instructions, and unfinished GNN/legacy scripts were removed in the cleanup commit. Public-facing evidence is now concentrated in `benchmark_results/`, `artifacts/`, `docs/`, and the reproducible scripts under `scripts/`.

---

## Commercial model

QECTOR Decoder v3 is source-available.

| Use | License required |
|---|---|
| Personal, academic, educational, non-commercial research | Free under the repository license |
| Company use, commercial R&D, institutional funded work | Paid commercial license |
| SaaS, hosted API, OEM embedding, product integration, redistribution | Paid commercial license |
| Commercial benchmarking, paid consulting, revenue-linked work | Paid commercial license |

Website and commercial licensing: [https://www.qector.store](https://www.qector.store)

Commercial contact: admin@qector.store

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

[https://www.qector.store](https://www.qector.store) · admin@qector.store

</div>
