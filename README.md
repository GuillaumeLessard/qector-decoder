# QECTOR Decoder v3

[![CI](https://github.com/qectorlab/qector-decoder/actions/workflows/tests.yml/badge.svg)](https://github.com/qectorlab/qector-decoder/actions/workflows/tests.yml)

**QEC decoding for teams that need reproducible proof, not just a fast baseline.**

QECTOR is a source-available Rust/Python quantum-error-correction R&D platform.
It combines PyMatching-compatible validation, belief-matching accuracy
experiments, BP-OSD for LDPC/qLDPC workflows, CPU/GPU batch paths, and
artifact-backed benchmark evidence.

## Current release

| Item | Status |
|---|---|
| Package version | `0.5.0` |
| Rust package | `qector_decoder_v3` |
| Python package | `qector-decoder-v3` |
| Canonical repository | `https://github.com/qectorlab/qector-decoder` |
| License | Source-available, proprietary / commercial license required for commercial use |
| Commercial contact | `admin@qector.store` |

## Validated scope

The current v0.5 validation report and checked-in benchmark artifacts support the
following scoped claims:

- **832 Python tests passed** in the local validation report.
- **87 Rust unit tests passed** in the local validation report.
- **Exact-MWPM LER parity with PyMatching** on tested rotated-surface-code Stim
  workloads through **d=15**.
- **Belief-matching is an accuracy mode** with lower observed LER on selected
  correlated workloads; it is slower and should not be sold as the latency path.
- **BP-OSD / LDPC / qLDPC workflows** are supported and cross-checked against
  reference packages on selected workloads.
- **CUDA/OpenCL batch paths are tested for CPU bit-identity** on supported test
  configurations. Throughput is hardware-specific and must be regenerated before
  quoting sales or scientific speed claims.
- **gRPC optional stack migrated to tonic/prost 0.14** with `tonic-prost` and
  vendored `protoc` support for `--features grpc` / `--features full` builds.

CI workflow status is surfaced by the badge above, but public claims should still
cite the exact commit and run/artifact used for the claim.

## Evidence snapshot

Selected checked-in artifacts:

| Artifact | Purpose |
|---|---|
| `benchmark_results/competitive_belief.json` | PyMatching vs QECTOR MWPM vs QECTOR belief-matching on d=3/5/7 selected correlated workloads |
| `benchmark_results/belief_grid.json` | d=5 belief-matching seed/probability grid |
| `benchmark_results/stim_ler_d13_d15.json` | d=13/d=15 Stim LER parity audit against PyMatching |
| `benchmark_results/native_memory.json` | CPU/GPU memory profile reference artifact |
| `artifacts/reference_validation_manifest.json` | SHA-256 manifest for reference validation artifacts |
| `BENCHMARK_GPU.md` | GPU benchmark claim boundaries and reproduction notes |
| `docs/REPRODUCE.md` | Supported reproduction commands |
| `docs/REPRODUCIBILITY_CHECKLIST.md` | Pre-publication checklist for claims and artifacts |
| `docs/PLATFORM_ARTIFACT_ROADMAP.md` | CI, cross-platform benchmark, SBOM, and wheel roadmap |

These artifacts are **reference evidence**, not universal hardware claims. Regenerate
locally before quoting throughput, latency, GPU speedup, memory, or deployment
readiness.

## Install from source

The repository currently does **not** include `install.py`. Use the real source
build path below.

### Windows PowerShell

Rust must be installed first from <https://rustup.rs/>.

```powershell
git clone https://github.com/qectorlab/qector-decoder.git
cd qector-decoder

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip maturin

$env:PYO3_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features

.\.venv\Scripts\python.exe -c "from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')"
```

Verified Windows result from a fresh clone on a second PC:

```text
Installed qector-decoder-v3-0.5.0
QECTOR OK
```

Ignored optional-extra messages during the base install are normal. The public
command installs the minimal CPU-safe runtime build; install test, Stim, and
benchmark extras only when needed.

### Git Bash on Windows

```bash
git clone https://github.com/qectorlab/qector-decoder.git
cd qector-decoder

python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip maturin

export PYO3_PYTHON="$(pwd -W)/.venv/Scripts/python.exe"
python -m maturin develop --release --no-default-features

python -c "from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')"
```

### Optional development/test dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install "pytest>=7" "hypothesis>=6" "fastapi>=0.110" "uvicorn>=0.29" "httpx>=0.27" stim pymatching sinter ldpc beliefmatching psutil matplotlib tabulate scipy
.\.venv\Scripts\python.exe -m pytest python/tests -q --tb=short
```

### Optional GPU builds

The default public command uses `--no-default-features` for a CPU-safe build.
Build CUDA/OpenCL only when the local machine has the required drivers/toolchain.

```powershell
# CUDA only
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features --features cuda

# OpenCL only
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features --features opencl
```

## Quick API example

```python
import numpy as np
from qector_decoder_v3 import UnionFindDecoder, generate_ring_code_checks

checks, n_qubits = generate_ring_code_checks(5)
decoder = UnionFindDecoder(checks, n_qubits)
syndrome = np.zeros(len(checks), dtype=np.uint8)
correction = decoder.decode(syndrome)
print(correction)
```

## Decoder stack

| Decoder | Use | Stability |
|---|---|---|
| `UnionFindDecoder` / `FastUnionFindDecoder` | Fast approximate matching-graph path | Stable local API |
| `BlossomDecoder` | Exact weighted MWPM / PyMatching LER parity path | Stable local API |
| `SparseBlossomDecoder` | Region-growing near-optimal Blossom path, not exact MWPM | Supported, claim-scoped |
| `BeliefMatching` | Accuracy mode for selected correlated circuit-level workloads | Supported, workload-sensitive |
| `BPOSDDecoder` / `BpOsdDecoder` | BP-OSD for LDPC/qLDPC workflows | Supported, workload-sensitive |
| `SlidingWindowDecoder` / `StreamingDecoder` | Multi-round / streaming workflows | Supported, workload-sensitive |
| `CPUBatchDecoder` / `BatchDecoder` | CPU batch / Rayon workflows | Stable local API |
| `CUDABatchDecoder` / `OpenCLBatchDecoder` | GPU batch workflows with CPU fallback and bit-identity tests | Optional, hardware-sensitive |
| `AutoDecoder` | Calibrated CPU/GPU backend selection | Optional, calibration-sensitive |
| Hybrid/GNN components | Research/preview path | Experimental |
| REST/gRPC/MCP/metrics | Local/demo or partner-review service surfaces | Experimental / deployment-reviewed |

See `docs/API_STABILITY.md` for stable vs experimental API boundaries.

## Honest limitations

- PyMatching remains the latency leader for exact MWPM on common surface-code
  workloads. QECTOR's MWPM value is parity, source access, workflow packaging,
  and integration with the broader platform.
- Belief-matching is slower because it rebuilds weights per shot. It is an
  accuracy mode, not a fast path.
- Sparse Blossom is region-growing and near-optimal, but not exact MWPM. Use
  `BlossomDecoder` when exact minimum weight is required.
- GPU wins only at suitable batch sizes and hardware configurations. Use local
  calibration and benchmark artifacts before making speed claims.
- REST/gRPC/MCP/metrics are not enterprise SaaS surfaces unless reviewed,
  hardened, and covered by a separate commercial agreement.
- Linux/macOS benchmark artifacts and prebuilt wheels are roadmap items, not
  current public proof assets. See `docs/PLATFORM_ARTIFACT_ROADMAP.md`.

## Security and deployment

QECTOR is safest as a local Rust/Python library. Optional service and GPU
surfaces require extra review.

| Document | Use |
|---|---|
| `SECURITY.md` | Private vulnerability reporting, supported versions, scope |
| `docs/SECURITY_DEPLOYMENT.md` | REST/gRPC/MCP/Docker/GPU hardening and SBOM-lite commands |
| `docs/SERVICE_API_SCHEMA.md` | Current local REST schema and service limitations |
| `docs/API_STABILITY.md` | Stable local APIs vs experimental surfaces |

Do not expose REST/gRPC/MCP directly on a public network without authentication,
TLS, request limits, rate limits, audit logs, resource quotas, and deployment
review.

## Commercial model

QECTOR Decoder v3 is source-available for personal, academic, educational, and
non-commercial research use.

Commercial use requires a paid commercial license. Commercial use includes
company use, commercial R&D, government or institutional funded work, paid
consulting, SaaS, hosted API use, OEM embedding, product integration,
redistribution, internal business operations, commercial benchmarking, and
revenue-linked use.

See:

- `LICENSE`
- `COMMERCIAL.md`
- `SECURITY.md`
- `CONTRIBUTING.md`
- <https://www.qector.store>

Commercial contact: **admin@qector.store**

## Reproduction and docs

- `docs/REPRODUCE.md` — install and benchmark reproduction
- `docs/METHODOLOGY.md` — validation method
- `docs/CORRECTNESS_AUDIT.md` — correctness invariants
- `docs/REPRODUCIBILITY_CHECKLIST.md` — claim/audit checklist
- `docs/SCALING.md` — scaling notes
- `docs/BEYOND_PYMATCHING.md` — belief-matching and BP-OSD positioning
- `docs/API_STABILITY.md` — stable vs experimental API boundary
- `docs/SERVICE_API_SCHEMA.md` — REST schema and service limitations
- `docs/SECURITY_DEPLOYMENT.md` — security hardening and SBOM-lite commands
- `docs/PLATFORM_ARTIFACT_ROADMAP.md` — CI, cross-platform artifacts, SBOM, wheels
- `BENCHMARK_GPU.md` — GPU benchmark boundaries
- `RELEASE_NOTES.md` — v0.5.0 release notes

## CI

`.github/workflows/tests.yml` runs Rust tests, Python tests across Linux,
Windows, and macOS, benchmark smoke artifacts, Docker build, and advisory
ruff/mypy checks.

## License

**QECTOR Decoder Source-Available License v1.0** — see `LICENSE`.

Copyright © 2026 Guillaume Lessard / iD01t Productions. All rights reserved.
