# Contributing to QECTOR Decoder

QECTOR Decoder is source-available proprietary software. Contributions are welcome only under the terms below.

## Contribution model

This repository is not an OSI open-source project. Opening an issue, submitting a pull request, forking the repository, reviewing source, or reproducing benchmarks does **not** grant commercial rights.

Commercial use requires a written paid commercial license from Guillaume Lessard / iD01t Productions.

Contact:

```text
admin@qector.store
```

## What contributions are useful

Useful contributions include:

```text
Reproduction reports from clean clones
Bug reports with minimal failing examples
Benchmark artifacts with environment blocks and SHA-256 hashes
Documentation corrections
Windows / Linux / macOS installation feedback
Stim / PyMatching / BP-OSD compatibility reports
CI failure reports
Security disclosures sent privately
```

Do not submit:

```text
Unscoped performance marketing claims
Large rewrites without prior discussion
Public exploit proofs for security issues
Generated dependency churn without test evidence
License-removal or commercial-rights changes
```

## Legal terms for contributions

By submitting a pull request, patch, issue reproduction, benchmark artifact, or documentation change, you agree that:

```text
1. You have the right to submit the contribution.
2. The contribution may be used, modified, sublicensed, redistributed, or commercialized by Guillaume Lessard / iD01t Productions as part of QECTOR.
3. The contribution does not grant you commercial use rights to QECTOR.
4. The contribution does not change the source-available proprietary license.
5. You will not submit code copied from incompatible licenses.
```

If those terms do not work for you, open a discussion or send a private email before contributing.

## Development setup

Use the real source-build path. The repository currently does not ship `install.py`.

PowerShell:

```powershell
git clone https://github.com/GuillaumeLessard/qector-decoder.git
cd qector-decoder

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip maturin

$env:PYO3_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features

.\.venv\Scripts\python.exe -c "from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')"
```

Optional test dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install "pytest>=7" "hypothesis>=6" "fastapi>=0.110" "uvicorn>=0.29" "httpx>=0.27" stim pymatching sinter ldpc beliefmatching psutil matplotlib tabulate scipy
.\.venv\Scripts\python.exe -m pytest python/tests -q --tb=short
```

Rust tests:

```powershell
cargo test --release --lib
```

## Benchmark contribution rules

Before publishing or submitting benchmark results, follow:

```text
docs/REPRODUCIBILITY_CHECKLIST.md
docs/METHODOLOGY.md
docs/REPRODUCE.md
BENCHMARK_GPU.md
```

Benchmark contributions must include:

```text
command used
git commit
clean/dirty working tree state
OS / CPU / RAM / GPU
Python, Rust, Stim, PyMatching, SciPy, NumPy versions when relevant
raw JSON/CSV output
SHA-256 hash
safe wording
unsafe wording to avoid
```

Do not claim universal speed, universal accuracy, SaaS readiness, OEM readiness, or real-time hardware readiness from a single local benchmark.

## Stable vs experimental areas

Before extending an API, check:

```text
docs/API_SURFACES.md
```

Stable user-facing areas include the core Python package import path, the basic decoder classes, and the documented source-build install flow. Experimental areas include neural/hybrid modules, Workbench, REST, gRPC, MCP, metrics, and GPU throughput claims.

## Security issues

Do not open public issues for security vulnerabilities. Follow `SECURITY.md`.
