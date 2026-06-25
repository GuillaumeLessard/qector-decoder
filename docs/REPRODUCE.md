# Reproducing QECTOR Benchmarks

This page is the reviewer path for QECTOR Decoder v3. It shows how to build the
live public repository, verify the package imports, install optional comparison
packages, rerun the headline evidence, and separate safe claims from hardware-
specific claims.

Reference benchmark artifacts are checked in under `benchmark_results/` for
review. They are evidence snapshots, not universal performance claims. Regenerate
locally before quoting throughput, latency, memory, GPU speedup, SaaS readiness,
OEM readiness, or real-time hardware behavior.

## 0. Claim-to-command map

| Claim | Evidence file | Reproduction command |
|---|---|---|
| Base package builds on Windows/Python 3.11 | local install smoke | Section 1 |
| Full local validation: 832 Python tests | pytest suite | Section 3 |
| Rust unit validation: 87 tests | cargo test | Section 3 |
| d=13/d=15 Stim LER parity vs PyMatching | `benchmark_results/stim_ler_d13_d15.json` | Section 5 |
| Belief-matching selected d=5/d=7 lower observed LER | `benchmark_results/competitive_belief.json` | Section 6 |
| CUDA/OpenCL output bit-identity against CPU | GPU bit-identity tests | Section 7 |
| Artifact hash / environment snapshot discipline | JSON artifacts + hash commands | Section 8 |
| Independent PyPI validation (87/87 checks, 100k shots) | `benchmark_results/results_v053_retest.json` | Section 9 |

## 1. Install the public repository

The current public repository does **not** include `install.py`. Build the
Rust/Python extension with `maturin`.

### Windows PowerShell

Install Rust first from <https://rustup.rs/>. Then run:

```powershell
git clone https://github.com/GuillaumeLessard/qector-decoder.git
cd qector-decoder

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip maturin

$env:PYO3_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features

.\.venv\Scripts\python.exe -c "from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')"
```

Expected final smoke output:

```text
QECTOR OK
```

Messages such as `Ignoring stim`, `Ignoring pymatching`, or `Ignoring pytest`
are normal during the base install. They only mean optional extras were not
installed for the CPU-safe runtime build.

### Git Bash on Windows

```bash
git clone https://github.com/GuillaumeLessard/qector-decoder.git
cd qector-decoder

python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip maturin

export PYO3_PYTHON="$(pwd -W)/.venv/Scripts/python.exe"
python -m maturin develop --release --no-default-features

python -c "from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')"
```

## 2. Install comparison, benchmark, and test dependencies

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install stim sinter pymatching ldpc beliefmatching scipy psutil matplotlib tabulate pytest hypothesis fastapi uvicorn httpx
```

Git Bash after activating the virtual environment:

```bash
python -m pip install stim sinter pymatching ldpc beliefmatching scipy psutil matplotlib tabulate pytest hypothesis fastapi uvicorn httpx
```

## 3. Verify the validation suite

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest python/tests -q --tb=short
cargo test --release --lib
```

Git Bash:

```bash
python -m pytest python/tests -q --tb=short
cargo test --release --lib
```

The v0.5 local validation report records **832 Python tests passed** and
**87 Rust unit tests passed**. Treat those as the release report numbers until
your own local run finishes. If your environment lacks optional packages, GPU
runtimes, or compiler features, rerun the relevant focused tests after installing
those dependencies.

## 4. Run the reproducible general benchmark harness

```bash
python scripts/run_competitive_benchmark.py \
    --code rotated_surface \
    --distances 3 5 7 9 \
    --decoders union_find blossom sparse_blossom \
    --trials 5000 --warmup 500 \
    --error-rate 0.08 --seed 1 \
    --out benchmark_results/competitive_local
```

This writes:

- `benchmark_results/competitive_local.json` — full results plus environment block.
- `benchmark_results/competitive_local.csv` — one row per decoder/distance with latency,
  throughput, memory, and PyMatching cross-check data.

Available `--code` values include `repetition`, `ring`, `rotated_surface`,
`unrotated_surface`, `toric`, and `heavy_hex`.

## 5. Reproduce the d=13/d=15 Stim LER parity audit

The checked-in reference file is `benchmark_results/stim_ler_d13_d15.json`.
It records Stim rotated-surface-code memory workloads at d=13 and d=15, with
QECTOR weighted Blossom and PyMatching producing identical logical error rates
on the sampled workloads.

PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\competitive_stim_ler.py --distances 13 15 --shots 20000 --out benchmark_results\stim_ler_d13_d15_local
```

Git Bash:

```bash
python scripts/competitive_stim_ler.py --distances 13 15 --shots 20000 --out benchmark_results/stim_ler_d13_d15_local
```

Report these numbers only with the distance, rounds, noise, shot count, seed if
applicable, git revision, Python/Rust/package versions, and hardware environment
block. Do not turn this into a universal PyMatching speed claim; the checked-in
artifact shows LER parity, while PyMatching remains the faster exact-MWPM latency
baseline on these workloads.

## 6. Reproduce the belief-matching selected-workload audit

The checked-in reference file is `benchmark_results/competitive_belief.json`.
It records the selected 3,000-shot correlated workload used for the public
belief-matching claim.

PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\competitive_belief_matching.py --distances 3 5 7 --shots 3000 --no-ref --out benchmark_results\competitive_belief_local
```

Git Bash:

```bash
python scripts/competitive_belief_matching.py --distances 3 5 7 --shots 3000 --no-ref --out benchmark_results/competitive_belief_local
```

Safe wording:

```text
QECTOR belief-matching shows lower observed LER than PyMatching on selected
correlated workloads at d=5 and d=7 in the checked-in v0.5 artifacts.
```

Unsafe wording:

```text
QECTOR is universally more accurate than PyMatching.
QECTOR belief-matching is the fast path.
QECTOR beats PyMatching on every code, distance, and noise model.
```

## 7. Verify CPU/GPU bit-identity

Build the optional backend first, then run the focused tests.

CUDA:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features --features cuda
.\.venv\Scripts\python.exe -m pytest python/tests/test_cuda_cpu_bit_identical.py -q --tb=short
```

OpenCL:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features --features opencl
.\.venv\Scripts\python.exe -m pytest python/tests/test_opencl_cpu_bit_identical.py -q --tb=short
```

Safe wording:

```text
CUDA/OpenCL batch paths are bit-identical to the CPU reference on tested
configurations.
```

Unsafe wording:

```text
GPU is universally faster.
QECTOR is 2x faster on RTX hardware.
QECTOR is real-time hardware decoding infrastructure.
```

## 8. Inspect artifact hashes and environment blocks

PowerShell hash check:

```powershell
Get-FileHash benchmark_results\competitive_belief.json -Algorithm SHA256
Get-FileHash benchmark_results\stim_ler_d13_d15.json -Algorithm SHA256
Get-FileHash benchmark_results\native_memory.json -Algorithm SHA256
```

Git Bash hash check:

```bash
sha256sum benchmark_results/competitive_belief.json
sha256sum benchmark_results/stim_ler_d13_d15.json
sha256sum benchmark_results/native_memory.json
```

Every JSON benchmark artifact should include an `environment` object containing
at least Python version, platform, package versions, git commit where captured,
and the command used to create the artifact. If the environment block is missing,
do not use the file as public evidence.

## 9. Programmatic single-configuration inspection

```python
from qector_decoder_v3 import codes, benchmarking as bm

code = codes.rotated_surface_code(7)
report = bm.benchmark_decoder("blossom", code, n_trials=5000, warmup=500, seed=1)
print(report["latency_us"]["p99"], "us p99")
print(report["cold_path_us"]["median"], "us construction")
print(bm.capture_environment())
```

## 10. CI artifacts

`.github/workflows/tests.yml` defines Rust tests, Python tests across Linux,
Windows, and macOS, plus a benchmark smoke job. Treat CI as an additional gate,
not a replacement for local reproduction when making hardware-specific claims.

## 9. Independent PyPI validation artifact

The file `benchmark_results/results_v053_retest.json` is the machine-readable record of
the 2026-06-25 independent validation run (PyPI install, isolated venv, Windows 10,
AMD Ryzen 16-core, NVIDIA GTX 1660 Ti CUDA 7.5, Python 3.11, PyMatching 2.4.0,
stim 1.16.0). It contains:

- Full environment snapshot
- All 17 code structural validation records
- All 23 single-syndrome correctness/latency entries (every decoder × code combo)
- Repetition-code LER table (d=3–9, 100k shots, seed 777) with Wilson 95% CIs
- Rotated-surface LER table (d=3–7, 100k shots) with Blossom–PyMatching delta
- CPU batch throughput by decoder for repetition d=9
- CUDA vs CPU speedup at 100k shots (7.67× rep-d9, 6.93× surf-d5)
- Workbench latency percentiles (p50 3.50 µs, p90 5.20 µs, p99 9.50 µs)
- Findings summary (F-1 through F-5) with status

PowerShell hash check:

```powershell
Get-FileHash benchmark_results\results_v053_retest.json -Algorithm SHA256
```

Git Bash hash check:

```bash
sha256sum benchmark_results/results_v053_retest.json
```

## 11. Short reviewer checklist

For a complete external review, capture:

1. Git commit and `git status --short`.
2. Python version and package versions.
3. Rust version and Cargo version.
4. Exact command used.
5. Raw JSON/CSV artifact.
6. SHA-256 hash of the artifact.
7. Hardware/OS environment block.
8. The exact claim being made.
9. Whether the claim is LER, latency, throughput, memory, or correctness.
10. Whether the claim is a reference snapshot or a regenerated local result.

See `docs/REPRODUCIBILITY_CHECKLIST.md` for the one-page version.