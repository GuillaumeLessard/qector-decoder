# Reproducibility Quick Checklist

Use this one-page checklist before publishing a benchmark result, updating the
README/site, sending a commercial validation report, or comparing QECTOR against
PyMatching, Stim, BP-OSD, CUDA, or OpenCL.

## 1. Identify the claim

Write the claim in one sentence before running anything.

Examples of scoped claims:

```text
QECTOR weighted Blossom matches PyMatching LER on this d=15 Stim workload.
QECTOR belief-matching shows lower observed LER on this selected d=5 correlated workload.
CUDA batch output is bit-identical to CPU output on this tested configuration.
```

Reject broad claims such as:

```text
QECTOR is universally faster than PyMatching.
QECTOR is always more accurate.
QECTOR is production real-time QEC hardware infrastructure.
```

## 2. Capture the repository state

PowerShell:

```powershell
git rev-parse HEAD
git status --short
```

Git Bash:

```bash
git rev-parse HEAD
git status --short
```

The working tree should be clean for public evidence. If it is not clean,
record exactly which files were modified.

## 3. Capture the environment

PowerShell:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip list
rustc --version
cargo --version
```

Also record OS, CPU, RAM, GPU model, CUDA/OpenCL driver/runtime versions when
GPU claims are involved.

## 4. Run the correct command for the claim

| Claim | Command family |
|---|---|
| Install smoke | `maturin develop --release --no-default-features` then import test |
| Full Python validation | `.\.venv\Scripts\python.exe -m pytest python/tests -q --tb=short` |
| Rust validation | `cargo test --release --lib` |
| d=13/d=15 LER parity | `scripts\competitive_stim_ler.py --distances 13 15 --shots 20000` |
| Belief-matching selected workload | `scripts\competitive_belief_matching.py --distances 3 5 7 --shots 3000 --no-ref` |
| CUDA bit-identity | `test_cuda_cpu_bit_identical.py` after CUDA build |
| OpenCL bit-identity | `test_opencl_cpu_bit_identical.py` after OpenCL build |
| General benchmark | `scripts\run_competitive_benchmark.py` |

## 5. Save raw artifacts

Save raw JSON/CSV output under `benchmark_results/` with a name that includes the
claim or workload. Do not rely only on screenshots or console summaries.

Good names:

```text
benchmark_results/stim_ler_d13_d15_local.json
benchmark_results/competitive_belief_local.json
benchmark_results/gpu_batch_rtx4090_local.json
```

## 6. Hash the artifacts

PowerShell:

```powershell
Get-FileHash benchmark_results\YOUR_FILE.json -Algorithm SHA256
```

Git Bash:

```bash
sha256sum benchmark_results/YOUR_FILE.json
```

Include the hash in any report or sales attachment.

## 7. Check the artifact includes an environment block

The artifact should include:

- command used
- Python version
- QECTOR version
- git commit if available
- PyMatching / Stim / SciPy versions when used
- OS/platform
- CPU and RAM
- CUDA/OpenCL availability when relevant

If the artifact has no environment block, regenerate it before using it as public
evidence.

## 8. Attach safe wording

Every result needs a safe wording line.

Example:

```text
On this checked-in v0.5 artifact, QECTOR weighted Blossom and PyMatching produced
identical LER on the tested d=15 Stim workload; PyMatching remained faster on
latency.
```

## 9. Attach unsafe wording to avoid

Every report should also state what the result does **not** prove.

Example:

```text
This does not prove universal speed superiority, real-time hardware readiness,
SaaS readiness, OEM readiness, or accuracy on untested codes/noise models.
```

## 10. Final release gate

Before publishing, confirm:

- [ ] The command is reproducible from a clean clone.
- [ ] The working tree state is recorded.
- [ ] The environment is recorded.
- [ ] Raw JSON/CSV artifacts are saved.
- [ ] SHA-256 hashes are recorded.
- [ ] The claim is scoped to the measured workload.
- [ ] Confidence intervals or sampling uncertainty are present for LER claims.
- [ ] Hot/cold path and p50/p99 are present for latency claims.
- [ ] GPU claims separate bit-identity from throughput.
- [ ] No claim implies real-time hardware, SaaS, or OEM readiness unless a separate
      production artifact and agreement exist.

## Related documents

- `docs/REPRODUCE.md` — detailed commands
- `docs/METHODOLOGY.md` — measurement rules
- `docs/CORRECTNESS_AUDIT.md` — decoder contracts and known limits
- `BENCHMARK_GPU.md` — GPU claim boundaries
- `benchmark_results/` — checked-in reference artifacts
