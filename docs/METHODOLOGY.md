# Benchmark Methodology

This document defines how QECTOR Decoder v3 benchmarks are produced, reported,
and bounded so they are reproducible under external review. The goal is not to
make broad marketing claims; the goal is to connect each claim to a command,
artifact, environment block, and safe interpretation.

Every number QECTOR reports should come from the harnesses in
`python/qector_decoder_v3/benchmarking.py` or the scripts under `scripts/`, then
be stored as JSON/CSV/PDF artifacts with enough metadata for review. This file
explains how to measure. See `docs/REPRODUCE.md` for commands.

## 1. Benchmark objects

A benchmark result must define all of the following:

| Field | Required detail |
|---|---|
| Code family | repetition, ring, rotated surface, unrotated surface, toric, heavy-hex-style, LDPC/qLDPC, or DEM-derived |
| Distance / size | distance, rounds, checks, qubits, detectors, or code parameters |
| Noise model | physical error rate and noise channel/circuit source |
| Decoder | exact decoder class and mode, including belief/weighted/batch/GPU flags |
| Sample count | trials, shots, warmup count, and seed where applicable |
| Metric | correctness, LER, latency, throughput, memory, GPU bit-identity, or scaling |
| Environment | OS, CPU, RAM, Python/Rust/package versions, GPU/runtime availability, git commit |
| Artifact | raw JSON/CSV path plus SHA-256 hash |

A benchmark without this information is a smoke test, not public evidence.

## 2. Hot path vs cold path

A decoder can look fast if only the inner loop is timed. The harness reports both:

| Path | What is timed | Field in output | Safe use |
|---|---|---|---|
| Cold | Decoder construction: graph build, weight preprocessing, allocation | `cold_path_us` | Setup cost, repeated rebuild workflows |
| Hot | `decode()` on a pre-built decoder with syndromes already in memory | `latency_us` | Prebuilt decoder, repeated decode workflows |

Reporting only the hot path is acceptable only for a clearly labelled “prebuilt
decoder, repeated decode” workload. Do not use hot-path-only results to claim
end-to-end superiority.

## 3. Statistics and tail latency

For latency measurements, report the distribution, not only the mean:

- `n`, `mean`, `median`, `std`, `min`, `max`
- `p50`, `p90`, `p95`, `p99`
- confidence interval fields when produced by the script

Percentiles use the benchmark harness implementation. Throughput should be
computed from the same timing basis that is reported, usually median hot-path
latency for repeated decode workloads.

Safe wording:

```text
On this machine, for this code/distance/noise/seed/batch configuration, decoder
A had median hot-path latency X and p99 Y.
```

Unsafe wording:

```text
Decoder A is universally faster.
```

## 4. Logical error rate methodology

For circuit-level Stim workflows, LER is measured from detector samples and
observable outcomes. A correction is successful when the predicted observable
flip matches the sampled observable outcome. Do **not** score QEC decoders using
`correction != sampled_error`; degenerate stabilizer shifts can change the raw
correction without changing the logical outcome.

LER reports must include:

| Required item | Reason |
|---|---|
| Circuit generator or `.stim` file | Defines the workload |
| Detector error model settings | `decompose_errors`, graphlike collapse, raw/collapsed mechanism counts |
| Distance and rounds | Required for surface-code claims |
| Physical noise value | Required for all LER comparisons |
| Shots and seed | Required for reproducibility |
| Logical errors and LER | Core result |
| Confidence interval | Shows sampling uncertainty |
| PyMatching/reference version | Required for comparisons |

For the checked-in v0.5 reference artifacts, the d=13/d=15 LER parity audit is
stored in `benchmark_results/stim_ler_d13_d15.json`, and the selected
belief-matching comparison is stored in `benchmark_results/competitive_belief.json`.

## 5. Reproducibility controls

Each run records, and should be determined by:

- Seed — all sampled errors/syndromes should come from a named RNG seed where the
  script supports it.
- Warmup count — untimed iterations before measurement.
- Timed iteration count or shot count.
- Physical error rate and noise model.
- Reachable syndromes when testing decoders directly: `s = H·e mod 2`, not
  arbitrary bit patterns unless the test is explicitly adversarial.
- Environment block, including CPU model, logical core count, RAM, OS, Python,
  NumPy, SciPy, PyMatching, Stim, Rust/Cargo versions, GPU/runtime availability,
  and git commit.
- Raw artifact path and SHA-256 hash.

## 6. Correctness gate

Benchmarks are invalid if the decoder is wrong. Before timing, the harness must
verify the faithfulness condition where applicable:

```text
H · decode(s) == s   (mod 2)
```

A run with `syndrome_faithful == false` must be discarded. Full correctness
validation, including brute-force optimality, property-based faithfulness,
PyMatching cross-validation, and CPU/GPU bit-identity tests, is summarized in
`docs/CORRECTNESS_AUDIT.md`.

## 7. Competitor comparison: PyMatching

PyMatching remains the baseline for exact MWPM latency on common surface-code
matching workloads. QECTOR comparisons must therefore distinguish:

| Comparison type | Correct interpretation |
|---|---|
| LER parity | QECTOR weighted Blossom and PyMatching produced the same LER on the tested sampled workload |
| Matching weight | Corrections may differ while having equal weight and valid syndrome faithfulness |
| Latency | PyMatching may be faster even when LER is equal |
| Belief-matching | Accuracy mode on selected correlated workloads, not a PyMatching latency replacement |

Allowed public claim:

```text
QECTOR weighted Blossom reaches PyMatching LER parity on the checked-in tested
Stim workloads through d=15, while PyMatching remains the latency leader.
```

Disallowed public claim:

```text
QECTOR replaces PyMatching as the fastest MWPM decoder.
```

## 8. GPU methodology

GPU evidence is split into two different claim types:

| Claim type | Required proof |
|---|---|
| Correctness / bit-identity | Focused CUDA/OpenCL tests comparing GPU batch outputs to CPU reference outputs |
| Throughput / speedup | Local hardware benchmark with batch size, GPU model, driver/runtime versions, CPU baseline, latency distribution, and raw artifact hash |

Current public-safe GPU claim:

```text
CUDA/OpenCL batch paths are bit-identical to the CPU reference on tested
configurations.
```

Not allowed without dedicated local artifacts:

```text
GPU is universally faster.
QECTOR is 2x faster on RTX hardware.
QECTOR is production real-time hardware decoding infrastructure.
```

## 9. Memory methodology

Peak Python-side allocation per run is measured with `tracemalloc` where the
harness supports it. When `psutil` is installed, total/available RAM is recorded
in the environment block. Native Rust and GPU memory require backend diagnostics
or external vendor tools and should be labelled separately.

Memory reports must not mix Python allocation, process RSS, native heap, and VRAM
as if they were the same metric.

## 10. Safe vs unsafe public claims

| Topic | Safe wording | Unsafe wording |
|---|---|---|
| Tests | “v0.5 local validation report: 832 Python tests and 87 Rust tests passed” | “Every future environment is guaranteed green” |
| MWPM | “LER parity with PyMatching on checked-in tested workloads through d=15” | “Universally better than PyMatching” |
| Belief-matching | “Lower observed LER on selected correlated d=5/d=7 workloads” | “Always more accurate” |
| Latency | “Measured median/p99 on this machine and workload” | “Fastest decoder” |
| GPU | “CPU/GPU bit-identical on tested configs” | “Universal GPU speedup” |
| Workbench/API | “Local/demo or roadmap unless separately contracted” | “Enterprise SaaS / OEM ready” |
| Hardware | “Simulation/offline R&D platform” | “Real-time hardware QEC control stack” |

## 11. What to report in papers, README, website, and sales material

Always include:

1. Code/circuit/workload.
2. Decoder and mode.
3. Distance/rounds/noise/shots/seed.
4. Hot vs cold path if timing is discussed.
5. Median and p99 latency if latency is discussed.
6. Logical errors, LER, and confidence interval if LER is discussed.
7. Environment block.
8. Artifact hash.
9. Exact git revision and package version.
10. Claim boundary and known limitation.

## 12. Reviewer route

For external review, start here:

1. `docs/REPRODUCE.md`
2. `docs/REPRODUCIBILITY_CHECKLIST.md`
3. `docs/CORRECTNESS_AUDIT.md`
4. `BENCHMARK_GPU.md`
5. Checked-in JSON artifacts under `benchmark_results/`

A result is only strong when a reviewer can rerun it, inspect the artifact,
verify the hash, and see exactly which claim is supported.