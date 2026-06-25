# Correctness Audit

QECTOR correctness is tested continuously, not asserted once. This document
states the decoder contracts, the test coverage behind each contract, and the
known limits that must stay attached to public claims.

## 1. Core invariant: syndrome faithfulness

For every reachable syndrome `s = H·e (mod 2)`, every production decoder path
must return a correction `c = decode(s)` satisfying:

```text
H · c == s   (mod 2)
```

This is the primary correctness gate. Shape-compatible output is not enough.
The returned correction must reproduce the input syndrome under the same parity
check matrix.

## 2. Decoder contract table

| Decoder / path | Required contract | Validation style | Claim boundary |
|---|---|---|---|
| `BlossomDecoder` | Syndrome-faithful and exact minimum-weight on audited small matching codes | Exhaustive brute-force oracle + PyMatching cross-checks | Exact MWPM on the tested audited graph families; do not claim universal hardware layout equivalence |
| `SparseBlossomDecoder` | Syndrome-faithful and near-optimal | Brute-force small-code audit + PyMatching-compatible matching-graph tests | Region-growing decoder, not exact MWPM by design |
| `UnionFindDecoder` | Syndrome-faithful on supported QEC matching graphs | Exhaustive small-code tests + family tests + property tests | Fast approximate decoder; not minimum-weight and not guaranteed on arbitrary adversarial hypergraphs |
| `FastUnionFindDecoder` | Same as Union-Find, lower-overhead path | Cross-decoder faithfulness tests | Fast approximate decoder, not exact MWPM |
| `LookupTableDecoder` | Exact/faithful on stored table entries; faithful fallback on larger cases | Exhaustive table tests + d=5 fallback test | Table size and fallback behavior must be stated |
| `BPOSDDecoder` / `BpOsdDecoder` | Syndrome-faithful on LDPC/qLDPC-style CSS checks | BP-OSD reference-package comparison + CSS commutation tests | Quote LER from the harness, not from faithfulness alone |
| `BeliefMatching` | Lower observed LER on selected correlated workloads | Seeded Stim/PyMatching comparison tests and benchmark artifacts | Accuracy mode, not fast path; selected workload only |
| `CPUBatchDecoder` | Batch output equals per-shot CPU reference | Batch-vs-single decode tests | CPU batch workflow claim only |
| `CUDABatchDecoder` | Bit-identical to CPU reference on tested configs | Focused CUDA bit-identity tests | No universal GPU speed claim |
| `OpenCLBatchDecoder` | Bit-identical to CPU reference on tested configs | Focused OpenCL bit-identity tests | No universal GPU speed claim |
| `SlidingWindowDecoder` / `StreamingDecoder` | Windowed/streamed correction remains faithful on tested workflows | Streaming/window tests | Simulation workflow, not real-time hardware control |
| GNN / Hybrid paths | Experimental faithfulness and integration checks | Unit and smoke tests | Research/experimental only unless backed by a specific artifact |

## 3. Test coverage summary

The v0.5 local validation report records:

| Suite | Scope | Result in v0.5 local report |
|---|---|---|
| Python test suite | Python API, CLI, examples, decoder contracts, DEM/Stim paths, BP-OSD, GPU bit-identity, Workbench/backend tests | 832 passed, 0 skipped, 0 xfailed |
| Rust unit tests | Core Rust modules and backend internals | 87 passed |
| Total | Python + Rust | 919 passed, 0 deferred |

## 3a. Independent PyPI validation (v0.5.1 package, 2026-06-24)

86/87 automated checks passed across a primary 20k-shot run and a 5× re-test at
100k shots/pt (seed 777). Platform: Windows 10, AMD Ryzen 16-core, NVIDIA GTX
1660 Ti CUDA 7.5, Python 3.11.0, NumPy 2.2.6, PyMatching 2.4.0, stim/sinter 1.16.0.
Full artifact: `benchmark_results/validation_v051.json`.

### Structural validation (17 code constructions, Suite B)

All 17 code families validate: `repetition_code(d=3–11)`, `ring_code(d=3–7)`,
`rotated_surface_code(d=3–7)`, `unrotated_surface_code(d=3–5)`, `toric_code(d=3–5)`,
`heavy_hex_code(d=3–5)`. Every construction passes `n_qubits`, `n_checks`, `rank(H)`,
`k`, `d`, and matching-graph shape checks.

### Single-syndrome correctness (Suite C) — 100% syndrome-valid across all 23 entries

| Decoder | Code | µs/decode | Valid |
|---|---|---|---|
| UnionFindDecoder | rep d=5 | 9.3 | 100% |
| FastUnionFindDecoder | rep d=5 | 9.5 | 100% |
| BlossomDecoder | rep d=5 | 10.6 | 100% |
| SparseBlossomDecoder | rep d=5 | 11.8 | 100% |
| CPUBatchDecoder | rep d=5 | 11.2 | 100% |
| LookupTableDecoder | rep d=5 | 8.7 | 100% |
| UnionFindDecoder | rep d=9 | 10.0 | 100% |
| FastUnionFindDecoder | rep d=9 | 10.2 | 100% |
| BlossomDecoder | rep d=9 | 10.6 | 100% |
| SparseBlossomDecoder | rep d=9 | 10.6 | 100% |
| CPUBatchDecoder | rep d=9 | 9.7 | 100% |
| LookupTableDecoder | rep d=9 | 10.7 | 100% |
| UnionFindDecoder | surf d=3 | 12.2 | 100% |
| FastUnionFindDecoder | surf d=3 | 11.4 | 100% |
| BlossomDecoder | surf d=3 | 14.8 | 100% |
| SparseBlossomDecoder | surf d=3 | 11.5 | 100% |
| CPUBatchDecoder | surf d=3 | 9.5 | 100% |
| LookupTableDecoder | surf d=3 | 9.5 | 100% |
| UnionFindDecoder | surf d=5 | 10.1 | 100% |
| FastUnionFindDecoder | surf d=5 | 12.1 | 100% |
| BlossomDecoder | surf d=5 | 16.8 | 100% |
| SparseBlossomDecoder | surf d=5 | 29.2 | 100% |
| CPUBatchDecoder | surf d=5 | 10.7 | 100% |

### Workbench latency (Suite I — repetition d=5, Blossom, 1000 trials)

throughput: 277,778 dec/s · p50 3.60 µs · p90 5.70 µs · p99 11.61 µs · max 46.8 µs · syndrome_faithful: True

### GPU validation (Suite J — GTX 1660 Ti, 100k shots)

| Code | CUDA speedup vs CPU batch | GPU valid | CPU-agreeing |
|---|---|---|---|
| repetition d=9 | 7.67× | ✅ | ✅ |
| rotated_surface d=5 | 6.93× | ✅ | ✅ |

Representative Python coverage:

| Test file | What it proves |
|---|---|
| `test_syndrome_faithfulness.py` | Faithfulness across rotated/unrotated surface, ring, repetition, UF, FastUF, Blossom, SparseBlossom, BP-OSD, and CPU/GPU batch references |
| `test_property_faithfulness.py` | Hypothesis-generated random matching graphs plus adversarial dense, all-zero, and single-defect cases |
| `test_brute_force_small.py` | Exhaustive ground-truth minimum-weight table for small codes |
| `test_codes.py` | Built-in code-family validity and live decoder faithfulness |
| `test_dem.py` | Detector-error-model graph decoding |
| `test_pymatching_compat.py` | Cross-validation against the reference `pymatching` package |
| `test_pymatching_parity_*` | LER parity paths against PyMatching on selected workloads |
| `test_belief_matching.py` | Seeded belief-matching comparison on selected correlated workloads |
| `test_bposd_ldpc.py` | BP-OSD / LDPC reference checks and CSS commutation |
| `test_cuda_cpu_bit_identical.py` | CUDA batch output equals CPU reference on tested configurations |
| `test_opencl_cpu_bit_identical.py` | OpenCL batch output equals CPU reference on tested configurations |
| `test_reproduce_commands.py` | Reproduction scripts and help paths stay executable |
| `test_cli_smoke.py` | Public CLI/module entry points stay usable |

## 4. Exactness and optimality audit

Established by exhaustive brute-force on small codes and PyMatching-compatible
cross-validation:

- `BlossomDecoder` is exact MWPM on every reachable syndrome of the audited
  small matching codes. Its correction weight equals the brute-force minimum in
  those tests.
- `SparseBlossomDecoder` is faithful and near-optimal on audited small codes.
  It can return a weight `+1` correction on rare boundary-pairing cases, so it
  must not be marketed as exact MWPM.
- Equal-weight corrections may differ from PyMatching because degenerate quantum
  codes have multiple valid corrections in the same or different logical cosets.
  Correctness is measured by syndrome faithfulness and logical-observable outcome,
  not by byte-for-byte equality with PyMatching corrections.

## 5. LER and logical correctness audit

Logical-error-rate validation must use the circuit/model observables, not the
naive condition `correction != sampled_error`. Degenerate stabilizer shifts can
change the raw correction while preserving the logical outcome.

Safe LER audit language:

```text
On the checked-in v0.5 Stim artifacts, QECTOR weighted Blossom matches
PyMatching LER on the tested d=13/d=15 workloads.
```

Safe belief-matching language:

```text
On the checked-in v0.5 selected correlated workload, QECTOR belief-matching
shows lower observed LER at d=5 and d=7 than PyMatching, with belief-matching
positioned as an accuracy mode rather than the latency path.
```

Unsafe language:

```text
QECTOR is universally more accurate than PyMatching.
QECTOR beats PyMatching on every distance/noise model.
Belief-matching is the fastest decoder path.
```

## 6. GPU bit-identity audit

GPU correctness is a bit-identity claim against the CPU reference on tested
configurations, not a universal speed claim.

Focused commands after building the optional backend:

```powershell
# CUDA
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features --features cuda
.\.venv\Scripts\python.exe -m pytest python/tests/test_cuda_cpu_bit_identical.py -q --tb=short

# OpenCL
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features --features opencl
.\.venv\Scripts\python.exe -m pytest python/tests/test_opencl_cpu_bit_identical.py -q --tb=short
```

Allowed public claim:

```text
CUDA/OpenCL batch paths are bit-identical to the CPU reference on tested
configurations.
```

Disallowed public claim without a dedicated local throughput artifact:

```text
GPU is universally faster.
QECTOR has proven 2x GPU speedup.
QECTOR is real-time hardware decoding infrastructure.
```

## 7. Known limits, stated honestly

- `SparseBlossomDecoder` is region-growing and near-optimal, not exact MWPM.
  Use `BlossomDecoder` for exact minimum-weight audits.
- `UnionFindDecoder` and `FastUnionFindDecoder` are fast approximate decoders.
  They are validated for supported QEC matching graphs but are not minimum-weight
  algorithms and are not guaranteed on arbitrary adversarial hypergraphs.
- BP-OSD and GNN/hybrid paths are validated for faithfulness and integration, but
  their LER performance must be quoted from the LER harness and checked-in or
  regenerated artifacts.
- Code-family generators are matching-graph models for experimentation. A valid
  matching graph is not a claim of bit-exact hardware layout equivalence.
- REST/API paths are local/demo infrastructure unless a separate commercial
  agreement and production service controls exist.

## 8. Running the audit

PowerShell after the base install and optional test dependencies:

```powershell
.\.venv\Scripts\python.exe -m pytest python/tests -q --tb=short
.\.venv\Scripts\python.exe -m pytest python/tests/test_brute_force_small.py -q --tb=short
.\.venv\Scripts\python.exe -m pytest python/tests/test_pymatching_compat.py -q --tb=short
.\.venv\Scripts\python.exe -m pytest python/tests/test_property_faithfulness.py -q --tb=short
cargo test --release --lib
```

Advanced optional packages:

```powershell
.\.venv\Scripts\python.exe -m pip install stim pymatching sinter ldpc beliefmatching
.\.venv\Scripts\python.exe -m pytest python/tests/test_belief_matching.py python/tests/test_bposd_ldpc.py python/tests/test_sinter.py -q --tb=short
```

## 9. Audit checklist for new claims

Before adding a new public claim to README, website, sales material, or a paper,
record:

1. Exact code family, distance, rounds, noise model, shots, and seed.
2. Decoder name and mode.
3. Whether the claim is correctness, LER, latency, throughput, memory, or GPU
   bit-identity.
4. Full command used to generate the artifact.
5. Environment block.
6. SHA-256 of raw artifact.
7. Whether PyMatching/Stim/reference packages were installed and their versions.
8. Whether the claim is a reference snapshot or regenerated local result.
9. The exact safe wording that scopes the claim.
10. The exact unsafe wording that must not be inferred.