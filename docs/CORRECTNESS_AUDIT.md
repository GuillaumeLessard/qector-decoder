# Correctness Audit — v0.5.7

QECTOR correctness is tested continuously, not asserted once. This document
states the decoder contracts, the test coverage behind each contract, and the
known limits that must stay attached to public claims.

## 1. Core invariant: syndrome faithfulness

For every reachable syndrome `s = H·e (mod 2)`, every production decoder path
must return a correction `c = decode(s)` satisfying:

```text
H · c == s   (mod 2)
```

Shape-compatible output is not enough. The returned correction must reproduce the
input syndrome under the same parity check matrix.

## 2. Decoder contract table

| Decoder / path | Required contract | Validation style | Claim boundary |
|---|---|---|---|
| `BlossomDecoder` | Syndrome-faithful and exact minimum-weight on audited small matching codes | Exhaustive brute-force oracle + PyMatching cross-checks | Exact MWPM on tested audited graph families; do not claim universal hardware layout equivalence |
| `SparseBlossomDecoder` | Syndrome-faithful and near-optimal | Brute-force small-code audit + PyMatching-compatible matching-graph tests | Region-growing decoder, not exact MWPM by design |
| `UnionFindDecoder` | Syndrome-faithful on supported QEC matching graphs | Exhaustive small-code tests + family tests + property tests | Fast approximate decoder; not minimum-weight |
| `FastUnionFindDecoder` | Same as Union-Find, lower-overhead path | Cross-decoder faithfulness tests | Fast approximate decoder, not exact MWPM |
| `LookupTableDecoder` | Exact/faithful on stored table entries; faithful fallback on larger cases | Exhaustive table tests + d=5 fallback test | Table size and fallback behaviour must be stated |
| `PredecodedDecoder` | `.batch_decode(syndromes)` returns faithful corrections via the selected backend | Faithfulness tests (v0.5.5+) | Constructor: `(check_to_qubits, n_qubits, backend)` |
| `BPOSDDecoder` / `BpOsdDecoder` | Syndrome-faithful on LDPC/qLDPC-style CSS checks | BP-OSD reference-package comparison + CSS commutation tests | Quote LER from the harness, not from faithfulness alone |
| `BeliefMatching` | Lower observed LER on selected correlated workloads | Seeded Stim/PyMatching comparison tests and benchmark artifacts | Accuracy mode, not fast path; selected workload only; also accepts raw numpy H (v0.5.3+) |
| `CPUBatchDecoder` / `BatchDecoder` | Batch output equals per-shot CPU reference; `.decode()` single-shot works (v0.5.3+) | Batch-vs-single decode tests | CPU batch workflow claim only |
| `CUDABatchDecoder` | Bit-identical to CPU reference on tested configs | Focused CUDA bit-identity tests | No universal GPU speed claim; always call `is_available()` first |
| `OpenCLBatchDecoder` | Bit-identical to CPU reference on tested configs | Focused OpenCL bit-identity tests | No universal GPU speed claim; note AMD OCL SDK Light false-negative (v0.5.5) |
| `SlidingWindowDecoder` / `StreamingDecoder` | Windowed/streamed correction remains faithful on tested workflows | Streaming/window tests | Simulation workflow, not real-time hardware control |
| `NeuralPredecoder` | `predict()` / `decode()` faithful on any numpy; `train()` requires numpy<2.0 | Unit and smoke tests | Experimental; `train()` raises clear `RuntimeError` on numpy>=2.0 (v0.5.4+) |
| GNN / Hybrid paths | Experimental faithfulness and integration checks | Unit and smoke tests | Research/experimental only |

## 3. Test coverage summary

v0.5.5 local validation report:

| Suite | Scope | Result |
|---|---|---|
| Python test suite | Python API, CLI, examples, decoder contracts, DEM/Stim paths, BP-OSD, GPU bit-identity, Workbench/backend | 775 passed, 71 skipped, 0 failed |
| Rust unit tests | Core Rust modules and backend internals | 87 passed |

v0.5.7 PyPI wheel smoke test (19/19 PASS, 2026-06-29, Windows 10, Python 3.11,
NumPy 2.4.6, PyPI wheel `qector_decoder_v3-0.5.7-cp311-cp311-win_amd64.whl`):

| # | Check | Result |
|---|---|---|
| 1–2 | dist-info + `__version__` both == 0.5.7 | ✅ |
| 3–6 | UnionFind, FastUnionFind, Blossom, SparseBlossom `.decode()` | ✅ |
| 7–8 | BatchDecoder `.parallel_batch_decode()` and `.decode()` single-shot | ✅ |
| 9–10 | `CUDABatchDecoder.is_available()` callable; clean `RuntimeError` when no GPU | ✅ |
| 11–12 | `stim_compat` both entry points importable | ✅ |
| 13 | `stim_circuit_to_check_matrix` and `from_stim_detector_error_model` are separate `__code__` objects (parallel impls) | ✅ |
| 14 | `BeliefMatching(H, p=...)` raw numpy constructor | ✅ |
| 15–17 | `QectorSinterDecoder`, `QectorDecoderWrapper` alias, `qector_sinter_decoders()` guard | ✅ |
| 18 | `PredecodedDecoder(check_to_qubits, n_qubits, backend).batch_decode(syndromes)` shape `(8, 5)` | ✅ |
| 19 | `LookupTableDecoder.decode()` | ✅ |

## 3a. Independent PyPI validation (v0.5.3, 2026-06-25)

87/87 automated checks PASS across a primary 20k-shot run and a 5× re-test at
100k shots/pt (seed 777). Platform: Windows 10, AMD Ryzen 16-core, NVIDIA GTX
1660 Ti CUDA 7.5, Python 3.11.0, NumPy 2.2.6, PyMatching 2.4.0, stim 1.16.0.
Full artifact: `benchmark_results/results_v053_retest.json`.

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
| BlossomDecoder | rep d=9 | 10.6 | 100% |
| SparseBlossomDecoder | rep d=9 | 10.6 | 100% |
| UnionFindDecoder | surf d=3 | 12.2 | 100% |
| BlossomDecoder | surf d=3 | 14.8 | 100% |
| UnionFindDecoder | surf d=5 | 10.1 | 100% |
| BlossomDecoder | surf d=5 | 16.8 | 100% |
| SparseBlossomDecoder | surf d=5 | 29.2 | 100% |
| CPUBatchDecoder | surf d=5 | 10.7 | 100% |

### Workbench latency (repetition d=5, Blossom, 1000 trials)

285,713 dec/s · p50 3.50 µs · p90 5.20 µs · p99 9.50 µs · max 28.7 µs · syndrome_faithful: True

### GPU validation (GTX 1660 Ti, 100k shots)

| Code | CUDA speedup vs CPU batch | GPU valid | CPU-agreeing |
|---|---|---|---|
| repetition d=9 | 7.67× | ✅ | ✅ |
| rotated_surface d=5 | 6.93× | ✅ | ✅ |

## 4. Exactness and optimality audit

- `BlossomDecoder` is exact MWPM on every reachable syndrome of the audited small
  matching codes. Correction weight equals the brute-force minimum.
- `SparseBlossomDecoder` is faithful and near-optimal on audited small codes. It
  can return a weight-`+1` correction on rare boundary-pairing cases and must not
  be marketed as exact MWPM.
- Equal-weight corrections may differ from PyMatching because degenerate quantum
  codes have multiple valid corrections. Correctness is measured by syndrome
  faithfulness and logical-observable outcome, not byte-for-byte equality.

## 5. LER and logical correctness audit

Logical-error-rate validation must use the circuit/model observables, not the
naive condition `correction != sampled_error`. Degenerate stabilizer shifts can
change the raw correction while preserving the logical outcome.

Safe LER audit language:

```text
On the checked-in v0.5 Stim artifacts, QECTOR weighted Blossom matches
PyMatching LER on the tested d=13/d=15 workloads.
```

Unsafe language:

```text
QECTOR is universally more accurate than PyMatching.
QECTOR beats PyMatching on every distance/noise model.
```

## 6. GPU bit-identity audit

GPU correctness is a bit-identity claim against the CPU reference on tested
configurations, not a universal speed claim. See `docs/REPRODUCE.md` Section 7
for focused build and test commands.

## 7. Known limits

- `SparseBlossomDecoder` is region-growing and near-optimal, not exact MWPM.
- `UnionFindDecoder` and `FastUnionFindDecoder` are fast approximate decoders,
  not minimum-weight on all inputs.
- `NeuralPredecoder.train()` requires numpy<2.0 (Rust binding issue, v0.5.4+).
- `OpenCLBatchDecoder.is_available()` false-negative on AMD OCL SDK Light legacy
  runtime (documented v0.5.5, fix planned).
- Single-round code-capacity noise does not produce surface-code threshold curves.
  Use circuit-level Stim DEM with `qector_sinter_decoders()` for threshold work.
- GPU throughput speedups are hardware, batch-size, and configuration dependent.
  Always report with hardware model, driver/runtime, batch size, and artifact hash.
