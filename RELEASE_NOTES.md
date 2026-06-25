# QECTOR Decoder v0.5.3 — Release Notes

**Version**: 0.5.3
**Date**: 2026-06-25
**Codename**: Neutron-3

---

## What's new in 0.5.3

Hotfix release. Closes **all 4 open findings** from the post-0.5.2 PyPI
independent validation: version string regression (F-1) and the 3 API gaps
discovered during the 87-check re-test suite (fresh venv, Python 3.11, Windows 10).

### Fixed

- **`__version__` version string (F-1)** — Python `qector_decoder_v3.__version__`
  had drifted two consecutive patch releases (was `"0.5.0"` while dist-info and
  Rust core reported `0.5.2`). Now all three surfaces agree at `"0.5.3"`:
  `pyproject.toml [project] version`, `Cargo.toml [package] version`, and the
  Python fallback string. `importlib.metadata` derivation is used as the primary
  path so it can never drift again (fallback matches the compile-time literal).
- **`BatchDecoder.decode(syndrome)`** — single-shot `.decode()` method was absent;
  only `.parallel_batch_decode()` was exposed. Added `.decode()` as a 1-row batch
  wrapper with identical dtype/shape contract to every other decoder.
- **`BeliefMatching(H, p=...)`** — constructor accepted only a `_Matrices`
  dataclass (returned by `build_matching_matrices`). Now also accepts a raw numpy
  check matrix `H` of shape `(num_detectors, num_qubits)` with uniform prior `p`
  (default 0.1); observable matrix defaults to identity. All existing call sites
  using `BeliefMatching.from_detector_error_model(dem)` are unchanged.
- **`QectorSinterDecoder.decode(syndrome, dem)`** — Sinter wrappers
  `QectorSinterDecoder` / `QectorDecoderWrapper` exposed no `.decode()` method for
  standalone (non-Sinter) usage. Added single-syndrome decode with DEM caching:
  first call requires `dem=` kwarg, subsequent calls reuse the cached DEM.

### Validation result (v0.5.3)

**87/87 automated checks PASS** (primary 20k shots/pt, seed 12345;
re-test 100k shots/pt, seed 777). Tested as a clean PyPI wheel install.

| Suite | Checks | Result |
|---|---|---|
| A — Environment & version metadata | 1 | ✅ PASS (was FAIL in 0.5.2) |
| B — Code structural (17 families) | 17 | ✅ PASS |
| C — Single-syndrome correctness (23 combos) | 23 | ✅ PASS |
| D — Repetition code LER (d=3–9, 100k shots) | varies | ✅ PASS |
| E — CPU batch throughput & correctness | varies | ✅ PASS |
| F — Edge cases / robustness | 6 | ✅ PASS |
| G — Specialized decoders | 8 | ✅ PASS |
| H — Compat layers (pymatching, stim) | 3 | ✅ PASS |
| I — Workbench end-to-end (JSON/CSV/PDF) | 5 | ✅ PASS |
| J — CUDA GPU correctness + speedup | 2 | ✅ PASS |

---

**Version**: 0.5.2
**Date**: 2026-06-25  
**Codename**: Neutron-2

---

## What's new in 0.5.2

This patch release ships the independent validation data from the 86/87-check PyPI
audit and closes the remaining cosmetic and operational findings from that report.

### Fixed
- **`__version__` bump to `0.5.2`** — Python `qector_decoder_v3.__version__` now
  agrees with the dist-info and Rust core (Finding F-1 from the v0.5.1 report).
- **`sinter_compat`: added `QectorDecoderWrapper` backward-compat alias** for
  `QectorSinterDecoder`. Older docs and examples import cleanly.
- **`stim_compat`: added `stim_circuit_to_check_matrix` public alias** for
  `from_stim_detector_error_model`. Both names are importable and documented.
- **`CUDABatchDecoder.__init__`: clean `RuntimeError` when no CUDA driver present**
  (Finding F-5 neighbour — opaque Rust-layer error replaced by a Python-side guard
  with actionable guidance). Always call `CUDABatchDecoder.is_available()` first.
- **`_bp_core.sum_product_bp`: silenced `RuntimeWarning: divide by zero in log`**
  (Finding F-5). The inner loop is now wrapped in
  `np.errstate(divide='ignore', invalid='ignore')`; numeric output is unchanged.

### Added — Validated benchmark data embedded
Independent validation (2026-06-24, PyPI venv, Windows 10, AMD Ryzen, GTX 1660 Ti,
Python 3.11, PyMatching 2.4.0, 100k shots/pt seed 777) is now embedded as:

- **`benchmark_results/validation_v051.json`** — machine-readable artifact with all
  LER tables, latency percentiles, throughput figures, GPU speedups, code structural
  validation, and findings summary for both the 20k primary and 100k re-test runs.
- **Class docstrings updated** (`CPUBatchDecoder`, `CUDABatchDecoder`,
  `LookupTableDecoder`, `UnionFindDecoder`, module-level docstring) to carry the
  independently measured numbers inline.
- **`PYPI_README.md` validation table** bumped to v0.5.2 and expanded with the
  Workbench latency section (277,778 decodes/s, p50 3.60 µs, p99 11.61 µs).

### Validation summary (86/87 checks PASS)

| Suite | Coverage | Result |
|---|---|---|
| Environment & metadata | OS, CPU, GPU, versions | PASS |
| Code structural (17 families) | n_qubits, n_checks, rank(H), k, d | PASS |
| Single-syndrome correctness (23 decoder/code combos) | 100% syndrome-valid | PASS |
| Repetition code LER (d=3–9, 100k shots) | Blossom == PyMatching (0.00% delta) | PASS |
| Surface code LER (d=3–7, 100k shots) | Blossom within 1.78% of PyMatching | PASS |
| CPU batch throughput (rep d=9) | 0.34M–2.70M shots/s by decoder | PASS |
| CUDA GPU correctness + speedup (100k shots) | 7.67× rep-d9, 6.93× surf-d5 | PASS |
| Workbench end-to-end | detect→benchmark→export JSON/CSV/PDF | PASS |
| Specialized decoders (Suite G) | AutoDecoder, LookupTable, Sliding, Streaming, BeliefMatching | PASS |
| Compatibility/robustness (Suite H) | pymatching_compat, stim_compat, edge cases | PASS |

One check failure: version metadata mismatch (F-1, cosmetic) — fixed this release.

---

# QECTOR Decoder v0.5.0 — Release Notes

**Version**: 0.5.0  
**Date**: 2026-06-23  
**Codename**: Lepton

---

## What's new in 0.5.0

This release promotes the validated work that had accumulated under *Unreleased*
into a tagged build, and refreshes the compiled wheel against current upstream
crates.

### Fixed
- **Blossom exactness at large distance (adaptive-k).** `BlossomDecoder` now uses
  an adaptive candidate cap `k = max(12, 4·√n_defects)` instead of a fixed `k=12`,
  restoring exact-MWPM logical-error-rate parity with PyMatching through **d=15**
  (`memory_x` / `memory_z`). Locked by `test_blossom_adaptive_k_regression.py`,
  `test_blossom_d15_no_gap.py`, `test_blossom_candidate_set_contains_optimal.py`,
  `test_weight_gap_histogram.py`, and `test_defect_count_vs_weight_gap.py`.

### Added
- **QECTOR Workbench** (`qector_decoder_v3.workbench.Workbench`): headless,
  fully-tested controller to load `.stim`/`.dem` files, run cancelable benchmark
  jobs through a FIFO queue, and export JSON/CSV/PDF reports built from real
  artifacts (no fabricated data), with backend detection + environment snapshot.
- **Evidence & reproduction scripts**: `run_due_diligence_bundle.py`,
  `belief_reference_compare.py`, `gpu_memory_profile.py`,
  `auto_backend_calibrate.py`, `leak_test.py`.
- **Provenance**: `benchmarking.capture_environment()` records `git_commit`, so
  every JSON artifact and report figure traces back to the exact build.
- **Expanded validation suite** (832 tests green): exact-MWPM parity, DEM-collapse
  equivalence with d=11/d=15 fixtures, logical-observable / stabilizer-coset
  correctness, belief-matching reference cross-check, BP-OSD on
  BB[[72,12]]/BB[[144,12,12]]/HGP/bicycle, GPU CPU-bit-identity + fallback +
  calibration, latency percentiles, and memory/leak profiling.
- **Full technical report** (`QECTOR_Decoder_v3_Full_Report.pdf`, 36 pages)
  regenerated for 0.5.0: accuracy parity through **d=15**, belief-matching
  **−34.8% at d=5 / −50.0% at d=7**, BP-OSD on `[[72,12]]`/`[[144,12,12]]`, GPU
  CPU-bit-identity, and a ~0.8% threshold — every figure traces to a SHA-256-hashed
  artifact in `benchmark_results/`.

### Build / dependencies
- Refreshed Rust dependencies (`rayon` 1.12, `fastrand` 2.4) and migrated the
  optional `grpc`/`full` stack to `tonic` 0.14 / `prost` 0.14 with a **vendored
  `protoc`** (`protoc-bin-vendored`) so gRPC builds need no system `protoc`.
- Default wheel features unchanged (`opencl`, `cuda` with graceful CPU fallback).

---

# QECTOR Decoder v0.4.0 — Release Notes

**Version**: 0.4.0  
**Date**: 2026-06-20  
**Codename**: Neutron

---

## Ecosystem & tooling update — 2026-06-22

A pure-Python ecosystem layer was added on top of the compiled `0.4.0` core (no
Rust rebuild required — the new modules ship in the same wheel via
`python-source = "python"`). This addresses the competitive gap analysis:
reproducibility, Stim compatibility, automatic backend selection, drop-in
PyMatching API, and benchmark transparency.

### New Python modules (`qector_decoder_v3.*`)

| Module | Purpose |
|--------|---------|
| `codes` | One-call code-family helpers: `repetition_code`, `ring_code`, `rotated_surface_code`, `unrotated_surface_code`, `toric_code`, `heavy_hex_code`, `from_parity_check_matrix` (dense/`scipy.sparse`), `hypergraph_product` (CSS). All surface families are validated matching graphs. |
| `dem` | **Correct** Stim Detector Error Model loader: `parse_dem`, `load_dem_file`, `from_stim`. Mechanisms = columns, detectors = rows; handles `repeat` / `shift_detectors` / `^` decomposition; emits check matrix, observables matrix, priors and matching weights. Works without Stim installed. |
| `result` | `DecodeResult` / `decode_with_diagnostics`: correction as uint8 / sparse / bit-packed, logical flips, matching weight, timing, backend metadata, `to_json()`, `explain()`. |
| `backend` | `AutoDecoder` routes CPU / Rayon / CUDA / OpenCL by batch size, with `calibrate()` crossover measurement, manual override, graceful GPU fallback and diagnostics. |
| `pymatching_compat` | `Matching` — a drop-in subset of `pymatching.Matching` (`from_check_matrix`, `from_detector_error_model`, `add_edge`, `add_boundary_edge`, `decode`, `decode_batch`). |
| `benchmarking` | Reproducible harness: environment capture, seeds, warmup, mean/median/std + p50/p90/p95/p99 + 95% CI, hot-vs-cold split, peak memory, JSON + CSV. |

### Corrected behaviour

- `stim_compat.from_stim_detector_error_model` now delegates to `dem`. The previous
  implementation conflated detector indices with qubit indices and produced an
  incorrect `H`; the detector graph is now built correctly (one column per fault
  mechanism, one row per detector).

### Verified invariants (from the test suite)

- `BlossomDecoder` is **exact MWPM** — brute-force optimal on every enumerated
  syndrome of the small codes (weight gap 0).
- `SparseBlossomDecoder` is a **region-growing** decoder — always syndrome-faithful
  and near-optimal (≥99% of small-code syndromes optimal, weight gap ≤1). It is
  not exact MWPM by design; use `BlossomDecoder` when exact minimum weight matters.
- QECTOR matching is **never heavier than PyMatching** across repetition d=11,
  rotated surface d=5/7, and toric L=4 (differences are equal-weight tie
  representatives).

### Tooling, packaging, docs

- New driver: `scripts/run_competitive_benchmark.py` → JSON + CSV + environment block.
- New examples: `examples/example_codes_and_diagnostics.py`, `example_stim_dem.py`,
  `example_pymatching_and_backend.py` (all exercised by `test_examples.py`).
- New CI: `.github/workflows/tests.yml` (Linux/Windows/macOS × Python 3.9–3.12,
  Rust `cargo test`, benchmark smoke job, coverage, ruff/mypy).
- New packaging extras: `[stim]`, `[bench]`, `[cuda]`, `[opencl]`, `[all]`.
- New docs: `docs/METHODOLOGY.md`, `docs/REPRODUCE.md`, `docs/SCALING.md`,
  `docs/CORRECTNESS_AUDIT.md`.
- Test suite: **387 passing** (2 skipped, 1 xfailed) including new code/DEM/result/
  backend/PyMatching/benchmark suites plus property-based and exhaustive
  brute-force correctness tests.

> Version stays **0.4.0**: the layer is additive and the compiled core is unchanged,
> so `qector_decoder_v3.__version__` continues to report `0.4.0`.

---

## Advanced decoders update — 2026-06-22

Three additions that move QECTOR from "matches PyMatching" to "beats PyMatching on
accuracy and covers the LDPC frontier", all pure-Python on the `0.4.0` core and
cross-validated against reference packages. See `docs/BEYOND_PYMATCHING.md`.

- **`belief_matching.BeliefMatching`** — sum-product BP on the hyperedge detector
  graph + QECTOR exact weighted MWPM on the edge graph (belief-matching). Achieves
  a **lower logical error rate than PyMatching** on Stim circuit-level shots
  (rotated surface, p=0.005): **25.5% LER reduction at d=5** (0.0062 vs 0.0083),
  parity at d=3. Verified directly and through Sinter; cross-checked against the
  reference `beliefmatching` package.
- **`bposd.BpOsdDecoder`** — self-contained sum-product BP + ordered-statistics
  (OSD-0 / OSD-w) for arbitrary GF(2) / LDPC check matrices, plus LDPC code
  families (`codes.bivariate_bicycle_code`, `codes.bicycle_code`). On the
  `[[72,12]]` BB code its logical error rate is within ~10% of the reference `ldpc`
  package (0.0370 vs 0.0340) and always syndrome-faithful.
- **`sinter_compat`** — `qector_blossom` / `qector_belief` / `qector_unionfind`
  exposed as `sinter.Decoder`s, so QECTOR drops into the community-standard
  Monte-Carlo harness used to benchmark PyMatching and fusion-blossom.
- **`predecoder.PredecodedDecoder`** — faithful local-matching predecoder (resolves
  adjacent defect pairs before the residual decoder) and `quantize_weights`.

Shared infrastructure: vectorised min-sum and sum-product BP (`_bp_core`),
GF(2) ordered-statistics solver, and `dem.DemModel.collapse_to_graph` (parallel-edge
merge). Benchmark drivers: `scripts/competitive_belief_matching.py`,
`scripts/competitive_stim_ler.py`. Test suite now **414 tests** (adds
belief-matching, BP-OSD/LDPC, Sinter, predecoder suites, all cross-validated);
the fast core is verified stable over a 20× repeated-run stability sweep.

> Requires the optional packages for the advanced paths: `stim`, `pymatching`
> (matching/belief), `ldpc` (BP-OSD cross-checks), `sinter` (harness). Install via
> the `[stim]` / `[all]` extras.

---

## Summary

This release delivers the complete QECTOR v3 decoder suite with 4 algorithmic backends, GPU acceleration via OpenCL, precision decoders (BP-OSD, Neural), Sparse Blossom with blossom contraction/shattering, and production infrastructure (gRPC, Prometheus, MCP). All 72 Rust tests and 260+ Python tests pass.

---

## What's New

### Algorithmic Decoders

| Decoder | Status | Key Feature |
|---------|--------|-------------|
| `UnionFindDecoder` | Stable | Hot-path 1.6 µs, SIMD + pooled allocators |
| `BlossomDecoder` | Stable | Edmonds MWPM, exact for d≤7 |
| `SparseBlossomDecoder` | Stable | Region-growing BFS, blossom contraction + shattering, exact DP n≤20 |
| `BPOSDDecoder` | Stable | Belief propagation + ordered statistics, LER 0.086 @ d=5, p=0.05 |
| `NeuralPredecoder` | Stable | MLP Xavier/ReLU, hybrid fallback 35-93% |
| `GNNPredecoder` | Experimental | Message-passing + edge readout, forward pass OK |
| `LookupTableDecoder` | Stable | Exact d=3,5,7 precomputed, SIMD fallback |
| `HybridDecoder` | Stable | Auto-selection per syndrome difficulty |
| `StreamingDecoder` | Stable | Sliding window, 1.6 µs real-time |
| `BatchDecoder` / `CPUBatchDecoder` | Stable | SIMD, parallel, pooled — 4.1M dec/s d=5 |
| `OpenCLBatchDecoder` | Stable | GPU dual-kernel, transparent fallback, resilience |

### GPU Acceleration (OpenCL)

- Dual-kernel: global memory (batch≥1024) + local memory (batch<1024)
- Transparent CPU fallback on GPU failure
- Auto-recovery: exits degraded mode after 10 successful calls
- Observability: `consecutive_failures`, `total_failures`, `gpu_recoveries`, `degraded_calls`
- Performance: 14.6M dec/s @ d=5, batch=10000

### Production Infrastructure

- **Feature flags**: `opencl`, `grpc`, `cuda`, `full` (Cargo.toml)
- **gRPC server**: Decode + batch decode endpoints (commented, `grpc` feature)
- **Prometheus metrics**: `metrics` feature, `start_metrics_server()`
- **MCP server**: JSON-RPC 2.0 for Claude Code integration
- **Examples**: `examples/example_basic.py`, `example_batch.py`, `example_streaming.py`, `example_blossom.py`

### Documentation

- `README.md` — Quick start, decision matrix, validated scope & known limitations
- `CHANGELOG.md` — Release history (incl. the adaptive-k fix)
- `INSTALL.md` — Installation instructions (this release)
- `docs/QECTOR_Decoder_v3_Full_Report.pdf` — Full technical report (26 sections)
- `docs/BENCHMARK_COMPETITIVE.md` — Competitive methodology vs PyMatching
- `docs/BEYOND_PYMATCHING.md` — Belief-matching, BP-OSD, GPU
- `docs/CORRECTNESS_AUDIT.md` — Correctness audit
- `docs/METHODOLOGY.md`, `docs/SCALING.md`, `docs/REPRODUCE.md`
- `docs/reports/` — Historical reports (GNN training, decoder correctness v3.6)
- `docs/internal/` — Competitive analysis, roadmap (internal)

---

## Performance Highlights

| Metric | Value | Conditions |
|--------|-------|------------|
| Single-shot latency | 1.6 µs | CPU `decode()`, d=5 |
| GPU batch throughput | 14.6M dec/s | d=5, batch=10000, OpenCL |
| CPU batch throughput | 4.1M dec/s | d=5, batch=10000, SIMD |
| BP-OSD LER | 0.086 | d=5, p=0.05, 10k shots |
| UnionFind LER | 0.321 | d=5, p=0.05, 10k shots |
| Blossom LER | 0.198 | d=5, p=0.05, 10k shots |
| Sparse vs Blossom bit-perfect | 100% | d=5, ring code, 100k trials |
| All tests pass | 260+ (Python) + 72 (Rust) | Default + `full` features |

---

## Breaking Changes

None — this is a forward-compatible release from v0.3.0. Feature flags are additive.

---

## Known Issues

1. **Dead code warnings**: 11 warnings in `blossom.rs` (unused fields/methods) and `sparse_blossom.rs` (unused structs/methods). These are intentional保留 fields for future Edmonds blossom algorithm activation and Radix heap optimization. They do not affect functionality.
2. **Neural predecoder**: 35-93% fallback rate. Recommendations: switch to GNN, train on Blossom teacher, or predict edge weights for SparseBlossom.
3. **Sparse Blossom toric boundaries**: Surface code with periodic boundaries (toric) has incomplete boundary handling. Use planar boundaries or ring code for now.
4. **CUDA Backend**: Fully implemented using NVIDIA CUDA Driver API, dynamic NVRTC kernel compilation, and a reusable workspace (no longer a stub).

---

## Contributors

- Guillaume Lessard / iD01t Productions — Author, core algorithm design & Rust implementation

---

## License

**QECTOR Decoder Source-Available License v1.0** — see `LICENSE`.
Copyright © 2026 Guillaume Lessard / iD01t Productions. All rights reserved.
Free for non-commercial use; commercial use requires a paid license
(guiliguili2705@gmail.com · https://www.qector.store).

---

## Upgrade Notes

From v0.3.0:
- No API changes — all Python classes remain compatible
- New feature flags available: `grpc`, `cuda` (no-op), `full`
- New `OpenCLBatchDecoder.reset()` method for clearing resilience counters
- New Python property: `OpenCLBatchDecoder.gpu_recoveries`

---

## Links

- Repository: https://github.com/qector/qector-decoder
- Issues: https://github.com/qector/qector-decoder/issues
- Documentation: See `README.md` and `DECODER_COMPARISON.md`
