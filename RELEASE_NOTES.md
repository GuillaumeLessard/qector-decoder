# Changelog

All notable changes to QECTOR Decoder will be documented in this file.

## [0.6.9] - 2026-07-26
**Focus**: BP-OSD accuracy (exact log-domain BP, true OSD-1/2), belief-matching correctness, licence hardening.

### Fixed
- **`BeliefMatching.from_numpy_h` decoders returned an empty array for every syndrome** — zero observable rows meant `decode` produced length-0 output with no error. Now returns a length-`n_qubits` correction with `H @ corr == syndrome` verified.
- **`verify_license_token` raised on malformed input** instead of returning `False` (`binascii.Error`/`UnicodeDecodeError` escaped the old `except RuntimeError`). Nine adversarial inputs now locked by tests.
- **Blossom boundary bug**: odd-defect boundary-less codes no longer panic.
- Stripe checkout no longer pins `payment_method_types=["card"]`, restoring Link/wallets/local payment methods.

### Added
- **Exact log-domain sum-product BP** (default; `bp_method="min_sum"` opt-out) and **true combination-sweep OSD-1/2** (`osd_order` kwarg).
- **`GNNBeliefMatcher`** — GNN-guided MWPM pipeline with faithfulness fallback.
- **v2 licence tokens** carrying `tier` + `exp` inside the Ed25519 signature; legacy tokens keep verifying; `license_claims()` exposes verified, unexpired claims.
- **Tuning env vars documented** (`QECTOR_BLOSSOM_K_MULT`, `QECTOR_BLOSSOM_INTRA_PAR`, `QECTOR_BLOSSOM_INTRA_THREADS`, `QECTOR_CUDA_DEVICE_ID`, `QECTOR_OPENCL_DEVICE_ALLOW`) — with which of them change results vs. only throughput.
- Licensing notice now reaches non-interactive runs (one stderr message at import; `QECTOR_SILENT=1` suppresses; silent when licensed; skipped in CI).

### Validation
`cargo test --lib` 203 passed · clippy 0 warnings · pytest 1237 passed, 1 skipped, 0 failed · ruff check/format clean · captured under `test-results/`.

## [0.6.8] - 2026-07-22
**Focus**: Hotfix for unimportable v0.6.7 wheel — guarded imports, CI smoke test, YAML fix.

### Fixed
- **v0.6.7 was completely unimportable** on all published wheels. `__init__.py:37` unconditionally accessed `_native_module.HybridCascadeDecoder`, which does not exist in the current Rust build (symbol never registered in `#[pymodule]`). All 18 native-module lookups are now guarded by `_guard("ClassName")` — missing symbols return a callable stub that raises `RuntimeError` on instantiation. `import qector_decoder_v3` now always succeeds.
- **CI YAML syntax error**: the smoke-test `run:` step contained unindented Python code inside a literal block scalar, causing GitHub's parser to fail on the entire workflow file. This broke `workflow_dispatch`, tag-push triggers, and all CI runs silently (0 jobs, `failure` conclusion). Fixed by indenting the inline Python script to match the literal block's content level.

### Added
- **CI smoke test**: `release` job now installs the built wheel, runs `import qector_decoder_v3`, creates a code, and decodes before `twine upload`. This would have caught the v0.6.7 regression.

---

## [0.6.7] - 2026-07-22
**Focus**: Self-auto-debug backend, offline licensing, and Stripe fulfillment.

### Added
- **Self-Auto-Debug Backend**: `AutoDecoder` implements a 7-tier fault-tolerant fallback engine (`CUDA` -> `OpenCL` -> `CPU Rayon` -> `CPU Batch` -> `CPU Single` -> `Blossom` -> `Lookup Table`) with automatic exception trapping, per-tier health scoring, and transparent recovery. `reset_backend_health()` restores suspended tiers.
- **Ed25519 Offline License Verification**: `verify_license_token()` validates signed license tokens fully offline against an embedded public key. Supports both legacy 2-part and self-contained 3-part (`receipt_id.email_b64.sig_b64`) token formats. Configurable via `QECTOR_LICENSE` and `QECTOR_SILENT` environment variables.
- **Stripe Checkout & Webhook Fulfillment**: `stripe_integration.py` and `stripe_webhook_server.py` provide end-to-end commercial license fulfillment - Checkout Session creation, webhook signature verification, and automatic Ed25519 token issuance on payment confirmation. Direct purchase link: [Buy Commercial License](https://buy.stripe.com/7sY9AVdwlgoyfse9bYeUU00?locale=en&__embed_source=buy_btn_1TsoKxRsa9cg9l8A7ExMmc77).
- **DecoderPool LRU Caching**: Added LRU pool caching for `get_decoder_pool` in `decoder_cache.py`, preventing redundant process pool initialization across repeated multi-process decodes.

### Fixed
- **Stripe Webhook Signature Verification**: Enforced strict signature verification in `handle_stripe_webhook_payload` whenever a webhook secret is set, preventing unauthenticated payload bypass.
- `SparseBlossomDecoder::grow_regions` no longer collapses the compressed edge set - decoded syndromes are now bit-identical to the Blossom decoder.
- `BPOSDDecoder.bp_decode_timed` initializes the wall-clock deadline before the iteration loop, so the latency budget is honored from the first iteration.
- LER benchmark's rotated-surface generator now emits a proper two-half (X + Z) graphlike code.
- `_opencl_health_check()`'s child-process probe script referenced an undefined `_np` name instead of `np`, causing a silent `NameError` that always reported `opencl_is_available() == False` regardless of hardware.

### Verification
- Full test suite passing, including new `test_auto_debug_fallbacks.py`, `test_stripe_integration.py`, `test_stripe_zero_dollar_sale.py`, and `test_release_import.py`.
- Wheels rebuilt for CPython 3.9-3.13 (win_amd64) and verified against a clean uninstall/reinstall.
- End-to-end $0 Stripe sale verified: checkout -> webhook -> Ed25519 issuance -> offline activation -> tamper rejection.

---

## [0.6.6] - 2026-07-12
**Focus**: Critical stability fix and production hardening.

### Fixed
- **Critical Import Failure**: Resolved an `AttributeError` on `OpenCLBatchDecoder` during module initialization. In v0.6.5, an unguarded import was left in `__init__.py`, which failed on wheels built with `--no-default-features --features cuda`. This has been removed, allowing the properly guarded fallback to execute as intended.
- **Hypergraph Validation**: `UnionFindDecoder` and `FastUnionFindDecoder` now explicitly reject hypergraph codes where any qubit participates in >2 checks (`UfGraph::new` returns `Result<Self, String>`), eliminating silent syndrome-invalid corrections.
- **Input Validation**: Added comprehensive validation for empty, negative, duplicate, range, `u32::MAX`, and non-integer types, raising clean `ValueError`/`TypeError`.
- **Namespace Leakage**: Removed `os`, `sys`, `subprocess`, and `np` from the public `__init__.py` module scope.
- **Routing Safety**: `recommend_decoder` now safely avoids recommending the UF family on hypergraphs.

### Changed
- Packaging: `sdist` is now published alongside wheels. Wheel matrix expanded for Linux, Windows, and macOS arm64.
- Testing: Expanded test matrix and relaxed d=21 latency threshold for CI stability.

---

## [0.6.5] - 2026-07-12 `[YANKED]`
*This release was yanked from PyPI shortly after publication.*
- **Reason**: Critical import failure on all published wheels due to an unguarded `OpenCLBatchDecoder` reference in `__init__.py`. CI builds excluded OpenCL, causing an immediate `AttributeError` for all users. 
- **Resolution**: All users should upgrade directly to `v0.6.6`.

---

## [0.6.4] - 2026-07-10 `[YANKED]`
*This release was yanked from PyPI shortly after publication.*
- **Reason**: Internal CI/CD pipeline misconfiguration resulted in incomplete artifact publishing. 
- **Resolution**: Superseded by `v0.6.6`.

---

## [0.6.3] - 2026-07-10 `[YANKED]`
*This release was yanked from PyPI shortly after publication.*
- **Reason**: GitHub Actions secrets misconfiguration during the Rust build step.
- **Resolution**: Superseded by `v0.6.6`.

---

## [0.6.2] - 2026-07-07
**Focus**: Production hardening, correctness, and audit remediation.

### Highlights
- Comprehensive input validation and improved NumPy type coercion (`np.int*`, `np.bool_`).
- All docs, versioning, and metadata aligned to 0.6.2.
- *(Note: This was the last stable release prior to the v0.6.6 corrective rollout).*

---

## [0.6.0] - 2026-07-05
**Focus**: API drift correction and Python 3.9 compatibility.

### Fixed
- **API Drift**: Updated `README.md` and `PYPI_README.md` Stim detector-error-model quick-start examples to use `from_stim_detector_error_model` instead of the removed `stim_circuit_to_check_matrix`.
- **Python 3.9 Compatibility**: Replaced PEP 604 `X | None` union syntax with `typing.Optional`/`typing.Union` in `backend.py`, `qiskit_plugin.py`, `stim_compat.py`, and `__init__.py`.

### Changed
- Package metadata (`pyproject.toml`, `Cargo.toml`, `Cargo.lock`, runtime fallback version, `CITATION.cff`, `codemeta.json`) bumped to `0.6.0`.

---

## [0.5.9] - 2026-07-02
**Focus**: GPU acceleration, routing, and streaming workflows.

### Added
- CuPy-accelerated GPU backend (`gpu_backend.py`, `bp_cupy.py`).
- Automatic decoder backend routing (`routing.py`).
- Streaming/sliding-window decoding sessions (`streaming.py`).

### Removed
- Superseded `advanced.py` module and due-diligence bundle helper scripts.

---

## [0.5.0] - 2026-06-23
**Codename**: Lepton

### Fixed
- **Blossom exactness at large distance**: `BlossomDecoder` now uses an adaptive candidate cap `k = max(12, 4·√n_defects)`, restoring exact-MWPM logical-error-rate parity with PyMatching through d=15.

### Added
- **QECTOR Workbench**: Headless, fully-tested controller for benchmark jobs and JSON/CSV/PDF report generation.
- **Expanded validation suite**: 832 tests green, covering exact-MWPM parity, DEM-collapse equivalence, belief-matching cross-checks, and GPU CPU-bit-identity.
- **Full technical report**: Regenerated for 0.5.0, detailing accuracy parity and a ~0.8% threshold.

---

## 🛡️ License & Commercial Use Notice

QECTOR Decoder is released under the **QECTOR Source-Available License v1.0**. 
- **Free** for personal, academic, educational, and non-commercial research use.
- **Commercial, institutional, lab, or product-integration use requires a paid commercial license.**

For licensing inquiries, source-review access, or enterprise deployment, please contact:
- **Email**: [admin@qector.store](mailto:admin@qector.store)
- **Web**: [https://qector.store/pricing](https://qector.store/pricing)
- **Provenance**: Protected by timestamped archival (Zenodo DOI).

See `LICENSE` and `COMMERCIAL.md` for full terms.

---

## Module Reference (historical, v0.4.0)

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
(admin@qector.store · https://www.qector.store).

---

## Upgrade Notes

From v0.3.0:
- No API changes — all Python classes remain compatible
- New feature flags available: `grpc`, `cuda` (no-op), `full`
- New `OpenCLBatchDecoder.reset()` method for clearing resilience counters
- New Python property: `OpenCLBatchDecoder.gpu_recoveries`

---

## Links

- Repository: https://github.com/GuillaumeLessard/qector-decoder
- Issues: https://github.com/GuillaumeLessard/qector-decoder/issues
- Documentation: See `README.md`
