# Changelog

All notable changes to QECTOR Decoder v3 are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning. Every benchmark artifact is stamped with the git commit and
environment so report figures trace back to a specific build.

## [Unreleased]

_Nothing yet._

## [0.6.9] - 2026-07-26

### Added
- **v2 licence tokens carrying `tier` and `exp`.** Legacy tokens sign only
  `receipt_id:email`, so a 60-day evaluation was cryptographically identical to
  a perpetual licence. Format `v2.{claims_b64}.{sig}` over
  `{rid, email, tier, exp}`; the signature covers the encoded claims segment
  verbatim and is verified *before* the claims are parsed. `license_claims()`
  returns verified, unexpired claims so callers can gate on tier.
  Legacy 2- and 3-part tokens continue to verify unchanged, and a v2 token fails
  **closed** on installs predating v2 support.
- **Tuning environment variables documented** in the README: `QECTOR_BLOSSOM_K_MULT`,
  `QECTOR_BLOSSOM_INTRA_PAR`, `QECTOR_BLOSSOM_INTRA_THREADS`,
  `QECTOR_CUDA_DEVICE_ID`, `QECTOR_OPENCL_DEVICE_ALLOW` — with which of them can
  change results rather than only throughput.
- **Task A — Exact log-domain BP**: `BpMethod{MinSum,Exact}` enum in `bp_osd.rs`, default `Exact`. `phi(x) = -ln(tanh(x/2))` via hybrid exact (<0.25) + 65536-entry interpolated LUT ([0.25, 20]) + 0.0 (≥20). Deterministic reliability ranking (`rel_key` 1e-6 quantization + index tie-break) fixes float-noise OSD-0 basis flipping. PyO3 `bp_method` kwarg (`"exact"`/`"min_sum"`).
- **Task B — Higher-order OSD**: true combination-sweep OSD-1/2 in `bp_osd.rs` — flip subsets of W≤12 least-reliable selected columns, residual GF(2) re-solve with syndrome pre-subtraction, min-weight faithful candidate wins. PyO3 `osd_order` kwarg (0 default preserved, non-breaking).
- **Task C — GNN-enhanced belief matching**: `GNNBeliefMatcher` class in `belief_matching.py` — end-to-end GNN-guided MWPM pipeline (`DetectorGraph` → `GNNPredecoder.predict_with_node_probs` → per-edge weights → max-per-qubit fan-in → `SparseBlossomDecoder.decode_with_weights` → faithfulness fallback). Optional synthetic training (`train_samples`/`error_rate`/`train_epochs`/`seed`). `decode_with_gnn` one-shot helper. Both re-exported at package top level.
- **`BPOSDDecoder` Python wrapper** now forwards `bp_method`/`osd_order` kwargs (additive, default-preserving).
- **`backend.py`**: batch-decode 1D-output reshape fix (handles decoders returning flat arrays for 2D input).
- **`stripe_integration.py`**: `create_license_token` import fallback when `generate_license_keys` is absent.

### Fixed
- **`BeliefMatching.from_numpy_h` produced a decoder that returned an empty
  array for every syndrome.** The constructor built `hyper_obs` with zero rows,
  and `decode` returns `hyper_obs @ hard` — so any `BeliefMatching` built from a
  raw parity-check matrix silently decoded nothing and reported no error.
  `edge_obs` carried the identical defect, breaking the matching fallback for
  exactly the syndromes BP found hardest. Both matrices now carry qubit
  incidence, so `decode` returns a length-`n_qubits` correction; verified
  faithful (`H @ corr == syndrome`) on all 16 repetition-code d=5 syndromes.
- **`verify_license_token` raised instead of rejecting malformed input.** The
  `except RuntimeError` clauses caught neither `binascii.Error` nor
  `UnicodeDecodeError` (both `ValueError` subclasses), so a garbage token
  propagated an exception to the caller rather than returning `False`. Any
  caller passing attacker-influenced input crashed. Now catches
  `(InvalidSignature, ValueError, TypeError)`; nine adversarial inputs are
  locked by tests.
- **Checkout no longer pins `payment_method_types=["card"]`.** Pinning it
  disables Stripe dynamic payment methods, so Link, Apple/Google Pay and local
  methods never render and every buyer must type a card number. Omitting the
  parameter lets Stripe show the highest-converting eligible methods per
  customer, configured from the Dashboard.
- **blossom.rs boundary bug**: boundary node matched without `boundary_spt`, panicking on odd-defect boundary-less codes.

### Performance / internals (dev.md items, this cycle)
- f32 GNN stack (Task 2.1); seeded GNN init (6.5); hybrid hot-path allocations (6.7); word-packed GF(2) solver `src/gf2.rs` shared by BP-OSD and Blossom (1.1); hyperedge `BestEffortVerified` policy (7.2); mwpm dense fallback threshold (4.1); RadixHeap SmallVec buckets (4.4); sliding-window bit-packed history (5.2); latency quantiles (5.4); NoHashHasher lookup table (2.3); safetensors strict shape guard (2.4).

## [0.6.8] - 2026-07-22

### Fixed
- **v0.6.7 was completely unimportable**: `__init__.py` unconditionally accessed `_native_module.HybridCascadeDecoder`, which is absent from the compiled module. All 18 native-module lookups now use `_guard("ClassName")` — missing symbols return a callable stub that raises `RuntimeError` at decode-time. Import always succeeds.
- **CI YAML broken**: smoke-test `run:` step had unindented Python code inside a literal block scalar, breaking GitHub's parser. No workflow triggers worked (`workflow_dispatch`, tag push, pull_request). Fixed by indenting the inline Python to match the content level.

### Added
- **CI smoke test before publish**: installs wheel, imports `qector_decoder_v3`, instantiates `sparse_blossom` decoder. Catches unimportable wheels before they reach PyPI.

## [0.6.7] - 2026-07-20

### Fixed
- **`indices[self.n_checks..]` panic in `bp_osd.rs` (`BPOSDDecoder.decode`)**: crashed the entire Python process (unrecoverable Rust panic across the PyO3 boundary, not a catchable exception) on hyperedge check structures, e.g. the 18-check/9-qubit rotated surface code. Reproduced deterministically in an isolated process, patched, rebuilt, reinstalled, and re-verified via the exact original repro (exit 0, correct output). Confirmed specific to hyperedge structures — the same call path runs cleanly on a graphlike repetition code both before and after the patch.
- **NaN `error_rate` panic in `bp_osd.rs`**: an unclamped NaN error rate silently poisoned belief propagation, burning through all 50 iterations before crashing (~133s wall time). Fixed by clamping `error_rate` at the constructor. Re-verified via the exact original repro: now returns in ~17s — matching the baseline runtime of every other clean script this session — instead of crashing at 133s.
- **`_opencl_health_check()`'s child-process probe script referenced an undefined `_np` name.** The probe script imports `numpy` as `np` in the child process, but the probe line referenced `_np` (the parent module's private alias, not defined in that child scope), raising a silent `NameError` on every invocation. This unconditionally set `_OPENCL_HEALTH_CACHE = False`, meaning `opencl_is_available()` always reported `False` regardless of real hardware or driver support. Fixed by using `np` consistently in the probe script. Verified directly against the shipped `__init__.py`: the embedded child-process script now reads `import numpy as np` and uses `np.asarray(...)` / `np.array(...)` throughout the probe, with no `_np` reference anywhere in that scope. (A same-machine black-box call to `opencl_is_available()` alone can't confirm this fix, since this machine has no OpenCL hardware and would correctly return `False` either way — the source-level check above is what closes the question.)
- **`SparseBlossomDecoder::grow_regions` / `RadixHeap` — bit-identical to `BlossomDecoder`.** Re-confirmed empirically: 10/10 trials with genuinely syndrome-reachable errors (constructed via `H @ error`, not hand-picked) on the 18-check hyperedge code produced bit-for-bit identical corrections between the two decoders. Note: this repo's own tests named "bitperfect" (`test_sparse_vs_blossom_bitperfect_ring`, `..._ring_20`) do not actually assert cross-decoder equality — they only check each decoder's own output independently satisfies the syndrome. The bit-identical claim rests on this session's empirical test, not on those two Rust tests.
- **`BPOSDDecoder`'s `bp_decode_timed` (Rust) / `decode_timed` (Python) — deadline honored before the first iteration.** Confirmed at three independent levels: source shows `Instant::now()` initialized before the loop with the deadline check as the literal first statement of each iteration; both relevant Rust unit tests pass (`test_bp_decode_timed_converges_and_matches_untimed`, `test_bp_decode_timed_respects_zero_latency_deadline`); and live timing on the 18-check hyperedge code shows `max_latency_ms=0.0` stabilizing at ~0.02ms vs. a generous budget's consistent ~0.68ms — a clean ~30x gap.

### Retracted
- **LER benchmark's rotated-surface generator does *not* emit a graphlike code.** A prior draft of this entry claimed `generate_surface_code_checks` was fixed to emit "a proper two-half (X + Z) graphlike code." Re-tested directly against the live 0.6.7 install: `generate_surface_code_checks(3)` still returns an 18-check, 9-qubit matrix where every qubit participates in 8 checks (graphlike requires ≤2). Unchanged from prior versions — still a rank-4-of-18 hyperedge code, exactly as its own docstring describes ("periodic/toric surface code," "hyperedges," "rank-deficient"). No code change was made here this cycle; the claim was inaccurate and is retracted, not fixed.

### Added
*(Not independently re-tested this session — carried over as-is from a prior draft of this entry.)*
- `SparseBlossomDecoder.k_nearest_via_radix` — public event-driven candidate-edge discovery backed by a new `RadixHeap<u32, HeapEvent>` structure exposed to downstream callers that need fine control over the candidate set.
- MCP server (`mcp_server`) now exposes 5 new tools: `decode_syndrome_blossom`, `batch_decode_blossom`, `run_ler_benchmark`, plus expanded `get_decoder_info` listing all 11 decoder families.

### Quality
- Cross-decoder syndrome-validity test suite (`src/cross_decoder_tests.rs`) covers UF / FastUF / LookupTable / SparseBlossom / BP-OSD / SlidingWindow / Streaming / Hybrid.
- SafeTensors loader now has a full round-trip test suite covering generic + runtime dispatch, dtype mismatch, missing tensors, and shape round-trip.
- Dead-code warnings eliminated across the crate (8 → 0).
- `cargo test --lib`: 142 passed, 0 failed, 0 ignored — re-run directly this session (finished in 14.91s), unchanged after this session's two `bp_osd.rs` edits. `cargo check --all-targets` not re-run this session.

## [0.6.6] - 2026-07-12

### Fixed
- **Critical: package import broken on every published v0.6.5 wheel.** `python/qector_decoder_v3/__init__.py` had a leftover, unguarded `_RustOpenCLBatchDecoder = _native_module.OpenCLBatchDecoder` line (a duplicate of the properly try/except-guarded assignment later in the file). Since the public CI release wheels are built with `--no-default-features --features cuda` (no `opencl` feature), the compiled module never has this attribute, so `import qector_decoder_v3` raised `AttributeError` immediately on a completely clean install. This was masked in local development because default-feature builds include `opencl`. Root cause found and reproduced by testing a fresh `pip install qector-decoder-v3==0.6.5` in an isolated venv and by rebuilding locally with the exact CI feature flags. Fixed by removing the dead duplicate line; verified the corrected package imports cleanly under the exact CI build configuration in a clean venv, with `cuda_is_available()` / `opencl_is_available()` correctly returning `False` rather than crashing.
- **v0.6.5 is not usable and should not be installed** — use v0.6.6 or later.

## [0.6.5] - 2026-07-10

### Fixed
- **mypy clean**: Resolved all 8 type errors across `decode_mmap.py`, `decoder_pool.py`, and `belief_matching.py` — strict type checking now passes on the full Python layer.
- **Test imports**: `test_comprehensive_suite.py` now correctly imports `DecoderPool`, `get_decoder`, `clear_decoder_cache`, `get_decoder_pool` from the local source.
- **CI resilience**: Ensured v0.6.5 Python layer matches the Rust source — no more version skew between wheel metadata and runtime API.
- **API consistency**: Fixed `PredecodedDecoder` backend validation to accept `"union_find"` (with underscore) matching the canonical decoder names.
- **Test suite NameError**: `test_comprehensive_suite.py::_run_pool_test` had a genuine bug (`syndrome` referenced instead of `syndromes`) — a live crash risk on any machine where Windows spawn multiprocessing succeeds. Fixed and verified: full suite run 1005 passed, 83 skipped, 0 failed (excluding one unrelated example-script issue, also fixed below).
- **ruff clean**: Full repo now passes `ruff format --check` and `ruff check` with zero errors; `.venv`, `.venv_clean_test`, `target`, `dist`, `lib`, `proto` excluded from lint scope; per-file ignores added for `cpu_benchmark_report.py` and `test_exports.py`.
- **`examples/example_batch.py`**: was constructing `CPUBatchDecoder`/`OpenCLBatchDecoder`/`CUDABatchDecoder` (Union-Find-based, weight ≤2 checks only) against a weight-4 surface code, which the decoders correctly reject at construction. Switched to `generate_ring_code_checks()`, the correct weight-2 graph-like code family for this decoder class. Verified: `python/tests/test_examples.py` passes (1 passed in 154.39s), and the script runs end-to-end.
- **CI secret injection**: Regenerated and verified the `RUST_SRC_B64_1/2/3` GitHub Actions secrets (byte-identical round-trip checked before upload). Confirmed full 15-platform wheel build (Linux/Windows/macOS x86_64/aarch64 x Python 3.9-3.13) succeeds end-to-end.

### Changed
- Bumped package, crate, runtime fallback, citation, and metadata versions to `0.6.5` across `pyproject.toml`, `Cargo.toml`, `python/qector_decoder_v3/__init__.py`, `CITATION.cff`, `codemeta.json`, `README.md`, `PYPI_README.md`, docs, and examples.

## [0.6.4] - 2026-07-10

### Fixed
- **CI secrets updated**: Rust source injected at build time now matches the v0.6.4 Python layer. The v0.6.3 wheel was built with stale Rust source (missing `LERBenchmark` and other v0.6.3 Rust changes) — it has been superseded by v0.6.4.

## [0.6.3] - 2026-07-10

### Added
- **BP-OSD convergence cap**: 50-iteration max, early-exit on belief convergence (max |Δ| < 1e-6), `decode_timed(max_latency_ms)` for tail-latency control.
- **AVX2 SIMD transpose + gather**: CPU batch decoder auto-detects AVX2 via `is_x86_feature_detected!` — 1.1M shots/s on surface d=3, batch=32768.
- **Blossom intra-decode Rayon parallelism**: k-NN search parallelized via `into_par_iter()` when n_defects > 40.
- **DecoderPool**: Multi-process batch decoding with auto-Rayon fallback on Windows (50–500× faster than multi-process IPC).
- **Cached decoder factory**: `get_decoder()` / `clear_decoder_cache()` / `get_decoder_pool()` — zero construction cost after first call.
- **`decode_mmap`**: Out-of-core decoding via memory-mapped NumPy arrays.
- **`DecodeResult` / `decode_with_diagnostics`**: Structured decode results with per-shot diagnostic metadata.
- **`Workbench`**: High-level orchestration for multi-decoder comparison and benchmarking.
- **Comprehensive test suite**: `test_comprehensive_suite.py` — 200+ scenario tests across all decoder families.

### Changed
- `FastUnionFindDecoder` docstring updated: "Consistently faster than UnionFindDecoder on surface and repetition codes (1.1M shots/s)".
- `run_mcp_server` gated behind `grpc` feature; `OpenCLBatchDecoder`/`opencl_is_available` gated behind `opencl`.
- CPUBatch `batch_decode()` now calls SIMD path by default; `batch_decode_par()` for explicit Rayon variant.
- Bumped package to 0.6.3 across all metadata files.

### Fixed
- `bposd.py` line 118: CRW consistency bug in belief tracking.
- DecoderPool on Windows: auto-selects single-process Rayon path instead of broken multi-process IPC.
- Memory layout optimizations: aligned Vecs, pre-reserved capacity in Blossom construction.

## [0.6.2] - 2026-07-06

### Added
- v0.6.2 release notes: `CHANGELOG_v0.6.2.md`.

### Changed
- Bumped package, crate, runtime fallback, citation, and metadata versions to `0.6.2` across `pyproject.toml`, `Cargo.toml`, `python/qector_decoder_v3/__init__.py`, `CITATION.cff`, `codemeta.json`, `README.md`, `PYPI_README.md`, docs, and examples.

### Fixed
- Hardened Union-Find decoder input validation and error handling in `python/qector_decoder_v3/__init__.py`.
- Expanded regression coverage for hypergraph rejection and relaxed latency validation.

## [0.6.1] - 2026-07-05

### Fixed
- **README.md**: the "Belief-matching accuracy mode" example called
  `BeliefMatching(check_to_qubits, n_qubits, error_rate=0.005)`, which does
  not match the real constructor (`BeliefMatching(matrices, max_iter=30,
  bp_shortcut=False)`) and raises `TypeError: unexpected keyword argument
  'error_rate'` if run verbatim. Replaced with a self-contained example using
  `BeliefMatching.from_stim_circuit(circuit)`, verified by executing it
  end-to-end against the published `0.6.0` wheel.
- Audited every class instantiation in every `*.md` file in the repo against
  the real `__init__` signatures (not just import-name existence, which the
  `0.6.0` audit covered) — this was the only mismatch found. `BpOsdDecoder`
  and the Sinter integration example were checked and confirmed correct.

## [0.6.0] - 2026-07-05

### Fixed
- **README.md / PYPI_README.md**: the Stim detector-error-model workflow
  example referenced `qector_decoder_v3.stim_compat.stim_circuit_to_check_matrix`,
  a function that does not exist (it was superseded by
  `from_stim_detector_error_model` during the 0.5.9 cleanup, without the
  docs being updated). Both quick-start examples now import
  `from_stim_detector_error_model` and build the `check_to_qubits` mapping
  from a real `stim.DetectorErrorModel` (`circuit.detector_error_model(...)`),
  matching the documented function's actual signature.
- **Python 3.9 compatibility**: replaced PEP 604 `X | None` union syntax with
  `typing.Optional`/`typing.Union` in `backend.py`, `qiskit_plugin.py`,
  `stim_compat.py`, and `__init__.py`. This syntax requires Python 3.10+ and
  would raise `TypeError` at import time on 3.9, contradicting the package's
  own `requires-python = ">=3.9"` and the `smoke-import-py3.9` CI job.
- Hardened `test_clean_venv_install.py`'s qiskit-absent smoke test to also
  stub out `qiskit`, not just `stim`/`pymatching`.
- Version-string consistency: bumped `pyproject.toml`, `Cargo.toml`,
  `Cargo.lock`, `python/qector_decoder_v3/__init__.py`, `CITATION.cff`, and
  `codemeta.json` to `0.6.0`, and updated all plain-text version labels in
  `INSTALL.md`, `README.md`, `PYPI_README.md`, `docs/GPU_AND_CUPY.md`,
  `docs/SERVICE_API_SCHEMA.md`, and the `examples/` scripts.

## [0.5.9] - 2026-07-02

### Added
- **CuPy-accelerated GPU backend** (`gpu_backend.py`, `bp_cupy.py`): batched
  belief-propagation / BP-OSD decoding on NVIDIA GPUs via CuPy, with automatic
  NumPy fallback on machines without a GPU. See `docs/GPU_AND_CUPY.md` and
  `examples/example_cupy_bp.py`.
- **Decoder auto-routing** (`routing.py`): automatic backend selection (CPU /
  native CUDA / CuPy) based on batch size and hardware availability. See
  `examples/example_auto_routing.py`.
- **Streaming / sliding-window sessions** (`streaming.py`): incremental,
  multi-round decoding sessions with window + commit semantics for long-running
  syndrome streams. See `examples/example_streaming_session.py`.
- Corresponding test suites: `test_gpu_backend.py`, `test_bp_cupy.py`,
  `test_routing.py`, `test_streaming.py`.

### Removed
- Superseded `advanced.py` module and its dedicated tests
  (`test_advanced_decoders.py`, `test_beliefmatching_bridge.py`,
  `test_kimi_findings.py`, `test_stim_circuit_to_check_matrix.py`), folded into
  the new routing/streaming/GPU-backend surface.
- Superseded due-diligence bundle helper scripts (`finalize_bundle.py`,
  `run_due_diligence_wrapper.py`), superseded by `run_due_diligence_bundle.py`.

### Fixed
- `ruff format --check python/` was failing in CI (`tests / ruff-and-mypy`) on
  9 files; reformatted with `ruff format` (lint and mypy were already passing).
- Version bumped to `0.5.9` across `pyproject.toml`, `Cargo.toml`, `Cargo.lock`,
  and the Python runtime fallback version, since PyPI `0.5.7` was already
  published under the prior module layout and cannot be overwritten.

## [0.5.7] - 2026-06-30

### Fixed
- Aligned Python packaging, Cargo metadata, runtime fallback version, and PyPI release bundle at `0.5.7`.
- Verified the Windows CPython 3.11 wheel imports the compiled extension and reports `qector_decoder_v3.__version__ == "0.5.7"`.

## [0.5.0] - 2026-06-23

### Fixed
- **Blossom exactness at large distance (adaptive-k).** `BlossomDecoder` previously
  used a fixed `k=12` candidate cap, which undershot the optimum on large dense
  circuit-level graphs (d ≥ 13–15), producing heavier matchings and a markedly
  worse logical error rate than PyMatching at d=15. The candidate set is now
  **adaptive**, `k = max(12, 4·√n_defects)`, restoring exact-MWPM LER parity with
  PyMatching through **d=15** (`memory_x` and `memory_z`). Locked permanently by
  `test_blossom_adaptive_k_regression.py`, `test_blossom_d15_no_gap.py`,
  `test_blossom_candidate_set_contains_optimal.py`, `test_weight_gap_histogram.py`,
  and `test_defect_count_vs_weight_gap.py`.

### Added
- **QECTOR Workbench** (`qector_decoder_v3.workbench.Workbench`): headless,
  fully-tested controller to load `.stim`/`.dem` files, run cancelable benchmark
  jobs through a FIFO queue, and export JSON/CSV/PDF reports (charts built from
  real artifacts, no fabricated data). Backend detection + environment snapshot.
- **Evidence & reproduction scripts**: `run_due_diligence_bundle.py` (one-command
  evidence bundle with hashes + git commit), `belief_reference_compare.py`,
  `gpu_memory_profile.py`, `auto_backend_calibrate.py`, `leak_test.py`.
- **Provenance**: `benchmarking.capture_environment()` now records `git_commit`, so
  every JSON artifact and report figure points to the exact build it came from
  (replaces "Git commit: unknown").
- **Expanded validation suite** covering: exact-MWPM parity (memory_x/z, p-sweep,
  rounds-sweep), DEM-collapse mathematical equivalence + d=11/d=15 regression
  fixtures (50,484→6,718 and 132,426→17,862), logical-observable / stabilizer-coset
  correctness, belief-matching seed×p grid + reference cross-check, BP-OSD on
  BB[[72,12]]/BB[[144,12,12]]/HGP/bicycle, GPU CPU-bit-identity + fallback +
  calibration, latency percentiles + tail, and memory/leak profiling.
- **Documentation**: README "Validated scope", "When to use which decoder" decision
  matrix, and a permanent "Known limitations" section with honest latency ratios.

### Build
- Refreshed Rust dependencies (`rayon` 1.12, `fastrand` 2.4) and migrated the
  optional `grpc`/`full` stack to `tonic` 0.14 / `prost` 0.14 with a vendored
  `protoc` (`protoc-bin-vendored`), so gRPC builds need no system `protoc`. The
  default wheel features (`opencl`, `cuda` with CPU fallback) are unchanged.

## [0.4.0]

### Added
- `SparseBlossomDecoder` (region-growing, RadixHeap, exact DP for n ≤ 20 with
  Edmonds primal-dual fallback), bit-validated against `BlossomDecoder`.
- Ecosystem layer: `codes`, `dem`, `result`, `backend`, `pymatching_compat`,
  `benchmarking`; belief-matching and BP-OSD decoders; Stim/Sinter compatibility.
- Native CUDA (NVRTC + Driver API) and OpenCL batch decoders with CPU fallback.

### Fixed
- Stim DEM loading uses the correct detector graph (mechanisms = columns,
  detectors = rows), replacing the earlier `stim_compat` heuristic.

## [0.2.0]

- Python + Numba baseline decoder core (pre-Rust rewrite).
