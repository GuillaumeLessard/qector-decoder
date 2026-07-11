# Changelog

All notable changes to QECTOR Decoder v3 are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning. Every benchmark artifact is stamped with the git commit and
environment so report figures trace back to a specific build.

## [Unreleased]

## [0.6.5] - 2026-07-10

### Fixed
- **mypy clean**: Resolved all 8 type errors across `decode_mmap.py`, `decoder_pool.py`, and `belief_matching.py` — strict type checking now passes on the full Python layer.
- **Test imports**: `test_comprehensive_suite.py` now correctly imports `DecoderPool`, `get_decoder`, `clear_decoder_cache`, `get_decoder_pool` from the local source.
- **CI resilience**: Ensured v0.6.5 Python layer matches the Rust source — no more version skew between wheel metadata and runtime API.
- **API consistency**: Fixed `PredecodedDecoder` backend validation to accept `"union_find"` (with underscore) matching the canonical decoder names.

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
