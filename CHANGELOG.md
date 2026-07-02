# Changelog

All notable changes to QECTOR Decoder v3 are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning. Every benchmark artifact is stamped with the git commit and
environment so report figures trace back to a specific build.

## [Unreleased]

## [0.5.8] - 2026-07-02

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
- Version bumped to `0.5.8` across `pyproject.toml`, `Cargo.toml`, `Cargo.lock`,
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
