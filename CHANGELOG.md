# Changelog

All notable changes to QECTOR Decoder v3 are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning. Every benchmark artifact is stamped with the git commit and
environment so report figures trace back to a specific build.

## [Unreleased]

## [0.5.2] - 2026-06-25

### Fixed
- **`sinter_compat`: added `QectorDecoderWrapper` backward-compat alias** for
  `QectorSinterDecoder`. Older docs and examples referencing `QectorDecoderWrapper`
  now import cleanly without `ImportError`.
- **`stim_compat`: added `stim_circuit_to_check_matrix` public alias** for
  `from_stim_detector_error_model`. Both names are now importable and documented.
- **`CUDABatchDecoder.__init__`: clean `RuntimeError` when no CUDA driver present.**
  Previously the Rust layer raised an opaque error at line 439 before Python could
  handle it. Now Python checks `CUDABatchDecoder.is_available()` first and raises a
  descriptive `RuntimeError` with actionable guidance. Always call
  `CUDABatchDecoder.is_available()` before constructing.
- **`_bp_core.sum_product_bp`: silenced `RuntimeWarning: divide by zero in log`.**
  The BP inner loop is now wrapped in `np.errstate(divide='ignore', invalid='ignore')`;
  the eps-clamped `tanh` already guarantees `t != 0` before `log`, so this is purely
  a warning suppression with no change in numeric output.
- **`__version__` fallback aligned with wheel dist-info.** Fallback string bumped
  `0.5.0 -> 0.5.1 -> 0.5.2` to match PyPI dist metadata and Rust core report.

### Added
- **`PYPI_README.md`: full API surface section** documenting `stim_compat`,
  `sinter_compat`, and GPU decoder usage patterns with correct import paths.
- **`PYPI_README.md`: Independent Validation table** (86/87 checks, 100k shots/pt,
  GTX 1660 Ti) and Known Limitations section grounded in measured data.
- **`stim_compat` module docstring**: circuit-level Stim DEM usage example
  explaining why single-round code-capacity noise does not produce surface-code
  threshold distance scaling (by-design; PyMatching shows the same behaviour).


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
