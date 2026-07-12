# AGENTS.md — QECTOR Decoder v3

Project-specific instructions for working in this repository.

## Project Overview

QECTOR Decoder v3 is a high-performance, source-available Rust/Python platform for quantum error correction (QEC) decoding.

- **Core**: Rust (PyO3 cdylib) implementing Union-Find, Blossom (exact MWPM), Sparse Blossom, BP-OSD, sliding-window, streaming, batch decoders, lookup tables, neural/GNN predecoders, hybrid decoders.
- **Python package**: `qector_decoder_v3` (published on PyPI).
- **Bindings**: maturin + PyO3. Zero-copy NumPy arrays, GIL-free hot paths where possible.
- **Hardware**: CPU (Rayon), optional CUDA batch, optional OpenCL batch. Runtime detection + graceful fallback.
- **Ecosystem**: PyMatching-compatible, Stim/Sinter integration, belief-matching, Qiskit plugin, REST/gRPC + full-featured MCP server (25 tools in the QectorWorkbench companion).
- **Emphasis**: Syndrome faithfulness, CPU/GPU bit-identical results, extensive regression + artifact-backed benchmarks, reproducibility.

**Current version**: 0.6.4 (Cargo + Python packaging).

**Workspace layout** (this checkout):

- `src/` — Rust core (all decoder implementations, batch engines, kernels, utils).
- `python/qector_decoder_v3/` — Python package sources (`__init__.py`, `backend.py`, `belief_matching.py`, `bposd.py`, `stim_compat.py`, `sinter_compat.py`, `workbench.py`, `dem.py`, etc.).
- `python/tests/` — 100+ pytest tests covering correctness, faithfulness, performance, GPU parity, edge cases, benchmarks.
- `proto/` — gRPC `.proto` (only for `grpc` / `full` features).
- `lib/` — Windows import libs (OpenCL etc.).
- `build.rs` — Handles optional protoc for gRPC + link paths.
- `Cargo.toml` / `pyproject.toml` — maturin build.

Note: This appears to be a full-source "build" checkout (src/ contains real implementations + .cu kernels).

## Build & Install (Windows PowerShell)

Use a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip maturin
python -m pip install -e ".[stim,bench]"   # or [all], [cuda], [opencl] etc.
```

- For iterative Rust+Python dev: `maturin develop --release` (or without --release for faster debug).
- Features are controlled in Cargo (default includes opencl + cuda in this workspace).
- gRPC/MCP: `--features full` or `maturin develop --features full`.
- CUDA kernels are compiled at runtime via NVRTC (no special build step for kernels).
- After changes to Rust, re-run `maturin develop` (or restart Python interpreter).

Verify:

```powershell
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder, CUDABatchDecoder; print('CUDA available:', CUDABatchDecoder.is_available() if hasattr(CUDABatchDecoder,'is_available') else 'check'); import qector_decoder_v3; print(qector_decoder_v3.__version__)"
```

## Testing

Run the full Python test suite (primary validation lives here):

```powershell
pytest python/tests/ -q
```

Useful filters:

- `pytest python/tests/ -k "unionfind or blossom or faithfulness"`
- `pytest python/tests/test_gpu* -q` (GPU paths)
- `pytest python/tests/test_bposd* python/tests/test_belief*`
- `pytest python/tests/test_clean_venv_install.py` (install hygiene)
- `pytest python/tests/test_public_api_imports.py test_type_hints.py`

Many tests are **correctness audits** (syndrome faithful, bit-identical across backends, regression against known good results). Do not weaken or skip them lightly.

Rust unit tests (if present):

```powershell
cargo test
```

Before claiming performance or correctness improvements, re-run relevant benchmark scripts (see README) and the due-diligence style tests.

## Coding Conventions & Style

### Rust

- Keep hot paths allocation-free or with reusable scratch (see `uf_core`, batch workspaces).
- Use `rayon` for data-parallel batch paths.
- Feature-gate GPU modules (`#[cfg(feature = "cuda")]`, `#[cfg(feature = "opencl")]`).
- Strong error messages for invalid inputs (see decoder constructors).
- Preserve exact numerical behavior across CPU/GPU; bit-identical results are a hard requirement in many tests.
- Update `build.rs` only when adding new rerun-if-changed or build-time dependencies.

### Python

- NumPy-centric. Use `numpy.typing`, keep dtypes explicit (uint8 for syndromes usually).
- Public API re-exports from the compiled extension + pure-Python shims/fallbacks.
- Backend selection logic lives in `backend.py` (AutoDecoder, calibration).
- Compat layers (`stim_compat`, `sinter_compat`, `pymatching_compat`) must stay faithful to the wrapped behavior.
- Type hints + mypy/ruff clean (dev deps include them).
- Avoid holding GIL during long Rust calls where the extension releases it.

### General

- Prefer adding tests over comments for invariants.
- Benchmark numbers and claims must be reproducible via checked-in scripts + artifacts.
- GPU code must degrade gracefully; never assume a device is present.
- Changes touching DEM handling, observables, or sliding window must touch the corresponding faithfulness tests.

## Common Tasks

- Add a new decoder variant: implement in Rust (new `Py*` class + registration in `lib.rs`), expose in `__init__.py`, add Python wrapper if needed, add tests under `python/tests/`.
- Fix a bug reported by a specific test: run that test in isolation first, then broader faithfulness suite.
- Change Python-only logic (e.g. `belief_matching.py`): no rebuild needed.
- Touch CUDA: ensure `cuda_is_available` and fallback paths still work; run GPU parity tests.
- Update public API: also update `test_public_api_imports.py`, docs if present, and type stubs if any.
- Release prep: version bumps in `Cargo.toml` + `pyproject.toml`, update `__fallback_version__` in `__init__.py`.

## Things to Watch Out For

- The compiled extension (`qector_decoder_v3*.pyd` or .so) must be rebuilt after any Rust change. Stale .pyd is a common source of confusion.
- Feature flags affect what symbols are present (CUDA/OpenCL/gRPC may be missing).
- Windows: OpenCL uses the pre-bundled .lib in `lib/`.
- Hyperedges: some fast decoders reject them (see validation code).
- Memory: batch decoders and large distance codes can be memory-heavy; tests include growth/leak checks.
- Never claim "faster than X" or "better LER" without running the competitive scripts on this machine and recording artifacts.

## When Running Commands

- Prefer `python -m pytest ...` over bare `pytest` for consistency.
- Use the activated venv.
- For long-running tests/benchmarks use the monitor tool or background where appropriate.
- When editing across Rust/Python boundary, verify roundtrips (encode/decode result objects, JSON, etc.).

## Project Rules Priority

These instructions apply in addition to any higher-level or user `AGENTS.md` / rules. Deeper directory rules (if added) take precedence for files within them.

Follow the spirit of the existing extensive test suite: prioritize correctness, reproducibility, and faithful behavior over micro-optimizations.

---

Generated by /init on first session in this workspace. Update this file as conventions evolve.
