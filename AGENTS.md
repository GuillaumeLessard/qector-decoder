# Repository Guidelines

## Project Structure & Module Organization

This repository packages QECTOR Decoder v3 as a Rust/Python project. Python source lives in `python/qector_decoder_v3/`, with public modules such as `backend.py`, `dem.py`, `routing.py`, and `workbench.py`. Python tests are in `python/tests/` and follow the `test_*.py` pattern configured in `pyproject.toml`. Rust packaging and extension metadata are in `Cargo.toml`, `Cargo.lock`, `build.rs`, and `src/lib.rs`; the public `src/` tree is a stub for licensed source builds. Examples are under `examples/`, benchmark and validation utilities under `scripts/`, protobuf definitions under `proto/`, and design/reproducibility notes under `docs/`.

## Build, Test, and Development Commands

Create a local environment before development:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip maturin
.\.venv\Scripts\python.exe -m pip install -e ".[dev,stim,bench]"
```

Build the Python extension in-place with:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features
```

Run the full Python test suite with `.\.venv\Scripts\python.exe -m pytest python/tests -q --tb=short`. Run targeted tests by passing a file, for example `python/tests/test_backend.py`. Use `cargo test --release --lib` for Rust library tests when licensed Rust sources are available.

## Coding Style & Naming Conventions

Python targets 3.9+ and uses Ruff with a 120-character line length. Keep modules and functions in `snake_case`, classes in `PascalCase`, and tests named `test_<behavior>.py` with `test_<expected_result>` functions. Preserve lazy optional imports for GPU and ecosystem dependencies. Rust uses edition 2021; prefer `cargo fmt` style and feature-gated code for optional CUDA, OpenCL, and gRPC paths.

## Testing Guidelines

Use `pytest` and `hypothesis` where property coverage is useful. Add tests beside related coverage in `python/tests/`, especially for decoder correctness, API compatibility, fallback behavior, and reproducibility commands. GPU tests must tolerate unavailable hardware unless explicitly marked or guarded by availability checks.

## Commit & Pull Request Guidelines

Git history is unavailable in this environment, so use concise imperative commit subjects, for example `Fix batch decode dtype handling`. Pull requests should include a scoped description, commands run, environment details for benchmark claims, linked issues when applicable, and screenshots or artifacts for Workbench or report-output changes. Follow `CONTRIBUTING.md` for licensing, benchmark evidence, and security disclosure rules.

## Security & Configuration Tips

Do not publish exploit details in issues; follow `SECURITY.md`. Treat performance claims as hardware- and build-specific. Include raw outputs, hashes, and environment blocks for benchmark submissions as described in `docs/REPRODUCIBILITY_CHECKLIST.md`.
