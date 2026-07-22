# AGENTS.md — QECTOR Decoder v3

Project-specific instructions for working in this repository.

## 🛡️ Licensing, IP & Compliance (CRITICAL)
- **License Model**: Source-available. Free for personal, academic, educational, and non-commercial research use only.
- **Commercial Use**: Institutional, lab, startup, SaaS, OEM, or product-integration use **requires a paid commercial license**.
- **Official Channels**: All commercial inquiries, pilot requests, and source-review access must be directed to **admin@qector.store** or **https://qector.store/pricing**.
- **Provenance**: The proprietary Rust core and licensing terms are protected and timestamped via controlled archival (e.g., Zenodo DOIs). 
- **Agent Rule**: If asked about licensing, sharing the core, or commercial use, *always* direct the user to the pricing page and official contact. **Never** suggest open-sourcing the Rust core, bypassing the commercial license, or removing copyright notices.

## Project Overview

QECTOR Decoder v3 is a high-performance, source-available Rust/Python platform for quantum error correction (QEC) decoding.

- **Core**: Rust (PyO3 cdylib) implementing Union-Find, Blossom (exact MWPM), Sparse Blossom, BP-OSD, sliding-window, streaming, batch decoders, lookup tables, neural/GNN predecoders, hybrid decoders.
- **Python package**: `qector-decoder-v3` (published on PyPI).
- **Bindings**: maturin + PyO3. Zero-copy NumPy arrays, GIL-free hot paths where possible.
- **Hardware**: CPU (Rayon), optional CUDA batch, optional OpenCL batch. Runtime detection + graceful fallback.
- **Ecosystem**: PyMatching-compatible, Stim/Sinter integration, belief-matching, Qiskit plugin, REST/gRPC + full-featured MCP server.
- **Emphasis**: Syndrome faithfulness, CPU/GPU bit-identical results, extensive regression + artifact-backed benchmarks, reproducibility.

**Current version**: 0.6.7 (Cargo + Python packaging).

**Workspace layout** (this checkout):

- `src/` — Rust core stub. ⚠️ *The real proprietary Rust core is injected during trusted CI/release builds via secrets. Do not attempt to distribute or expose this directory publicly.*
- `python/qector_decoder_v3/` — Python package sources (`__init__.py`, `backend.py`, `belief_matching.py`, `bposd.py`, `stim_compat.py`, `sinter_compat.py`, `workbench.py`, `dem.py`, etc.).
- `python/tests/` — 100+ pytest tests covering correctness, faithfulness, performance, GPU parity, edge cases, benchmarks.
- `proto/` — gRPC `.proto` (only for `grpc` / `full` features).
- `lib/` — Windows import libs (OpenCL etc.).
- `build.rs` — Handles optional vendored protoc for gRPC + link paths.
- `Cargo.toml` / `pyproject.toml` — maturin build configuration.

## Build & Install (Windows PowerShell / Linux / macOS)

Use a virtual environment.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip maturin
python -m pip install -e ".[stim,bench]"   # or [all], [cuda], [opencl] etc.
