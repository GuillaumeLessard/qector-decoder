# API Surfaces and Stability — v0.5.7

This document separates stable user-facing surfaces from experimental or
deployment-specific surfaces. It prevents over-promising and helps contributors
know where compatibility matters most.

## Stability levels

| Level | Meaning |
|---|---|
| Stable | Intended for normal users and examples. Changes documented and minimized within a minor release line. |
| Supported but specialized | Intended for advanced users, benchmarks, or optional feature builds. Regenerate artifacts locally before making claims. |
| Experimental | Useful for research or internal development but API/behaviour may change without compatibility promises. |
| Deployment review required | Network-facing or commercial deployment surfaces requiring separate hardening and/or commercial agreement. |

## Python package surface

| Surface | Status | Notes |
|---|---|---|
| `import qector_decoder_v3` | Stable | Main Python package import path |
| `UnionFindDecoder` | Stable | Fast approximate matching-graph decoder path |
| `FastUnionFindDecoder` | Stable | Lower-overhead Union-Find family path |
| `BlossomDecoder` | Stable | Exact weighted MWPM / PyMatching LER parity path on tested graphlike workloads |
| `SparseBlossomDecoder` | Supported but specialized | Near-optimal sparse path, not exact MWPM |
| `BPOSDDecoder` / `BpOsdDecoder` | Supported but specialized | LDPC/qLDPC workflow path; compare locally against reference packages |
| `CPUBatchDecoder` / `BatchDecoder` | Stable for batch workflows | CPU/Rayon batch path — includes `.decode()` single-shot (v0.5.3+) |
| `LookupTableDecoder` | Stable | Precomputed table for small codes, O(1) lookup |
| `PredecodedDecoder` | Stable (v0.5.5+) | Constructor: `(check_to_qubits, n_qubits, backend)` — provides `.batch_decode(syndromes)` |
| `CUDABatchDecoder` / `OpenCLBatchDecoder` | Supported but specialized | Requires local driver/runtime support; always call `is_available()` first |
| `SlidingWindowDecoder` / `StreamingDecoder` | Supported but specialized | Multi-round / streaming workflow path |
| `AutoDecoder` | Supported but specialized | Calibrated CPU/GPU routing; hardware-specific |
| `BeliefMatching` | Experimental accuracy mode | Selected correlated workloads only; also accepts raw numpy H (v0.5.3+) |
| `NeuralPredecoder` | Experimental | `train()` requires numpy<2.0; `predict()` / `decode()` unaffected on any numpy version |
| GNN / hybrid modules | Experimental | Research-oriented, not a stable product guarantee |
| Workbench modules | Experimental / roadmap | Local validation workstation direction |

## stim_compat surface (v0.5.7 clarification)

| Entry point | Input scope | Status |
|---|---|---|
| `from_stim_detector_error_model` | `DetectorErrorModel` or `str` | Stable |
| `stim_circuit_to_check_matrix` | `DetectorErrorModel`, `str`, or `stim.Circuit` | Stable — parallel implementation (not alias); superset of above |
| `to_stim_decoder` | — | Stable |
| `stim_decoder_from_dem` | — | Stable |

Both `from_stim_detector_error_model` and `stim_circuit_to_check_matrix` return
identical `(check_to_qubits, n_qubits)` for `DetectorErrorModel` input.
`stim_circuit_to_check_matrix` additionally handles `stim.Circuit` by calling
`.detector_error_model(decompose_errors=True)` internally. They are parallel
implementations with separate code objects, not a Python alias.

## sinter_compat surface

| Entry point | Status | Notes |
|---|---|---|
| `QectorSinterDecoder` | Stable | Primary Sinter-compatible wrapper |
| `QectorDecoderWrapper` | Stable | Backward-compat alias for `QectorSinterDecoder` |
| `qector_sinter_decoders()` | Stable | Returns dict of named `sinter.Decoder` instances; raises `ImportError` without sinter |
| `QectorSinterDecoder.decode(syndrome, dem)` | Stable (v0.5.3+) | Standalone decode, DEM cached on first call |

## PyPI wheel surface

Published wheels for v0.5.7:

| Platform | Python | Wheel |
|---|---|---|
| Linux x86\_64 | 3.9 – 3.13 | Published |
| Windows x64 | 3.9 – 3.13 | Published |
| macOS arm64 | 3.9 – 3.13 | Published |
| macOS x86\_64 | 3.9 – 3.13 | Not published in v0.5.x |
| CPython free-threaded `cp313t` | — | Not published in v0.5.x |

## Rust crate surface

The Rust crate is primarily exposed through PyO3 and Maturin. Direct Rust
integration is possible from source but the compatibility promise is weaker than
the Python package API.

| Surface | Status | Notes |
|---|---|---|
| PyO3 module registration in `src/lib.rs` | Stable for Python package build | Required for Python package compatibility |
| `uf_core` internal Union-Find engine | Internal stable core | Single source of truth for UF-family behaviour |
| Blossom / MWPM internals | Internal implementation | Correctness contracts matter more than exact internal function names |
| CUDA/OpenCL internals | Feature-gated specialized path | Runtime-driver dependent |
| gRPC/MCP/metrics modules | Deployment review required | Do not treat as stable public hosted-service APIs |

## CLI and source checkout wrappers

| Surface | Status | Notes |
|---|---|---|
| PyPI install command | Stable | `pip install qector-decoder-v3` |
| Source-build install command | Stable | Must remain accurate for clean clones; see `INSTALL.md` |
| Import smoke command | Stable | `from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')` |
| Benchmark scripts under `scripts/` | Supported but specialized | Output schemas reasonably stable for public artifacts |
| Reproduction docs under `docs/` | Stable documentation surface | Public claims should cite these commands |

## Network/service surfaces

| Surface | Status | Notes |
|---|---|---|
| REST API | Deployment review required | Local/demo workflow only |
| gRPC API | Deployment review required | Optional feature-gated; schema and auth review required |
| MCP / metrics | Deployment review required | Useful internally, not a public enterprise contract |
| Docker image | Deployment review required | Convenience packaging, not full hardening |

## Versioning guidance

For minor releases:

```text
Keep documented import paths stable when possible.
Update README, INSTALL.md, docs/REPRODUCE.md, and RELEASE_NOTES.md when user-facing commands change.
Record any benchmark schema change in RELEASE_NOTES.md.
Keep old public claims scoped to the artifact version that supports them.
```
