# API Surfaces and Stability

This document separates stable user-facing surfaces from experimental or deployment-specific surfaces. It is meant to prevent over-promising and to help contributors know where compatibility matters most.

## Stability levels

| Level | Meaning |
|---|---|
| Stable | Intended for normal users and examples. Changes should be documented and minimized within a minor release line. |
| Supported but specialized | Intended for advanced users, benchmarks, or optional feature builds. Regenerate artifacts locally before making claims. |
| Experimental | Useful for research or internal development, but API/behavior may change without compatibility promises. |
| Deployment review required | Network-facing or commercial deployment surfaces that require separate hardening and/or commercial agreement. |

## Python package surface

| Surface | Status | Notes |
|---|---|---|
| `import qector_decoder_v3` | Stable | Main Python package import path |
| `UnionFindDecoder` | Stable | Fast approximate matching-graph decoder path |
| `FastUnionFindDecoder` | Stable | Lower-overhead Union-Find family path |
| `BlossomDecoder` | Stable | Exact weighted MWPM / PyMatching LER parity path on tested graphlike workloads |
| `SparseBlossomDecoder` | Supported but specialized | Near-optimal sparse path, not exact MWPM |
| `BPOSDDecoder` / `BpOsdDecoder` | Supported but specialized | LDPC/qLDPC workflow path; compare locally against reference packages |
| `CPUBatchDecoder` / `BatchDecoder` | Stable for batch workflows | CPU/Rayon batch path |
| `CUDABatchDecoder` / `OpenCLBatchDecoder` | Supported but specialized | Requires local driver/runtime support; bit-identity is separate from speed |
| `SlidingWindowDecoder` / `StreamingDecoder` | Supported but specialized | Multi-round / streaming workflow path |
| `AutoDecoder` | Supported but specialized | Calibrated CPU/GPU routing; hardware-specific |
| `BeliefMatching` | Experimental accuracy mode | Selected correlated workloads only; slower than MWPM |
| GNN / neural / hybrid modules | Experimental | Research-oriented, not a stable product guarantee |
| Workbench modules | Experimental / roadmap | Local validation workstation direction, not shipped enterprise product |

## Rust crate surface

The Rust crate is primarily exposed through PyO3 and Maturin. Direct Rust integration is possible from source, but the compatibility promise is weaker than the Python package API.

| Surface | Status | Notes |
|---|---|---|
| PyO3 module registration in `src/lib.rs` | Stable for Python package build | Required for Python package compatibility |
| `uf_core` internal Union-Find engine | Internal stable core | Single source of truth for UF-family behavior |
| Blossom / MWPM internals | Internal implementation | Correctness contracts matter more than exact internal function names |
| CUDA/OpenCL internals | Feature-gated specialized path | Runtime-driver dependent |
| gRPC/MCP/metrics modules | Deployment review required | Do not treat as stable public hosted-service APIs |

## CLI and source checkout wrappers

| Surface | Status | Notes |
|---|---|---|
| Documented source-build install command | Stable | Must remain accurate for clean clones |
| Import smoke command | Stable | `from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')` |
| Benchmark scripts under `scripts/` | Supported but specialized | Output schemas should be kept reasonably stable when used as public artifacts |
| Reproduction docs under `docs/` | Stable documentation surface | Public claims should cite these commands |

## Network/service surfaces

| Surface | Status | Notes |
|---|---|---|
| REST API | Deployment review required | Local/demo workflow only unless auth, limits, logs, job control, and support terms exist |
| gRPC API | Deployment review required | Optional feature-gated service; schema and auth review required |
| MCP / metrics | Deployment review required | Useful internally, but not a public enterprise contract |
| Docker image | Deployment review required | Convenience packaging, not full hardening by itself |

## Claim boundaries by API area

Safe claims:

```text
QECTOR exposes a Rust/Python decoder platform with stable local package imports.
Core decoder classes support reproducible local experiments.
GPU paths support bit-identity checks on tested configurations.
Benchmark scripts can regenerate artifact-backed claims on local hardware.
```

Unsafe claims without extra artifacts and commercial review:

```text
The REST API is production SaaS.
The gRPC path is an enterprise service contract.
GPU builds are universally faster.
QECTOR is real-time hardware QEC infrastructure.
Experimental hybrid/GNN modules are production-grade decoders.
```

## Versioning guidance

For minor releases:

```text
Keep documented import paths stable when possible.
Update README, docs/REPRODUCE.md, and RELEASE_NOTES.md when user-facing commands change.
Record any benchmark schema change in RELEASE_NOTES.md.
Keep old public claims scoped to the artifact version that supports them.
```
