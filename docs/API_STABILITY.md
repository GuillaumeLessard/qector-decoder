# QECTOR API Stability Guide

QECTOR Decoder v3 is a source-available research and commercial-evaluation platform. This file separates the APIs that are intended to be stable for normal local use from experimental surfaces that require extra caution.

## Stability levels

| Level | Meaning | Examples |
|---|---|---|
| Stable local API | Expected to remain usable across compatible `0.5.x` releases, with bug-fix-level changes only | Core Python decoders, code generators, local source build, benchmark artifact format basics |
| Supported but workload-sensitive | Supported, but claims must be regenerated for the local code family, hardware, batch size, and dependency set | Stim/PyMatching comparisons, BP-OSD/LDPC experiments, CPU/GPU batch workflows |
| Experimental / preview | Available for research, demos, or partner review, but not a compatibility promise | REST service, gRPC, MCP, metrics, hybrid/GNN paths, hosted API, OEM embedding |
| Internal implementation detail | May change without notice | Rust module internals, fallback heuristics, private helper functions, benchmark script internals not documented here |

## Stable local Python APIs

The following APIs are the safest public surface for local use and examples:

```python
from qector_decoder_v3 import (
    UnionFindDecoder,
    FastUnionFindDecoder,
    BlossomDecoder,
    SparseBlossomDecoder,
    BPOSDDecoder,
    BpOsdDecoder,
    CPUBatchDecoder,
    BatchDecoder,
    generate_ring_code_checks,
    generate_repetition_code_checks,
    generate_surface_code_checks,
)
```

Expected input/output model:

```text
check_to_qubits: list[list[int]] or compatible Python sequence
n_qubits: optional integer when not inferable
syndrome: one-dimensional NumPy-compatible uint8/int/bool vector
correction: NumPy-compatible binary vector of length n_qubits
```

The core correctness contract for a syndrome-faithful decoder is:

```text
H · correction == syndrome (mod 2)
```

Use `docs/CORRECTNESS_AUDIT.md` for the exact decoder-by-decoder claim boundary.

## Supported but workload-sensitive APIs

These APIs are valid, but their performance and scientific claims depend strongly on local context:

```text
BeliefMatching
BPOSDDecoder / BpOsdDecoder
AutoDecoder
CUDABatchDecoder
OpenCLBatchDecoder
Stim / Sinter / PyMatching compatibility helpers
benchmark scripts under scripts/
```

Rules:

```text
Do not claim universal speedup.
Do not claim universal accuracy superiority.
Do not reuse Windows reference artifacts as Linux/macOS or GPU-vendor proof.
Regenerate benchmark artifacts on the target workload and target hardware.
Record git commit, OS, Python, Rust, dependencies, seeds, batch size, and raw JSON/CSV outputs.
```

## Experimental / preview surfaces

The following are not enterprise-stable APIs in the public `0.5.x` release:

```text
REST service
Docker REST deployment
gRPC service
MCP service
metrics exporter
hybrid decoder
GNN / neural predecoder components
hosted API
OEM / embedded integration
```

These surfaces are useful for demos, research, and partner evaluation, but they require a deployment review before customer-facing use. See:

```text
docs/SERVICE_API_SCHEMA.md
docs/SECURITY_DEPLOYMENT.md
COMMERCIAL.md
```

## Versioning policy for 0.5.x

During the `0.5.x` line:

```text
Patch releases may fix decoder behavior, docs, build issues, or benchmark scripts.
Public local Python classes should not be renamed without a compatibility note.
Experimental service surfaces may change schema or behavior.
Benchmark numbers may be replaced only with regenerated artifacts and clear environment metadata.
Commercial rights remain governed by LICENSE and COMMERCIAL.md, not by API availability.
```

## Safe public wording

Safe:

```text
QECTOR exposes a stable local Python API for source-built decoder experiments and reproducible benchmark workflows.
```

Unsafe:

```text
QECTOR REST/gRPC APIs are production SaaS APIs.
QECTOR GPU APIs always outperform CPU/PyMatching.
QECTOR hybrid/GNN APIs are stable production decoders.
```
