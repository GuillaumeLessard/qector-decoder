# QECTOR API Stability Guide — v0.5.7

QECTOR Decoder v3 is a source-available research and commercial-evaluation platform.
This file separates the APIs that are stable for normal local use from experimental
surfaces that require extra caution.

## Stability levels

| Level | Meaning | Examples |
|---|---|---|
| Stable local API | Expected to remain usable across compatible `0.5.x` releases with bug-fix-level changes only | Core Python decoders, code generators, local source build, benchmark artifact format |
| Supported but workload-sensitive | Supported, but claims must be regenerated for the local code family, hardware, batch size, and dependency set | Stim/PyMatching comparisons, BP-OSD/LDPC experiments, CPU/GPU batch workflows |
| Experimental / preview | Available for research, demos, or partner review but not a compatibility promise | REST service, gRPC, MCP, metrics, hybrid/GNN paths, hosted API, OEM embedding |
| Internal implementation detail | May change without notice | Rust module internals, fallback heuristics, private helper functions |

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
    LookupTableDecoder,
    PredecodedDecoder,          # v0.5.5+ — constructor: (check_to_qubits, n_qubits, backend)
    generate_ring_code_checks,
    generate_repetition_code_checks,
    generate_surface_code_checks,
)
```

Expected input/output model:

```text
check_to_qubits: list[list[int]] or compatible Python sequence
n_qubits:        optional integer when not inferable
syndrome:        one-dimensional NumPy-compatible uint8/int/bool vector
correction:      NumPy-compatible binary vector of length n_qubits
```

The core correctness contract for a syndrome-faithful decoder:

```text
H · correction == syndrome  (mod 2)
```

See `docs/CORRECTNESS_AUDIT.md` for the decoder-by-decoder claim boundary.

## Stable stim_compat entry points (v0.5.7 clarification)

`stim_compat` exposes two parallel entry points. As of v0.5.7 they are correctly
documented as **parallel implementations**, not a Python alias:

| Function | Accepts | Notes |
|---|---|---|
| `from_stim_detector_error_model` | `DetectorErrorModel` or `str` | Core entry point |
| `stim_circuit_to_check_matrix` | `DetectorErrorModel`, `str`, or `stim.Circuit` | Superset — converts `stim.Circuit` via `.detector_error_model(decompose_errors=True)` internally |

Both return `(check_to_qubits, n_qubits)` and produce identical results for
`DetectorErrorModel` input. The prior erroneous `# alias` comment has been
corrected in v0.5.7.

```python
from qector_decoder_v3.stim_compat import (
    from_stim_detector_error_model,   # accepts DetectorErrorModel or str
    stim_circuit_to_check_matrix,     # superset: also accepts stim.Circuit
    to_stim_decoder,
    stim_decoder_from_dem,
)
```

## Stable sinter_compat entry points

```python
from qector_decoder_v3.sinter_compat import (
    QectorSinterDecoder,              # primary Sinter-compatible decoder wrapper
    QectorDecoderWrapper,             # backward-compat alias for QectorSinterDecoder
    qector_sinter_decoders,           # returns dict of named sinter.Decoder instances
)

# standalone single-syndrome decode (v0.5.3+, no Sinter required)
dec = QectorSinterDecoder("blossom")
obs = dec.decode(syndrome, dem=dem)
```

## Stable single-shot additions (v0.5.3+)

| API | Added in | Notes |
|---|---|---|
| `BatchDecoder.decode(syndrome)` | v0.5.3 | 1-row batch wrapper, matches dtype/shape contract of all other decoders |
| `BeliefMatching(H, p=...)` | v0.5.3 | Raw numpy check matrix constructor with uniform prior |
| `QectorSinterDecoder.decode(syndrome, dem)` | v0.5.3 | Standalone decode, DEM cached on first call |

## Supported but workload-sensitive APIs

These APIs are valid but their performance and scientific claims depend on local
context:

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

## NeuralPredecoder limitation (v0.5.4+)

`NeuralPredecoder.train()` raises a clear `RuntimeError` on numpy>=2.0. This is a
known Rust binding issue; `predict()` and `decode()` are unaffected. To train a
model, use an environment with `numpy<2.0` installed until the binding is updated.

## OpenCL false-negative (documented v0.5.5)

`OpenCLBatchDecoder.is_available()` returns `False` on the AMD OCL SDK Light legacy
runtime even though `OpenCL.dll` loads. Root cause: the `ocl` crate's
`Device::list()` fails silently on that runtime. A ctypes-based fallback is
scheduled for a future release.

## Experimental / preview surfaces

Not enterprise-stable in the public `0.5.x` release:

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

See `docs/SERVICE_API_SCHEMA.md`, `docs/SECURITY_DEPLOYMENT.md`, `COMMERCIAL.md`.

## Versioning policy for 0.5.x

```text
Patch releases may fix decoder behaviour, docs, build issues, or benchmark scripts.
Public local Python classes will not be renamed without a compatibility note.
Experimental service surfaces may change schema or behaviour.
Benchmark numbers may be replaced only with regenerated artifacts and clear environment metadata.
Commercial rights remain governed by LICENSE and COMMERCIAL.md.
```

## Safe public wording

Safe:

```text
QECTOR exposes a stable local Python API for source-built decoder experiments
and reproducible benchmark workflows. PyPI wheels are available for CPython 3.9–3.13
on Linux x86_64, Windows x64, and macOS arm64.
```

Unsafe:

```text
QECTOR REST/gRPC APIs are production SaaS APIs.
QECTOR GPU APIs always outperform CPU/PyMatching.
QECTOR hybrid/GNN APIs are stable production decoders.
```
