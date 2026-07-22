# QECTOR Decoder v3

**Source-available Rust/Python quantum error correction decoding platform.**

QECTOR Decoder v3 provides a Python package backed by a native Rust extension for quantum error correction research and validation workflows. It includes PyMatching-compatible MWPM validation, Union-Find decoding, belief-matching experiments, BP-OSD/qLDPC workflows, batch decoding, optional GPU backend checks, a 7-tier self-auto-debug fallback engine, Ed25519 cryptographic license verification, and Stripe-integrated commercial license fulfillment.

**Companion projects**:

- The public package snapshot focuses on the decoder library, Python API, validation suite, and benchmark evidence.
- Additional desktop, automation, and documentation tooling may be distributed separately from this checkout.

Website: [https://www.qector.store](https://www.qector.store)  
Repository: [https://github.com/GuillaumeLessard/qector-decoder](https://github.com/GuillaumeLessard/qector-decoder)  
Commercial licensing: [https://www.qector.store](https://www.qector.store)

---

## Installation

```bash
pip install qector-decoder-v3
```

Supported package target for the public release workflow:

- Python 3.9 to 3.13
- Linux x86_64 wheels
- Windows x64 wheels
- macOS arm64 wheels
- Source distribution for custom/source builds

Optional research and validation extras:

```bash
# Stim, Sinter, PyMatching, LDPC and belief-matching ecosystem
pip install "qector-decoder-v3[stim]"

# Benchmark and plotting harness
pip install "qector-decoder-v3[bench]"

# Full validation environment
pip install "qector-decoder-v3[all]"
```

---

## Quick start

```python
import numpy as np
from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder

check_to_qubits = [[0, 1], [1, 2], [2, 3], [3, 4]]
n_qubits = 5
syndrome = np.array([0, 1, 0, 0], dtype=np.uint8)

fast = UnionFindDecoder(check_to_qubits, n_qubits)
print(fast.decode(syndrome))

mwpm = BlossomDecoder(check_to_qubits, n_qubits)
print(mwpm.decode(syndrome))
```

### Batch decoding

```python
import numpy as np
from qector_decoder_v3 import BatchDecoder, CUDABatchDecoder

checks = [[0, 1], [1, 2], [2, 3], [3, 4]]
syndromes = np.random.randint(0, 2, size=(4096, 4), dtype=np.uint8)

cpu = BatchDecoder(checks, n_qubits=5)
corrections = cpu.parallel_batch_decode(syndromes)

if CUDABatchDecoder.is_available():
    gpu = CUDABatchDecoder(checks, n_qubits=5)
    corrections = gpu.batch_decode(syndromes)
```

### AutoDecoder with self-auto-debug fallback

```python
from qector_decoder_v3 import AutoDecoder
import numpy as np

checks = [[0, 1], [1, 2], [2, 3], [3, 4]]
decoder = AutoDecoder(checks, n_qubits=5)

# Automatically selects best available backend (CUDA → OpenCL → CPU Rayon → ...)
# If any backend fails, it transparently falls back to the next tier
syndromes = np.random.randint(0, 2, size=(1024, 4), dtype=np.uint8)
corrections = decoder.batch_decode(syndromes)

# Inspect backend health and diagnostics
print(decoder._diag.backend_health)
print(decoder._diag.active_backend)
```

### Stim workflow

```python
import stim
from qector_decoder_v3 import BlossomDecoder
from qector_decoder_v3.stim_compat import from_stim_detector_error_model

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=5,
    rounds=5,
    after_clifford_depolarization=0.005,
)

dem = circuit.detector_error_model(decompose_errors=True)
checks, n_qubits = from_stim_detector_error_model(dem)
decoder = BlossomDecoder(checks, n_qubits)
```

### License verification

```python
import os
from qector_decoder_v3.license import verify_license_token

# Verify an Ed25519-signed license token offline
token = os.environ.get("QECTOR_LICENSE", "")
is_valid = verify_license_token(token)

# Or with explicit email verification
is_valid = verify_license_token(token, customer_email="user@example.com")
```

---

## Included decoder families

| Module | Primary use | Status |
| --- | --- | --- |
| `UnionFindDecoder` | Fast approximate decoding | Stable public API |
| `FastUnionFindDecoder` | Optimized Union-Find path | Stable public API |
| `BlossomDecoder` | Exact MWPM / PyMatching-parity validation | Stable public API |
| `SparseBlossomDecoder` | Faster near-optimal matching | Experimental |
| `BeliefMatching` | Correlated-noise accuracy experiments | Research/accuracy mode |
| `BpOsdDecoder` | LDPC and qLDPC workflows | Experimental |
| `BatchDecoder` / `CPUBatchDecoder` | CPU Monte Carlo sweeps | Stable public API |
| `CUDABatchDecoder` | CUDA batch decoding | Runtime/build dependent |
| `OpenCLBatchDecoder` | OpenCL batch decoding | Runtime/build dependent |
| `AutoDecoder` | 7-tier self-auto-debug backend with automatic fallback | Stable public API |
| `PredecodedDecoder` | Easy-syndrome prefiltering | Experimental |
| `DecoderPool` | Multi-process batch decoding | Stable public API |
| `get_decoder` | Cached decoder factory | Stable public API |
| `decode_mmap` | Out-of-core decoding via memmap | Stable public API |
| `DecodeResult` | Structured decode result | Stable public API |
| `decode_with_diagnostics` | Decode with diagnostics | Stable public API |
| `Workbench` | High-level orchestration | Stable public API |
| `SlidingWindowDecoder` | Multi-round streaming workflows | Experimental |
| `StreamingDecoder` | Continuous streaming decode session | Experimental |
| `HybridDecoder` | Combined Union-Find + Blossom fallback routing | Experimental |
| `LookupTableDecoder` | Precomputed small-code lookup decoding | Experimental |
| `NeuralPredecoder` | Learned predecoder front-end | Research/experimental |
| `GNNPredecoder` | Graph neural network predecoder | Research/experimental |
| `GNNTrainer` | Training harness for `GNNPredecoder` | Research/experimental |
| `LERBenchmark` | Logical error rate benchmarking harness | Experimental |
| `stim_compat` | Stim circuit and DEM conversion | Stable utility |
| `sinter_compat` | Sinter custom decoder integration | Stable utility |
| `rest_api` | Local decoding service | Local/partner review only |

---

## Self-Auto-Debug Backend Architecture (v0.6.7)

`AutoDecoder` implements a **7-tier fault-tolerant self-debugging fallback engine** that automatically selects, monitors, and recovers from hardware failures:

| Tier | Backend | Description |
| --- | --- | --- |
| 1 | CUDA Batch | GPU batch decoding via NVRTC-compiled kernels |
| 2 | OpenCL Batch | Cross-vendor GPU batch decoding |
| 3 | CPU Rayon | Multi-threaded parallel CPU batch decoding |
| 4 | CPU Batch | Single-threaded CPU batch decoding |
| 5 | CPU Single | Per-syndrome CPU decoding |
| 6 | Blossom | Exact MWPM fallback (guaranteed correctness) |
| 7 | Lookup Table / Python | Pure-Python last-resort fallback |

**Key features**:

- **Automatic error trapping**: Hardware exceptions (CUDA OOM, driver crashes, memory limits) are caught, logged, and bypassed transparently.
- **Health scoring**: Each backend tracks its health status in `_diag.backend_health`. Failed backends are automatically suspended.
- **Seamless recovery**: `reset_backend_health()` re-enables all backends for dynamic recovery.
- **Diagnostic logging**: All fallback events and error details are recorded in `_diag.debug_log`.

---

## Licensing & Activation (v0.6.7)

### Ed25519 Cryptographic License Verification

QECTOR uses **offline Ed25519 signature verification** for license tokens. No network calls are required for license validation.

**Token format**: Self-contained 3-part tokens (`{receipt_id}.{email_b64}.{signature_b64}`) embed the customer email and cryptographic signature, enabling fully offline verification.

**Environment configuration**:

| Variable | Description |
| --- | --- |
| `QECTOR_LICENSE` | Set to a valid Ed25519-signed license token to activate |
| `QECTOR_SILENT` | Set to `1` to suppress the startup licensing notice |

**Special override tokens**: `academic` and `commercial` are accepted as valid tokens for development and testing.

### Stripe Integration

Commercial licenses are issued automatically via Stripe Checkout webhooks:

1. Customer completes payment via Stripe Checkout at [qector.store](https://www.qector.store)
2. Stripe fires a `checkout.session.completed` webhook
3. `stripe_webhook_server.py` receives the event and generates an Ed25519-signed license token
4. Token is delivered to the customer and recorded in the local audit log

---

## Evidence-backed positioning

QECTOR Decoder v3 is positioned as a source-available QEC R&D platform, not as a blanket replacement for every mature decoder in every workload.

The repository includes public benchmark artifacts and reproduction scripts for:

- PyMatching-parity logical-error-rate checks on selected surface-code workloads
- belief-matching accuracy experiments on selected workloads
- GPU bit-identity checks against CPU output on a tested NVIDIA machine
- native memory profiling for selected decoder paths

Important boundaries:

- PyMatching remains faster for standard MWPM latency in the checked-in comparison artifacts.
- Belief-matching is an accuracy/research mode and is much slower in the provided experiments.
- GPU availability and performance depend on wheel build features, drivers, hardware, and runtime checks.
- OpenCL support must be confirmed on the target machine or built under the appropriate licensed/custom configuration.
- REST/API surfaces are for local experiments or controlled review unless separately hardened.
- **v0.6.4+**: CPU batch: UnionFind repetition code d=7-9 exceeds 1M shots/s single-core on cp312 Windows 11 x64. Blossom surface d=5 ~40K, d=11 ~4.8K per worker. Hardware dependent.

Full methodology, reproducibility notes, and benchmark artifacts are in the GitHub repository:

[https://github.com/GuillaumeLessard/qector-decoder](https://github.com/GuillaumeLessard/qector-decoder)

---

## GPU availability check

```python
try:
    from qector_decoder_v3 import CUDABatchDecoder
    print("CUDA:", CUDABatchDecoder.is_available() if hasattr(CUDABatchDecoder, "is_available") else False)
except Exception as e:
    print(f"CUDA check skipped: {e}")
try:
    from qector_decoder_v3 import OpenCLBatchDecoder
    print("OpenCL:", OpenCLBatchDecoder.is_available() if hasattr(OpenCLBatchDecoder, "is_available") else False)
except Exception as e:
    print(f"OpenCL: False (CUDA-only wheel) - {e}")
```

---

## v0.6.7 Highlights

| Feature / Fix | Description |
| --- | --- |
| **Self-Auto-Debug Backend** | `AutoDecoder` 7-tier fault-tolerant self-debugging fallback engine (`CUDA` → `OpenCL` → `CPU Rayon` → `CPU Batch` → `CPU Single` → `Blossom` → `Lookup Table` / Python Fallback) with automatic error trapping, health scoring, and seamless recovery |
| **Ed25519 License Verification** | Offline license token validation using Ed25519 signature checks (`verify_license_token`). Supports self-contained 3-part tokens with embedded email. Environment activation via `QECTOR_LICENSE` and `QECTOR_SILENT` |
| **Stripe License Fulfillment** | Complete Stripe Checkout and Webhook integration for automated commercial license issuance upon payment (`stripe_webhook_server.py` and `qector_decoder_v3.stripe_integration`) |
| SparseBlossom Bugfix | Fixed compressed edge set collapse in `SparseBlossomDecoder::grow_regions` — all decoded syndromes are now bit-identical to MWPM |
| BPOSD Timeout Bugfix | Fixed wall-clock deadline initialization in `BPOSDDecoder.bp_decode_timed` |
| LER Benchmark Fix | Rotated-surface generator now emits a proper two-half (X + Z) graphlike code |
| OpenCL Health Check Fix | Fixed `_opencl_health_check()` child-process probe script `NameError` |
| `k_nearest_via_radix` | Public event-driven candidate-edge discovery via `RadixHeap<u32, HeapEvent>` |
| MCP Server Expansion | 5 new tools: `decode_syndrome_blossom`, `batch_decode_blossom`, `run_ler_benchmark`, expanded `get_decoder_info` |
| Cross-decoder test suite | Covers UF / FastUF / LookupTable / SparseBlossom / BP-OSD / SlidingWindow / Streaming / Hybrid |
| SafeTensors round-trip tests | Full coverage: generic + runtime dispatch, dtype mismatch, missing tensors, shape round-trip |
| Dead-code cleanup | Warnings eliminated across the crate (8 → 0) |

---

## v0.6.6 — critical fix, upgrade immediately if on v0.6.5

**v0.6.5 fails to import at all** (`AttributeError` on `OpenCLBatchDecoder`) on every published wheel, because the release build (`--no-default-features --features cuda`) never compiles in OpenCL support, and `__init__.py` had a leftover unguarded reference to it. Fixed in v0.6.6 by removing the dead line; the properly-guarded assignment further down in the file (which already existed) now runs as intended. Verified against a clean install of the exact CI-built wheel.

---

## v0.6.5 Highlights

| Fix | Description |
| --- | --- |
| mypy clean | Resolved all 8 type errors across `decode_mmap.py`, `decoder_pool.py`, and `belief_matching.py` |
| Test suite fix | Genuine `NameError` (`syndrome` → `syndromes`) in the comprehensive test suite's multiprocessing pool test |
| `PredecodedDecoder` fix | Backend validation now accepts `"union_find"` (with underscore), matching canonical decoder names |
| ruff clean | Full repo passes `ruff format --check` and `ruff check` with zero errors |
| `examples/example_batch.py` fix | Was using a weight-4 surface code against Union-Find-only batch decoders (weight ≤2 only); switched to a ring code |

---

## v0.6.4 Highlights

| Feature | Description |
| --- | --- |
| BP-OSD `decode_timed` | Wall-clock deadline for BP iterations; falls back to hard-decision on timeout |
| AVX2 runtime dispatch | CPU batch: UnionFind repetition code d=7-9 >1M shots/s single-core (see Readiness Report). Hardware dependent. |
| Blossom intra-decode parallelism | Rayon-parallelized Blossom matching for multi-shot batches |
| DecoderPool Windows fix | Auto-Rayon fallback on Windows when multi-process pool is unavailable |
| `DecoderPool` | Multi-process batch decoding with automatic worker management |
| `get_decoder` / `clear_decoder_cache` | Cached decoder factory — zero construction cost after first call |
| `decode_mmap` | Out-of-core decoding via memory-mapped NumPy arrays |
| `DecodeResult` / `decode_with_diagnostics` | Structured decode results with per-shot diagnostic metadata |
| `Workbench` | High-level orchestration for multi-decoder comparison and benchmarking |

---

## Licensing

QECTOR Decoder v3 is source-available.

Personal, academic, educational and non-commercial research use is allowed under the repository license. Company use, funded institutional work, SaaS, hosted API deployment, OEM integration, redistribution, paid consulting, or commercial benchmarking requires a commercial license.

Commercial licensing:

[https://www.qector.store](https://www.qector.store)

Contact:

<admin@qector.store>

---

## Citation

```bibtex
@software{lessard2026qector,
  author  = {Guillaume Lessard},
  title   = {{QECTOR Decoder v3}: Rust/Python Quantum Error Correction Decoding Platform},
  year    = {2026},
  version = {0.6.7},
  url     = {https://www.qector.store},
  note    = {Source-available. Commercial license required for commercial use.}
}
```
