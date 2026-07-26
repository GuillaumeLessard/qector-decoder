# QECTOR Decoder v3

[![CI](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/tests.yml/badge.svg)](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/qector-decoder-v3)](https://pypi.org/project/qector-decoder-v3/)
[![Python](https://img.shields.io/pypi/pyversions/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![License](https://img.shields.io/badge/License-Source_Available-blue)](https://github.com/GuillaumeLessard/qector-decoder/blob/main/LICENSE)

**Production-grade quantum error correction decoding library — Python + Rust.**  
*Copyright © 2026 Guillaume Lessard / iD01t Productions. All Rights Reserved.*

PyMatching-compatible MWPM validation · Belief-matching accuracy mode · BP-OSD for LDPC/qLDPC · CPU/GPU batch decoding · 7-tier self-debugging fallback engine · Ed25519 cryptographic license verification · Artifact-backed benchmark evidence

[Website](https://www.qector.store) · [PyPI](https://pypi.org/project/qector-decoder-v3/) · [Commercial licensing](mailto:admin@qector.store)

---

## Install

```bash
pip install qector-decoder-v3
```

Supported: **Python 3.9–3.13** on Linux x86_64, Windows x64, and macOS arm64.

Optional extras:

```bash
pip install "qector-decoder-v3[stim]"    # Stim/Sinter/PyMatching/LDPC ecosystem
pip install "qector-decoder-v3[bench]"   # Benchmark harness (psutil, matplotlib, scipy)
pip install "qector-decoder-v3[all]"     # Full validation environment
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
from qector_decoder_v3 import BatchDecoder, CUDABatchDecoder

checks = [[0, 1], [1, 2], [2, 3], [3, 4]]
syndromes = np.random.randint(0, 2, size=(4096, 4), dtype=np.uint8)

cpu = BatchDecoder(checks, n_qubits=5)
corrections = cpu.parallel_batch_decode(syndromes)

if CUDABatchDecoder.is_available():
    gpu = CUDABatchDecoder(checks, n_qubits=5)
    corrections = gpu.batch_decode(syndromes)
```

### AutoDecoder — 7-tier self-debugging fallback

```python
from qector_decoder_v3 import AutoDecoder

decoder = AutoDecoder(checks, n_qubits=5)
corrections = decoder.batch_decode(syndromes)

# Inspect backend health
print(decoder._diag.backend_health)
print(decoder._diag.active_backend)
```

### Stim workflow

```python
import stim
from qector_decoder_v3 import BlossomDecoder
from qector_decoder_v3.stim_compat import from_stim_detector_error_model

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=5, rounds=5,
    after_clifford_depolarization=0.005,
)
dem = circuit.detector_error_model(decompose_errors=True)
checks, n_qubits = from_stim_detector_error_model(dem)
decoder = BlossomDecoder(checks, n_qubits)
```

### BP-OSD for LDPC / qLDPC codes

```python
from qector_decoder_v3 import codes
from qector_decoder_v3.bposd import BpOsdDecoder

cx, cz = codes.bivariate_bicycle_code(6, 6, ...)
decoder = BpOsdDecoder(cx.parity_check_matrix(), error_rate=0.05, osd_order=0)
correction = decoder.decode(syndrome)
```

### License verification

```python
import os
from qector_decoder_v3.license import verify_license_token

token = os.environ.get("QECTOR_LICENSE", "")
is_valid = verify_license_token(token)
# Or with explicit email check:
is_valid = verify_license_token(token, customer_email="user@example.com")
```

### Sinter integration

```python
import sinter
from qector_decoder_v3.sinter_compat import qector_sinter_decoders

samples = sinter.collect(
    num_workers=4, tasks=tasks,
    decoders=["qector_belief", "qector_blossom", "qector_unionfind"],
    custom_decoders=qector_sinter_decoders(),
)
```

---

## Decoder families

| Module | Best use | Status |
| --- | --- | --- |
| `UnionFindDecoder` | Low-latency approximate decoding | Stable |
| `FastUnionFindDecoder` | Optimized Union-Find hot path | Stable |
| `BlossomDecoder` | Exact MWPM / PyMatching-parity validation | Stable |
| `SparseBlossomDecoder` | Faster near-optimal matching | Experimental |
| `BeliefMatching` | Correlated-noise accuracy experiments | Research |
| `BpOsdDecoder` | LDPC / qLDPC decoding | Experimental |
| `BatchDecoder` / `CPUBatchDecoder` | CPU batch Monte Carlo sweeps | Stable |
| `CUDABatchDecoder` | CUDA batch decoding | Build/runtime dependent |
| `OpenCLBatchDecoder` | OpenCL batch decoding | Build/runtime dependent |
| `AutoDecoder` | 7-tier self-debugging backend fallback | Stable |
| `PredecodedDecoder` | Easy-syndrome prefiltering | Experimental |
| `DecoderPool` | Multi-process batch decoding | Stable |
| `get_decoder` / `clear_decoder_cache` | Cached decoder factory | Stable |
| `decode_mmap` | Out-of-core memmap decoding | Stable |
| `DecodeResult` / `decode_with_diagnostics` | Structured decode results | Stable |
| `Workbench` | High-level orchestration | Stable |
| `SlidingWindowDecoder` | Multi-round streaming | Experimental |
| `StreamingDecoder` | Continuous streaming sessions | Experimental |
| `HybridDecoder` | Union-Find + Blossom fallback routing | Experimental |
| `LookupTableDecoder` | Precomputed small-code lookup | Experimental |
| `NeuralPredecoder` | Learned predecoder front-end | Research |
| `GNNPredecoder` | Graph neural network predecoder | Research |
| `GNNTrainer` | Training harness for GNNPredecoder | Research |
| `LERBenchmark` | Logical error rate benchmarking | Experimental |
| `stim_compat` | Stim circuit / DEM conversion | Stable utility |
| `sinter_compat` | Sinter custom decoder integration | Stable utility |
| `rest_api` | Local decoding service | Local/partner review |

---

## Self-Auto-Debug Backend Architecture (v0.6.8)

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

Key features:
- **Automatic error trapping**: Hardware exceptions (CUDA OOM, driver crashes, memory limits) are caught, logged, and bypassed transparently.
- **Health scoring**: Each backend tracks its health status. Failed backends are automatically suspended.
- **Seamless recovery**: `reset_backend_health()` re-enables all backends for dynamic recovery.
- **Diagnostic logging**: All fallback events and error details are recorded for debugging.

---

## Licensing & Activation (v0.6.8)

### Ed25519 Cryptographic License Verification

QECTOR uses **offline Ed25519 signature verification** for license tokens. No network calls required.

**Token format**: Self-contained 3-part tokens (`{receipt_id}.{email_b64}.{signature_b64}`) embed the customer email and cryptographic signature for fully offline verification.

| Variable | Description |
| --- | --- |
| `QECTOR_LICENSE` | Set to a valid Ed25519-signed license token to activate |
| `QECTOR_SILENT` | Set to `1` to suppress the startup licensing notice |

**Override tokens**: `academic` and `commercial` accepted for development and testing.

## Tuning environment variables

These change decoder behaviour at construction time. Two of them affect
**matching quality**, and therefore logical error rate — set them deliberately,
and record them alongside any benchmark you publish.

| Variable | Default | Effect |
| --- | --- | --- |
| `QECTOR_BLOSSOM_K_MULT` | `2.0` | Candidate-neighbour multiplier for sparse MWPM: `k = max(12, ceil(mult · sqrt(n_defects)))`. **Affects accuracy.** Lowering it reduces latency but can exclude the optimal partner on dense instances, producing a heavier (sub-optimal) matching. `2.0` is the tuned minimum that preserved exact-MWPM parity at d ≥ 15. |
| `QECTOR_BLOSSOM_INTRA_PAR` | auto | Force intra-decode parallelism for candidate discovery. `0` disables, `1` forces. Unset selects automatically when the graph has ≥ 64 nodes (roughly d ≥ 9 for rotated surface codes). Performance only — output is bit-identical either way. |
| `QECTOR_BLOSSOM_INTRA_THREADS` | unset | Size a dedicated Rayon pool for candidate discovery, independent of the global batch pool. Unset or `< 1` uses the global pool. Performance only. |
| `QECTOR_CUDA_DEVICE_ID` | `0` | Which CUDA device the native batch/BP-OSD decoders bind to. |
| `QECTOR_OPENCL_DEVICE_ALLOW` | unset | Comma-separated substrings matched case-insensitively against OpenCL device names, e.g. `nvidia,geforce`. Unset accepts any device. Use it to avoid selecting an integrated GPU on multi-device hosts. |

Only `QECTOR_BLOSSOM_K_MULT` and `QECTOR_OPENCL_DEVICE_ALLOW` can change results
(matching quality and device selection respectively); the rest are purely
throughput knobs.

### Stripe Integration

Commercial licenses are issued automatically via Stripe Checkout:

1. Customer completes payment at [qector.store](https://www.qector.store)
2. Stripe fires a `checkout.session.completed` webhook
3. The server generates an Ed25519-signed license token
4. Token is delivered to the customer

**Direct purchase**: [Buy Commercial License](https://buy.stripe.com/7sY9AVdwlgoyfse9bYeUU00)

---

## v0.6.9 highlights

| Area | Description |
| --- | --- |
| **Belief matching** | `from_numpy_h` decoders no longer return empty corrections — output is a faithful length-`n_qubits` vector (`H @ corr == syndrome`) |
| **BP-OSD accuracy** | Exact log-domain sum-product BP by default; true combination-sweep OSD-1/2 via `osd_order` |
| **GNN belief matching** | `GNNBeliefMatcher` end-to-end GNN-guided MWPM with faithfulness fallback |
| **Licence hardening** | Malformed tokens return `False` instead of raising; v2 tokens carry tier + expiry inside the signature |
| **Payments** | Dynamic payment methods restored at checkout (Link, wallets, local methods) |
| **Docs** | Tuning env vars documented, incl. which change results vs. throughput; `docs/RELEASING.md` added |

## v0.6.8 highlights

| Area | Description |
| --- | --- |
| **Self-Auto-Debug Backend** | 7-tier fault-tolerant fallback engine with automatic error trapping and health scoring |
| **Ed25519 License Verification** | Offline cryptographic license token validation |
| **Stripe License Fulfillment** | Automated commercial license issuance via Stripe Checkout |
| SparseBlossom bugfix | All decoded syndromes now bit-identical to MWPM |
| BPOSD timeout bugfix | Wall-clock deadline now honored from the first iteration |
| OpenCL health check fix | Child-process `NameError` in `_opencl_health_check()` fixed |
| `k_nearest_via_radix` | Public event-driven candidate-edge discovery |
| MCP server expansion | 5 new tools, expanded decoder info |
| Cross-decoder test suite | Covers all 11 decoder families |
| SafeTensors round-trip tests | Full dtype, shape, and error-path coverage |
| Dead-code elimination | 8 warnings eliminated across the crate |

---

## Validated benchmark evidence

### MWPM parity against PyMatching

Artifact: `benchmark_results/stim_ler_d13_d15.md`

Environment: Windows 10/11 x64, Python 3.11+, QECTOR v0.6.4, PyMatching 2.4+, Stim 1.16+, 20,000 shots/distance.

| Distance | QECTOR Blossom LER | PyMatching LER | QECTOR us/shot | PyMatching us/shot |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 0.00075 | 0.00075 | 820.46 | 81.12 |
| 15 | 0.00050 | 0.00050 | 1965.15 | 203.20 |

Interpretation: QECTOR Blossom matched PyMatching logical-error counts. PyMatching remains faster for standard MWPM latency.

### Belief-matching accuracy

Artifact: `benchmark_results/competitive_belief.md`

| Distance | PyMatching LER | QECTOR MWPM LER | QECTOR Belief LER |
| ---: | ---: | ---: | ---: |
| 5 | 0.00767 | 0.00767 | 0.00500 |
| 7 | 0.00600 | 0.00600 | 0.00300 |

Belief-matching improved observed LER at d=5 and d=7 but is dramatically slower — an accuracy/research mode, not a production latency path.

### GPU bit-identity

Artifact: `benchmark_results/gpu_extensive.json`

| Claim | Result |
| --- | --- |
| CUDA bit-identical to CPU | True |
| OpenCL bit-identical to CPU | True |
| All tested GPU paths faithful | True |

### Native memory profile (distance 13, batch 16,384)

Artifact: `benchmark_results/native_memory.json`

| Decoder | RSS delta MiB |
| --- | ---: |
| `cpu_batch` | 9.41 |
| `blossom` | 5.88 |
| `fast_union_find` | 0.02 |
| `cuda_batch` | 2.67 |

---

## Reproduce benchmarks

```bash
# MWPM / PyMatching comparison
python scripts/competitive_stim_ler.py --distances 3 5 7 9 11 13 15 --shots 40000

# Belief-matching comparison
python scripts/competitive_belief_matching.py --distances 3 5 7 --shots 3000 --no-ref

# GPU correctness
python scripts/gpu_extensive_test.py --distances 3 5 7 9 11 13 --batches 1 64 1024 4096 16384 65536 --error-rate 0.05

# Native memory profile
python scripts/native_memory_profile.py --distances 5 9 13 --batch 16384
```

Benchmark results are hardware, driver, compiler, and workload dependent. Regenerate before quoting performance numbers.

---

## Architecture

```
qector_decoder_v3/
+-- Rust core (proprietary, injected during CI build or under license)
|   +-- Union-Find / Blossom / SparseBlossom engines
|   +-- CPU batch engine (SIMD-accelerated on x86)
|   +-- CUDA / OpenCL batch paths
|   +-- DEM collapse and Stim integration
|
+-- Python layer (open source in this repository)
    +-- __init__.py, backend.py, dem.py
    +-- belief_matching.py, bposd.py
    +-- predecoder.py, codes.py
    +-- stim_compat.py, sinter_compat.py
    +-- qiskit_plugin.py, rest_api.py
    +-- workbench.py
```

---

## REST API (local use only)

```bash
pip install "qector-decoder-v3[stim]" fastapi uvicorn
python -m qector_decoder_v3.rest_api
```

```bash
curl -X POST http://localhost:8000/decode \
  -H "Content-Type: application/json" \
  -d '{"check_to_qubits":[[0,1],[1,2],[2,3],[3,4]],"syndrome":[0,1,0,0]}'
```

For local experiments and controlled deployments only. Not hardened for public SaaS.

---

## Limits and boundaries

| Area | Boundary |
| --- | --- |
| MWPM latency | PyMatching remains faster on standard surface-code MWPM in the provided artifacts |
| Belief-matching | Accuracy/research mode — can improve LER but much slower |
| GPU performance | Correctness artifact-backed for tested machines; speedup not universal |
| OpenCL | Depends on build configuration; confirm locally |
| SparseBlossom | Near-optimal, not exact MWPM — use `BlossomDecoder` for exact |
| UnionFind | Fast approximate path; not universal for arbitrary graphs |
| REST/gRPC/MCP | Not hardened as public SaaS without separate security review |

---

## Licensing

QECTOR Decoder v3 is **source-available**. Personal, academic, educational, and non-commercial research use is allowed. Company use, funded institutional work, SaaS, hosted API deployment, OEM integration, redistribution, paid consulting, or commercial benchmarking requires a commercial license.

- **Pricing & tiers**: [https://www.qector.store/pricing](https://www.qector.store/pricing)
- **Direct purchase**: [Buy via Stripe](https://buy.stripe.com/7sY9AVdwlgoyfse9bYeUU00)
- **Contact**: [admin@qector.store](mailto:admin@qector.store)

### DOI references

- Licensing terms & user manual: [10.5281/zenodo.21363016](https://doi.org/10.5281/zenodo.21363016)
- Performance benchmarks (v0.6.6): [10.5281/zenodo.21339300](https://doi.org/10.5281/zenodo.21339300)
- Architecture whitepaper: [10.5281/zenodo.21320543](https://doi.org/10.5281/zenodo.21320543)

```bibtex
@software{lessard2026qector,
  author  = {Guillaume Lessard},
  title   = {{QECTOR Decoder v3}: Rust/Python Quantum Error Correction Decoding Platform},
  year    = {2026},
  version = {0.6.9},
  url     = {https://www.qector.store},
  note    = {Source-available. Commercial license required for commercial use.}
}
```

---

**Copyright © 2026 Guillaume Lessard / iD01t Productions. All rights reserved.**

[https://www.qector.store](https://www.qector.store) · [admin@qector.store](mailto:admin@qector.store)
