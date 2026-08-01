# QECTOR Decoder v3

[![CI](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/tests.yml/badge.svg)](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/qector-decoder-v3)](https://pypi.org/project/qector-decoder-v3/)
[![Python](https://img.shields.io/pypi/pyversions/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![License](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-blue)](https://github.com/GuillaumeLessard/qector-decoder/blob/main/LICENSE)

**Production-grade quantum error correction decoding library — Python + Rust.**  
*Copyright © 2026 Guillaume Lessard / iD01t Productions. All Rights Reserved.*

## 🚀 Support QECTOR Development

QECTOR is source-available and developed independently. Non-commercial use is free — sponsorship
and commercial licences are what keep the decoder maintained.

[![Sponsor qectorlab](https://img.shields.io/badge/GitHub_Sponsors-qectorlab-ea4aaa?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/qectorlab)
[![Commercial licence](https://img.shields.io/badge/Commercial_Licence-Buy_via_Stripe-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/6oU00l77Xc8ifsegEqeUU07)
[![Pricing](https://img.shields.io/badge/Pricing_&_Tiers-qector.store-0A7CFF?style=for-the-badge&logo=quantconnect&logoColor=white)](https://www.qector.store/pricing)

| Channel | Who it's for |
|---|---|
| [GitHub Sponsors](https://github.com/sponsors/qectorlab) | Individuals and companies funding ongoing development |
| [Commercial licence](https://www.qector.store/pricing) | Required for company, SaaS, OEM or funded institutional use — see [COMMERCIAL.md](https://github.com/GuillaumeLessard/qector-decoder/blob/main/COMMERCIAL.md) |
| [Direct purchase](https://buy.stripe.com/6oU00l77Xc8ifsegEqeUU07) | Immediate Stripe checkout, licence issued automatically |
| [admin@qector.store](mailto:admin@qector.store) | Site licences, custom terms, academic partnerships |


PyMatching-compatible MWPM validation · Belief-matching accuracy mode · BP-OSD for LDPC/qLDPC · CPU/GPU batch decoding · 7-tier self-debugging fallback engine · Ed25519 cryptographic license verification · Artifact-backed benchmark evidence

[Website](https://www.qector.store) · [PyPI](https://pypi.org/project/qector-decoder-v3/) · [Commercial licensing](mailto:admin@qector.store)

---

## Install

```bash
pip install qector-decoder-v3
```

Supported: **Python 3.9–3.13** (`requires-python = ">=3.9"`) on Linux x86_64,
Windows x64, and macOS arm64.

Each release publishes **15 binary wheels** — CPython 3.9/3.10/3.11/3.12/3.13 ×
`win_amd64` / `manylinux_2_17_x86_64` / `macosx_11_0_arm64`. There is **no sdist
and no aarch64, musllinux, or macOS x86_64 wheel**, so `pip install` on any other
platform will fail rather than fall back to a source build. Those targets need a
local build from a licensed source checkout.

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

**Pass the DEM's weights to the GPU.** Without `edge_weights` the GPU kernels run
unweighted cluster growth, which cannot distinguish a `p = 1e-4` mechanism from a
`p = 1e-2` one — on circuit-level noise that costs several times the logical
error rate, no matter how fast the GPU is:

```python
from qector_decoder_v3 import dem

model = dem.from_stim(circuit.detector_error_model(decompose_errors=True))
graph = model.collapse_to_graph()

gpu = CUDABatchDecoder(
    graph.check_to_qubits(),
    graph.num_errors,
    graph.weights().tolist(),   # log((1-p)/p) per mechanism
)
```

How much it costs, measured: on `surface_code:rotated_memory_x`, circuit-level
noise `p = 0.005`, `rounds = d`, `d = 11`, the **unweighted** GPU kernels score
LER 0.0438 (CUDA) and 0.0434 (OpenCL) against **0.0149** for the weighted CPU
`UnionFindDecoder` and **0.0062** for PyMatching — roughly 3× the logical error
rate, which no amount of throughput buys back. Throughput runs the other way:
unweighted OpenCL decodes that same `d = 11` at **9.6 µs/shot** versus 138 µs/shot
for the weighted CPU core and 82 µs/shot for PyMatching.

*Provenance:* `scripts/full_decoder_benchmark.py`, run 2026-07-30 against QECTOR
0.7.0 on Windows 11 / AMD Zen 2 / Python 3.11.9, seed 20260730, ≤20,000 shots per
cell. The harness trims the shot count per decoder to a time budget and records
the trimmed count, so rows are not equally precise — read `shots` and
`ler_ci95_*` before comparing two rows. Regenerate it yourself; the JSON it
writes carries its own environment and parameter block.

The weighted GPU kernel is the accuracy option. Its logical error rate is now
measured and published: `official_benchmark_results.*` carries
`qector_cuda`/`qector_opencl` in both configurations, scored through the same
circuit-level pipeline as every other decoder. Weighting restores distance
scaling — LER 0.026 at `d = 5`, 0.010 at `d = 9`, 0.007 at `d = 13`, against a
flat 0.059 / 0.048 / 0.035 unweighted — at roughly 32× the cost per shot. See
the benchmark table below and `docs/GPU_AND_CUPY.md`.

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

`DemModel.make_decoder` builds any shipped decoder family straight from the
model, already carrying its weights — enumerate them with
`DemModel.DECODER_KINDS`:

```python
from qector_decoder_v3 import dem

graph = dem.from_stim(circuit.detector_error_model(decompose_errors=True)).collapse_to_graph()

for kind in ("union_find", "fast_union_find", "blossom", "sparse_blossom",
             "bp_osd", "lookup_table", "hybrid_cascade", "ambiguity_cluster"):
    decoder = graph.make_decoder(kind)

# two_stage decodes the X and Z sectors separately, so it needs the sector of
# each detector -- a DEM does not record it:
#   graph.make_decoder("two_stage", check_types=[...])
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

### License Keys (v0.7.0)

```python
from qector_decoder_v3 import set_license_key, set_license_key_file, get_license_info

set_license_key("QECT-PRO-your-key")        # raises ValueError if the key is rejected
set_license_key_file("/path/to/license.key")  # or load it from a file

info = get_license_info()
print(f"Tier: {info['tier']}  status: {info['key_status']}")
```

The core also resolves a key on its own, in this order: `QECTOR_LICENSE_KEY`,
then `QECTOR_LICENSE_FILE`, then `~/.qector/license.key`. Prefer a file in
deployments — the key then never appears in a process listing or shell history.
Check `info["key_status"] == "valid"`, not just the tier: a `QECTOR_LICENSE_FILE`
that is set but unreadable is reported as an *invalid* key rather than silently
falling back to Community.

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
| `CUDABatchDecoder` | CUDA batch decoding (optional `edge_weights`) | Build/runtime dependent |
| `CUDABpOsdDecoder` | CUDA BP-OSD batch decoding | Build/runtime dependent |
| `OpenCLBatchDecoder` | OpenCL batch decoding (optional `edge_weights`) | Build/runtime dependent |
| `SpaceTimeDecoder` | 3D space-time (multi-round) decoding | Experimental |
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

## Licensing & Activation (v0.7.0)

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

**Direct purchase**: [Buy Commercial License](https://buy.stripe.com/6oU00l77Xc8ifsegEqeUU07)

---

## v0.7.0 highlights

| Area | Description |
| --- | --- |
| **`qector` CLI** | `qector decode` / `bench` / `serve`, plus `qector-doctor` — a 15-check environment diagnostic that tells you *why* a decoder is unavailable instead of failing at decode time |
| **Ecosystem entry points** | Five Sinter decoders and the qiskit-qec plugin are now registered entry points, so `sinter.collect(decoders=["qector_blossom", ...])` works without `custom_decoders=` |
| **`pymatching` shim** | `from qector_decoder_v3.pymatching import Matching` — the submodule spelling, not only the attribute |
| **New decoder families** | `AmbiguityClusterDecoder` (BP + \|LLR\| partition + exact per-cluster enumeration), `TwoStageDecoder` (X sector, propagate, Z sector), `ColourCodeDecoder` (BP-OSD on the *undecomposed* hypergraph — matching is not a correct colour-code decoder) |
| **Relay-BP** | Layered serial BP schedule for qLDPC (`bp_method="relay"`); each check sees the freshest messages |
| **Weighted Union-Find on the GPU** | `CUDABatchDecoder` and `OpenCLBatchDecoder` accept `edge_weights` and run adaptive weighted growth; both kernels agree, which is the cross-check that the port is faithful |
| **`DemModel.make_decoder`** | Covers all nine shipped families, not five — a DEM is the entry point real circuit-level workloads use |
| **Belief matching** | `from_numpy_h` decoders no longer return empty corrections — output is a faithful length-`n_qubits` vector (`H @ corr == syndrome`) |
| **BP-OSD accuracy** | Exact log-domain sum-product BP by default; true combination-sweep OSD-1/2 via `osd_order` |
| **Rust core: crash safety** | Six panic-to-abort paths removed — gRPC and CUDA mutex-poison propagation, swallowed CUDA async errors, `Bernoulli::new` unwrap, cascade-decoder `expect`. Under `panic = "abort"` each of these killed the host process |
| **Licence hardening** | Malformed tokens return `False` instead of raising; v2 tokens carry tier + expiry inside the signature; `QECTOR_LICENSE_FILE` and `~/.qector/license.key` are read, and an unreadable file reports *invalid* rather than silently dropping to Community |
| **Benchmark honesty** | The pre-v0.7.0 comparison tables are withdrawn; `ler.assert_comparable` now blocks cross-noise-model comparisons at the source |

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

## Benchmark evidence

### Withdrawn: the pre-v0.7.0 comparison tables

**Four benchmark tables that stood here — MWPM parity vs PyMatching at d=13/15,
belief-matching LER at d=5/7, GPU bit-identity, and the native memory profile —
are withdrawn. Do not cite them.**

Two independent reasons, either sufficient:

1. **Incompatible methodologies in one table.** The comparison ran QECTOR under
   **code-capacity** noise and PyMatching under **circuit-level** noise, then
   printed both LER columns side by side. Those two numbers are not comparable,
   so the "parity" the tables reported was an artifact of the harness, not a
   property of the decoders.
2. **The artifacts are unobtainable.** Each table cited a file under
   `benchmark_results/` — a path that is in `.gitignore` and has never been part
   of any published commit or wheel. The files are also no longer on disk. Nobody
   could have checked the numbers even when they were displayed.

The numbers are not being quietly deleted; they are being retracted, because the
method that produced them cannot support the claim they were used to make.
`qector_decoder_v3.ler` now tags every run with its noise model and refuses
cross-model comparisons through `assert_comparable`, so this class of error
cannot recur silently.

**What replaces them:** `scripts/regenerate_benchmark_artifacts.py` and
`scripts/run_custom_comparison_benchmark.py` both drive every decoder through
one circuit-level pipeline — `ler.estimate_ler_circuit_level`, one Stim circuit,
one decomposed DEM, one detector/observable sample set per cell, one
`decode_batch` resolver, scored against the circuit's own logical observables —
and stamp the result with its methodology, git commit, tree-dirty flag,
parameters and dependency versions. `ler.assert_comparable` gates the rows
before they are written.

```bash
python scripts/regenerate_benchmark_artifacts.py --dry-run   # show the plan
python scripts/regenerate_benchmark_artifacts.py --yes       # ~1.6M decodes

# QECTOR vs PyMatching vs ldpc, with a per-cell time budget:
python scripts/run_custom_comparison_benchmark.py \
    --distances 3,5,7,9,11,13,15 --shots 1000,5000,10000,50000,100000 --p 0.005
```

### Indicative circuit-level run (not a publication run)

`official_benchmark_results.{json,csv,md,pdf}` in the repo root hold a 77-cell
run at `p = 0.005`, `seed = 1`, `d ∈ {3..15}`, shots up to 100,000. Read it as
indicative only: it was taken on a **developer workstation that was not
quiesced**, and its provenance block records `git_tree_dirty: true`. A further
93 cells exceeded the per-cell decode budget and are listed as *not measured*,
carrying their measured probe rate and projected cost — no cell is extrapolated.

Largest shot count measured per cell. Throughput is decode time only; LER is
per shot with a 95% Wilson interval. Every row is one
`estimate_ler_circuit_level` call on the same circuit, DEM and samples.

| d | Decoder | Shots | Throughput (dec/s) | LER | 95% CI |
| ---: | --- | ---: | ---: | ---: | --- |
| 3 | PyMatching 2 | 100,000 | 2,478,782 | 0.01891 | [0.01808, 0.01977] |
| 3 | `qector_blossom` | 100,000 | 2,337,448 | 0.01891 | [0.01808, 0.01977] |
| 3 | `qector_unionfind` | 100,000 | 4,953,928 | 0.02210 | [0.02121, 0.02303] |
| 3 | `qector_cuda` (GPU, unweighted) | 100,000 | 1,446,123 | 0.02215 | [0.02126, 0.02308] |
| 3 | `qector_cuda` (GPU, weighted) | 100,000 | 1,201,288 | 0.02201 | [0.02112, 0.02294] |
| 3 | `qector_opencl` (GPU, unweighted) | 100,000 | 1,286,159 | 0.02215 | [0.02126, 0.02308] |
| 3 | `qector_opencl` (GPU, weighted) | 100,000 | 1,554,014 | 0.02201 | [0.02112, 0.02294] |
| 3 | ldpc BP-OSD | 50,000 | 2,399 | 0.01938 | [0.01821, 0.02063] |
| 5 | PyMatching 2 | 100,000 | 340,202 | 0.01596 | [0.01520, 0.01676] |
| 5 | `qector_blossom` | 100,000 | 111,843 | 0.01596 | [0.01520, 0.01676] |
| 5 | `qector_unionfind` | 100,000 | 694,192 | 0.02645 | [0.02547, 0.02746] |
| 5 | `qector_cuda` (GPU, unweighted) | 100,000 | 150,505 | 0.06094 | [0.05947, 0.06244] |
| 5 | `qector_cuda` (GPU, weighted) | 100,000 | 69,156 | 0.03182 | [0.03075, 0.03293] |
| 5 | `qector_opencl` (GPU, unweighted) | 100,000 | 144,886 | 0.06094 | [0.05947, 0.06244] |
| 5 | `qector_opencl` (GPU, weighted) | 100,000 | 61,421 | 0.03182 | [0.03075, 0.03293] |
| 5 | ldpc BP-OSD | 1,000 | 113 | 0.02100 | [0.01378, 0.03189] |
| 7 | PyMatching 2 | 100,000 | 97,941 | 0.01220 | [0.01154, 0.01290] |
| 7 | `qector_blossom` | 100,000 | 20,975 | 0.01233 | [0.01166, 0.01303] |
| 7 | `qector_unionfind` | 100,000 | 173,669 | 0.02042 | [0.01956, 0.02132] |
| 7 | `qector_cuda` (GPU, unweighted) | 100,000 | 45,353 | 0.04274 | [0.04150, 0.04401] |
| 7 | `qector_cuda` (GPU, weighted) | 10,000 | 13,552 | 0.01800 | [0.01557, 0.02080] |
| 7 | `qector_opencl` (GPU, unweighted) | 100,000 | 244,318 | 0.04274 | [0.04150, 0.04401] |
| 7 | `qector_opencl` (GPU, weighted) | 10,000 | 8,693 | 0.01800 | [0.01557, 0.02080] |
| 9 | PyMatching 2 | 100,000 | 42,295 | 0.00878 | [0.00822, 0.00938] |
| 9 | `qector_blossom` | 50,000 | 2,649 | 0.00906 | [0.00827, 0.00993] |
| 9 | `qector_unionfind` | 100,000 | 44,420 | 0.01732 | [0.01653, 0.01815] |
| 9 | `qector_cuda` (GPU, unweighted) | 50,000 | 18,273 | 0.04648 | [0.04467, 0.04836] |
| 9 | `qector_cuda` (GPU, weighted) | 10,000 | 3,382 | 0.01400 | [0.01188, 0.01650] |
| 9 | `qector_opencl` (GPU, unweighted) | 100,000 | 107,529 | 0.04663 | [0.04534, 0.04795] |
| 9 | `qector_opencl` (GPU, weighted) | 10,000 | 2,043 | 0.01400 | [0.01188, 0.01650] |
| 11 | PyMatching 2 | 100,000 | 21,511 | 0.00647 | [0.00599, 0.00699] |
| 11 | `qector_blossom` | 10,000 | 890 | 0.00780 | [0.00625, 0.00972] |
| 11 | `qector_unionfind` | 50,000 | 9,722 | 0.01678 | [0.01569, 0.01794] |
| 11 | `qector_cuda` (GPU, unweighted) | 10,000 | 13,621 | 0.04400 | [0.04015, 0.04820] |
| 11 | `qector_cuda` (GPU, weighted) | 1,000 | 330 | 0.01000 | [0.00544, 0.01831] |
| 11 | `qector_opencl` (GPU, unweighted) | 50,000 | 53,218 | 0.04222 | [0.04049, 0.04402] |
| 11 | `qector_opencl` (GPU, weighted) | 1,000 | 338 | 0.01000 | [0.00544, 0.01831] |
| 13 | PyMatching 2 | 100,000 | 12,096 | 0.00445 | [0.00406, 0.00488] |
| 13 | `qector_blossom` | 5,000 | 293 | 0.00380 | [0.00243, 0.00593] |
| 13 | `qector_unionfind` | 10,000 | 2,571 | 0.01400 | [0.01188, 0.01650] |
| 13 | `qector_cuda` (GPU, unweighted) | 10,000 | 7,634 | 0.04100 | [0.03729, 0.04507] |
| 13 | `qector_cuda` (GPU, weighted) | 1,000 | 151 | 0.00800 | [0.00406, 0.01571] |
| 13 | `qector_opencl` (GPU, unweighted) | 10,000 | 25,937 | 0.04100 | [0.03729, 0.04507] |
| 13 | `qector_opencl` (GPU, weighted) | 1,000 | 166 | 0.00800 | [0.00406, 0.01571] |
| 15 | PyMatching 2 | 50,000 | 4,847 | 0.00342 | [0.00295, 0.00397] |
| 15 | `qector_blossom` | 1,000 | 73 | 0.00200 | [0.00055, 0.00726] |
| 15 | `qector_unionfind` | 5,000 | 511 | 0.01440 | [0.01145, 0.01809] |
| 15 | `qector_cuda` (GPU, unweighted) | 10,000 | 4,398 | 0.03760 | [0.03405, 0.04151] |
| 15 | `qector_opencl` (GPU, unweighted) | 10,000 | 14,809 | 0.03760 | [0.03405, 0.04151] |

The complete 187-row table — every shot count, together with the 93 cells that
exceeded the per-cell decode budget and are therefore recorded as *not measured*
rather than estimated — is in `official_benchmark_results.md`.

Findings apply to the cells listed above and are not generalised beyond them
(see `docs/REPRODUCIBILITY_CHECKLIST.md`).

**Accuracy parity with PyMatching.** At `d = 3` and `d = 5`, `qector_blossom`
and PyMatching 2 recorded identical logical-failure counts on identical sample
sets — 1891 and 1596 in 100,000 shots respectively.

**Throughput.** PyMatching 2 leads at every distance measured here. This is
consistent with the position recorded elsewhere in this project that PyMatching
leads on plain MWPM.

**GPU accuracy depends on whether matching weights are supplied.**
`CUDABatchDecoder` and `OpenCLBatchDecoder` accept an optional `edge_weights`
argument. Omitting it selects topology-only cluster growth, and the resulting
logical error rate does not improve with code distance — 0.061 at `d = 5`,
0.043 at `d = 7`, 0.038 at `d = 15` — which is the signature of operation above
threshold. Supplying the DEM's `log((1-p)/p)` weights restores distance
scaling: 0.026 at `d = 5`, 0.010 at `d = 9`, 0.007 at `d = 13`. The weighted
path costs roughly 32× the throughput of the unweighted one.
`docs/BENCHMARK_COMPETITIVE.md` records the same effect for unweighted
Union-Find on CPU. See the quick-start above for the weighted construction.

**Backend agreement.** CUDA and OpenCL returned identical logical-failure counts
in every cell where both ran, consistent with the bit-identity property recorded
elsewhere in this project.

GPU throughput and GPU logical error rate should be cited together. The
unweighted configuration is the fastest column in the table and simultaneously
the least accurate; either figure alone misrepresents it.

Neither finding generalises beyond the cells above. Regenerate on quiesced
hardware, and state the noise model, before any number here is used in a claim.

### Published, citable evidence

Until that run lands, the reproducible accuracy and throughput evidence for this
project lives in the archived datasets, not in this file:

| Record | What it establishes | Methodology |
| --- | --- | --- |
| [10.5281/zenodo.21501377](https://doi.org/10.5281/zenodo.21501377) — Empirical benchmarks, v0.6.8 (CC-BY-4.0) | Exact LER and failure-count parity between `qector_blossom_weighted` and PyMatching 2.4.0 for `p ∈ [0.002, 0.008]`, `d ∈ {3,5,7,9}`; 100% syndrome faithfulness (`H·ê = s`) across odd `d ∈ [3,19]`; Union-Find 1.62×10⁵ shots/s at `d = 9`, 9.1× exact Blossom | Circuit-level, single pipeline. Ships 5 raw JSON datasets, 6 repro scripts, and a `manifest.json` carrying the wheel SHA256 and pinned dependency versions. Host: HP dual-core, 3.1 GB RAM, AntiX live USB, Python 3.13.5, pymatching 2.4.0, stim/sinter 1.16.0 |
| [10.5281/zenodo.21339300](https://doi.org/10.5281/zenodo.21339300) — Workbench benchmark master report, v0.6.6 (CC-BY-4.0) | 1,858 measurements over 105 runs; latency, throughput and peak memory for `d = 3–19` across 6 topologies | `p = 0.05`. Reports QECTOR decoders against each other — it is **not** a cross-library comparison |

Both are one release behind the working tree (v0.6.8 and v0.6.6 against 0.7.0);
read them as evidence about those versions.

Benchmark results are hardware, driver, compiler, and workload dependent.
Regenerate before quoting performance numbers, and state the noise model —
code-capacity and circuit-level LERs are different quantities.

---

## Reproduce benchmarks

```bash
# Every decoder family, one circuit-level pipeline, LER + Wilson CI + latency
# + syndrome faithfulness. Writes a JSON stamped with its own environment.
python scripts/full_decoder_benchmark.py

# The publication artifact: circuit-level throughout, provenance block embedded
python scripts/regenerate_benchmark_artifacts.py --yes

# MWPM / PyMatching comparison
python scripts/competitive_stim_ler.py --distances 3 5 7 9 11 13 15 --shots 40000

# Belief-matching comparison
python scripts/competitive_belief_matching.py --distances 3 5 7 --shots 3000 --no-ref

# GPU correctness
python scripts/gpu_extensive_test.py --distances 3 5 7 9 11 13 --batches 1 64 1024 4096 16384 65536 --error-rate 0.05

# Native memory profile
python scripts/native_memory_profile.py --distances 5 9 13 --batch 16384
```

These write into `benchmark_results/`, which is `.gitignore`d — the output stays
on the machine that produced it and is never committed. If you intend to publish
a number, publish the artifact alongside it.

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
| MWPM latency | PyMatching remains faster on standard surface-code MWPM. At `d = 11`, `p = 0.005`: 82 µs/shot for PyMatching against 1,559 µs/shot for exact `BlossomDecoder` (same run as the GPU figures above) |
| Belief-matching | Accuracy/research mode — can improve LER but much slower |
| GPU accuracy | Unweighted GPU kernels cost roughly 3× the logical error rate; pass `edge_weights` or accept that |
| GPU performance | Speedup is not universal, and the weighted kernel is currently slower than the weighted CPU path |
| Benchmark tables | The pre-v0.7.0 comparison tables are withdrawn (see above). Cite the archived datasets or regenerate |
| OpenCL | Depends on build configuration; confirm locally |
| SparseBlossom | Near-optimal, not exact MWPM — use `BlossomDecoder` for exact |
| UnionFind | Fast approximate path; not universal for arbitrary graphs |
| REST/gRPC/MCP | Not hardened as public SaaS without separate security review |

---

## Licensing

QECTOR Decoder v3 is **source-available** under the **PolyForm Noncommercial License 1.0.0** (see `LICENSE`). Personal, academic, educational, and non-commercial research use is allowed. Company use, funded institutional work, SaaS, hosted API deployment, OEM integration, redistribution, paid consulting, or commercial benchmarking requires a commercial license.

- **Pricing & tiers**: [https://www.qector.store/pricing](https://www.qector.store/pricing)
- **Direct purchase**: [Buy via Stripe](https://buy.stripe.com/6oU00l77Xc8ifsegEqeUU07)
- **Contact**: [admin@qector.store](mailto:admin@qector.store)

### DOI references

- Licensing terms & user manual: [10.5281/zenodo.21363016](https://doi.org/10.5281/zenodo.21363016)
- Performance benchmarks (v0.6.6): [10.5281/zenodo.21339300](https://doi.org/10.5281/zenodo.21339300)
- Architecture whitepaper: [10.5281/zenodo.21320543](https://doi.org/10.5281/zenodo.21320543)
- Empirical edge-hardware benchmarks (v0.6.8): [10.5281/zenodo.21501377](https://doi.org/10.5281/zenodo.21501377)
- Workbench GUI v3.5.0: [10.5281/zenodo.21360433](https://doi.org/10.5281/zenodo.21360433)
- Provenance archive (restricted): [10.5281/zenodo.20825980](https://doi.org/10.5281/zenodo.20825980)

```bibtex
@software{lessard2026qector,
  author  = {Guillaume Lessard},
  title   = {{QECTOR Decoder v3}: Rust/Python Quantum Error Correction Decoding Platform},
  year    = {2026},
  version = {0.7.0},
  url     = {https://www.qector.store},
  note    = {Source-available under PolyForm Noncommercial 1.0.0. Commercial license required for commercial use.}
}
```

---

**Copyright © 2026 Guillaume Lessard / iD01t Productions. All rights reserved.**

[https://www.qector.store](https://www.qector.store) · [admin@qector.store](mailto:admin@qector.store)
