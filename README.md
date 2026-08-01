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


PyMatching-compatible MWPM validation · Belief-matching accuracy mode · BP-OSD for LDPC/qLDPC · CPU/GPU batch decoding · 7-tier self-debugging fallback engine · Ed25519 cryptographic license verification · Reproducible benchmark harness

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

What it costs: the **unweighted** GPU kernels trade logical accuracy for
throughput. They decode faster than the weighted CPU core, at a materially higher
logical error rate that throughput does not buy back. Pass `edge_weights` when
accuracy matters.

The weighted GPU kernel is the accuracy option: weighting restores distance
scaling, at a higher per-shot cost. Both configurations are exposed for
`qector_cuda` and `qector_opencl` so you can measure the trade-off on your own
hardware and noise model.

**No benchmark figures are published for this release.** Decoder performance is
hardware-, code- and noise-dependent, and any number quoted here would not
describe your setup. Run the harness yourself — the JSON it writes carries its own
environment and parameter block, so results are traceable to the machine that
produced them. See `docs/GPU_AND_CUPY.md`.

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

> **Benchmark figures are not published for this release.**
> Decoder throughput and logical error rate depend on your hardware, code family,
> distance and noise model, so any table printed here would describe a machine that
> is not yours. The benchmark harness ships with the package and writes JSON carrying
> its own environment and parameter block — run it on your target hardware and compare
> decoders under the conditions you actually care about.


The complete 187-row table — every shot count, together with the 93 cells that
exceeded the per-cell decode budget and are therefore recorded as *not measured*
rather than estimated — is in `official_benchmark_results.md`.

Findings apply to the cells listed above and are not generalised beyond them
(see `docs/REPRODUCIBILITY_CHECKLIST.md`).

**Accuracy parity with PyMatching.** At the distances where both decoders were
measured on identical sample sets, `qector_blossom` and PyMatching 2 recorded
identical logical-failure counts. The counts live in the artifact; they are not
restated here because this release publishes no performance figures.

**Throughput.** PyMatching 2 led at every distance measured in that run. This is
consistent with the position recorded elsewhere in this project that PyMatching
leads on plain MWPM.

**GPU accuracy depends on whether matching weights are supplied.**
`CUDABatchDecoder` and `OpenCLBatchDecoder` accept an optional `edge_weights`
argument. Omitting it selects topology-only cluster growth, whose logical error
rate does not improve with code distance — the signature of operation above
threshold. Supplying the DEM's `log((1-p)/p)` weights restores distance scaling.
The weighted path costs more per shot than the unweighted one.
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
| [10.5281/zenodo.21501377](https://doi.org/10.5281/zenodo.21501377) — Empirical benchmarks, v0.6.8 (CC-BY-4.0) | Archived empirical benchmark dataset for v0.6.8, including syndrome-faithfulness verification (`H·ê = s`) and matching parity against PyMatching | Circuit-level, single pipeline. Ships 5 raw JSON datasets, 6 repro scripts, and a `manifest.json` carrying the wheel SHA256 and pinned dependency versions. Host: HP dual-core, 3.1 GB RAM, AntiX live USB, Python 3.13.5, pymatching 2.4.0, stim/sinter 1.16.0 |
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

## MCP server (stdio)

The package ships an **MCP server** (JSON-RPC 2.0 over stdio) in every published
wheel — no extra feature flag or install is needed:

```bash
python -c "import qector_decoder_v3; qector_decoder_v3.run_mcp_server()"
```

A ready-made client configuration lives in `mcp.json` at the repository root
(it launches `python -c "import qector_decoder_v3; qector_decoder_v3.run_mcp_server()"`
with `QECTOR_SILENT=1`). Point your MCP client at that file — e.g. Claude Code
supports `mcp.add` with the `qector` server name. The server advertises 13
tools, all verified on the released wheel:

| Tool | Purpose |
|---|---|
| `decode_syndrome` | Decode a syndrome with any decoder family (Union-Find, Blossom, SparseBlossom, BP-OSD, Cascade, Hybrid, and more) |
| `batch_decode` | Batch-decode multiple syndromes in parallel |
| `decode_hyperedge` | Hyperedge / qLDPC decoding (bypasses graphlike Union-Find restrictions) |
| `decode_syndrome_blossom` / `batch_decode_blossom` | Exact Blossom (MWPM) single and batch |
| `decode_syndrome_cascade` | Hybrid cascading decoder (UF pre-filter escalating to Blossom) |
| `benchmark_decoder` | Run a performance benchmark for a decoder family |
| `run_ler_benchmark` | LER benchmark across code distances |
| `get_decoder_info` | Decoder configuration, version info, family listing |
| `get_backend_health` | Backend health status across the 7 fallback tiers |
| `clear_decoder_cache` | Clear the decoder factory cache |
| `get_server_env` | Effective QECTOR environment variables |
| `recommend_decoder` | Decoder recommendation by code topology and priority |

The stdio reader enforces a 10 MB content limit and validates syndrome lengths
and decoder types, returning JSON-RPC errors instead of crashing. For local and
controlled use; like REST/gRPC, it is not hardened for public SaaS exposure.

---

## Limits and boundaries

| Area | Boundary |
| --- | --- |
| MWPM latency | PyMatching remains faster than exact `BlossomDecoder` on standard surface-code MWPM. QECTOR's value is decoder breadth and qLDPC coverage, not beating PyMatching at its own workload |
| Belief-matching | Accuracy/research mode — can improve LER but much slower |
| GPU accuracy | Unweighted GPU kernels trade logical accuracy for throughput; pass `edge_weights` or accept that |
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
