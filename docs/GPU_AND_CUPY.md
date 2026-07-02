# GPU & CuPy Acceleration

**QECTOR Decoder v3 — v0.5.7**

This document describes the **hybrid GPU acceleration strategy** in QECTOR
Decoder v3: a Rust native-CUDA batch hot path plus an optional CuPy backend for
the Python-side belief-propagation, orchestration and extensibility layers, with
an automatic NumPy fallback so the same code runs unchanged on a CPU-only
machine.

> **Honesty note.** Everything below describes *software* behaviour and
> *simulation* (synthetic syndromes / code-capacity noise) measured on a single
> developer GPU (NVIDIA GeForce GTX 1660 Ti, 6 GB, CuPy 14.x, CUDA 12.x). The
> illustrative timings are wall-clock `perf_counter` numbers from the bundled
> examples on that machine; they are **not** hardware-QEC latencies and make no
> real-time / fault-tolerance claim. The validity bar quoted throughout is the
> GF(2) syndrome equation `H·c == s (mod 2)`, which is checked, not asserted.

---

## 1. Two GPU paths, one decision point

QECTOR has **two independent GPU paths**, and they are suited to different workloads:

| Path | What it is | Where it lives | Best at |
|------|------------|----------------|---------|
| **Rust native CUDA** | The `CUDABatchDecoder` Union-Find batch kernel, a per-thread port of the CPU `uf_core` (bit-identical to CPU). | Compiled into the crate at build time (`maturin build --features cuda`). | Very large **graphlike** (matching) batches — surface / toric / repetition — where one kernel launch amortises over tens of thousands of shots. |
| **CuPy runtime** | The Python GPU backend (`gpu_backend`) and the GPU-resident batched belief propagation (`bp_cupy`) that feeds BP-OSD. | Pure Python, imported lazily; pip-installable extra. | Batched **BP-OSD on non-graphlike qLDPC** codes (bivariate-bicycle, hypergraph-product), and any vectorised orchestration math (logical-flip parity, validity checks, streaming telemetry). |

These compose rather than compete. `routing.AutoRouter` (see
[the routing example](../examples/example_auto_routing.py)) picks the *decoder
family* — matching vs BP-OSD — and `backend.AutoDecoder` picks the *execution
backend* for a graphlike batch (single-thread / Rayon / CUDA / OpenCL). The CuPy
path is what accelerates the BP-OSD branch that matching decoders cannot serve.

The single place every module asks "is a GPU usable here, and which array
library should I compute with?" is **`qector_decoder_v3.gpu_backend`**.

---

## 2. Install

CuPy is **optional**. The core decoders, the Rust CUDA path, and the full NumPy
fallback all work without it.

```bash
# Core library (CPU + Rust CUDA path if the wheel was built with --features cuda)
pip install qector-decoder-v3

# Add the CuPy runtime backend for batched BP-OSD on a CUDA 12.x toolkit
pip install "qector-decoder-v3[cupy-cuda12x]"

# …or defer to a pre-installed / source CuPy
pip install "qector-decoder-v3[cupy]"

# Everything (stim ecosystem + bench tools + CuPy)
pip install "qector-decoder-v3[all]"
```

Notes:

- `cupy-cuda12x` pulls the prebuilt CuPy wheel matching a CUDA 12.x runtime
  (the tested cupy 13/14 line). Use the generic `cupy` extra if you build CuPy
  from source or already have it installed.
- The Rust native-CUDA path (`CUDABatchDecoder`) is a **build-time** feature
  (`maturin build --features cuda`), independent of the CuPy extra. A wheel built
  without it still runs everything on CPU/CuPy. See `INSTALL.md`.
- If CuPy is absent, or installed but no usable CUDA device is present, **nothing
  raises** — every GPU path degrades to NumPy on the host.

---

## 3. Runtime detection

All capability questions are answered by `gpu_backend`, and they are split so you
can reason about the two paths separately:

```python
from qector_decoder_v3 import gpu_backend as gb

gb.has_cupy()       # True if the `cupy` module imports
gb.has_cuda_rust()  # True if the Rust CUDABatchDecoder reports a usable device
                    #   (the build-time CUDA path)
gb.gpu_available()  # True iff CuPy imports AND a CUDA device is usable
                    #   (probed once: device enumerated + a tiny alloc succeeds)

gb.get_backend().summary()
# {'cupy_available': True, 'cuda_rust_available': True, 'gpu_available': True,
#  'device_name': 'NVIDIA GeForce GTX 1660 Ti', 'total_mem_bytes': 6442123264,
#  'prefer_gpu': True, 'active_module': 'cupy'}
```

Key distinctions:

- **`has_cupy()`** is a pure import check — CuPy may be installed but the driver
  broken; that is still `True`.
- **`gpu_available()`** is the authoritative "can I actually compute on the GPU
  right now?" probe. It enumerates device 0 and performs a tiny test allocation,
  so a present-but-broken driver reports `False` instead of crashing later inside
  a hot loop. It is cached (hardware does not change mid-process).
- **`has_cuda_rust()`** is orthogonal: it reports the *Rust* build-time CUDA path,
  not the CuPy runtime path. You can have one without the other.

### Policy vs. capability

Detection reports **hardware**; a separate flag controls **policy**:

```python
gb.set_prefer_gpu(False)   # force every policy-driven path onto NumPy/host
gb.get_prefer_gpu()        # current global preference (default True)
```

`set_prefer_gpu(False)` is the switch for benchmarking and reproducibility: it
makes `get_array_module` / `xp` / `to_device` and the batched BP path all use
NumPy even on a GPU box. `gpu_available()` is unaffected — it still reports the
hardware truthfully.

### Choosing the array module

```python
xp = gb.get_array_module(prefer_gpu=True)   # cupy if usable+preferred, else numpy
a  = gb.to_device([1, 0, 1, 1])             # host -> device (counted), or no-op on CPU
gb.is_on_gpu(a)                             # True iff a is a CuPy device array
host = gb.to_host(a)                        # device -> host (counted)
```

`get_array_module` is policy-aware: if **any** argument already lives on the GPU
it returns `cupy` regardless of preference (you cannot operate on device arrays
with NumPy), otherwise it honours the `prefer_gpu` request, the global
preference, and `gpu_available()`.

---

## 4. When CuPy helps — and when it does not

This section is deliberately candid. CuPy is a throughput tool, not a latency tool.

### Where it helps: batched BP-OSD on qLDPC

`bp_cupy.BatchedBpDecoder` runs the belief-propagation stage for a whole stack of
syndromes `[batch, n_checks]` in lock-step, carrying the batch as a trailing
array axis so every variable→check and check→variable update is one vectorised
operation over `[n_edges, batch]` message arrays. The Tanner incidence, prior
LLRs and message buffers are built **once** on the device; per call only the
syndrome stack moves host→device and only the corrections (plus optional
posterior LLRs / convergence mask) move device→host — the minimal PCIe traffic.

`bposd.BpOsdDecoder.batch_decode` wraps this with an **OSD-0 fast path**: shots
that BP alone already explains (`H·c == s`) are returned directly; only the
residual non-converged shots take the exact GF(2) ordered-statistics solve,
seeded with that shot's GPU BP reliabilities. Every returned row satisfies its
syndrome.

This is the regime matching decoders **cannot** serve — bivariate-bicycle,
hypergraph-product and other non-graphlike codes have qubits of degree > 2, so a
matching decoder cannot in general satisfy `H·c == s`. `routing` therefore always
sends these to BP-OSD.

Illustrative run (`examples/example_cupy_bp.py`, X sector of the `[[72,12,6]]`
bivariate-bicycle code, 256 shots at `p = 0.03`, GTX 1660 Ti):

```
1. batched_bp_decode (BP only, GPU-resident when available):
   converged (BP alone explains syndrome): 214/256
   H·c == s on converged shots: True
   GPU telemetry (this call): {'h2d': 1, 'd2h': 2, 'gpu_calls': 1, 'fallbacks': 0}

2. BpOsdDecoder.batch_decode (BP-OSD; OSD-0 fast path):
   H·c == s on ALL 256 shots: True
   shots needing exact OSD post-process: 42
   GPU telemetry (this call): {'h2d': 1, 'd2h': 3, 'gpu_calls': 1, 'fallbacks': 0}
```

The whole 256-shot BP stage is **one** host→device transfer and **one** GPU
compute pass (`gpu_calls: 1`); only the 42 unconverged shots fall through to the
CPU OSD solve. The larger the batch, the better this amortises.

### Where it does not help: single-shot latency

For a **single** syndrome, the GPU is not the appropriate tool. Kernel-launch and
host↔device transfer overhead (microseconds each) dwarf the actual arithmetic for
one small problem, so a single-shot decode is typically *slower* on the GPU than
on the CPU. Accordingly:

- `BpOsdDecoder.decode` (the single-shot path) **never** touches the GPU — it runs
  the CPU `_bp_core` BP directly. Only `batch_decode` consults the GPU policy.
- The Rust `CUDABatchDecoder` is only recommended by `routing` once the batch
  crosses the GPU threshold; below it, `FastUnionFindDecoder` (lowest per-shot
  overhead) or Rayon `BatchDecoder` win.
- The streaming layer decodes one round at a time on the CPU inner decoder; its
  GPU use is confined to the vectorised logical-flip / validity math, not the
  per-round decode.

General guidance: **GPU for throughput on large batches; CPU for latency on single
shots.** The library encodes this in its routing so you usually do not need to
choose manually.

---

## 5. Telemetry

`gpu_backend` keeps cross-module counters in the module-level `TELEMETRY` dict,
mutated in place (so external references stay valid) and resettable:

```python
from qector_decoder_v3 import gpu_backend as gb

gb.reset_telemetry()
# ... run a batched decode ...
gb.TELEMETRY
# {'h2d': 1, 'd2h': 3, 'gpu_calls': 1, 'fallbacks': 0}
```

| Counter | Meaning |
|---------|---------|
| `h2d` | host → device array transfers (`to_device`, batched-BP syndrome upload) |
| `d2h` | device → host transfers (`to_host`, corrections / LLRs / convergence download) |
| `gpu_calls` | GPU compute operations recorded by callers (`note_gpu_call`) |
| `fallbacks` | times a requested GPU action degraded to the CPU/host path (`note_fallback`) |

These are **real measured counts**, not estimates. They make the host/device
traffic of any decode auditable: a well-amortised batched BP-OSD shows a tiny,
fixed number of transfers regardless of batch size, and `fallbacks` stays at 0
when the GPU path completes (it increments when a GPU error forces the CPU path).

The streaming layer additionally records **real per-window wall times** via
`time.perf_counter` around the inner-decoder call only, exposed on
`StreamingTelemetry` (`decode_seconds`, `per_window_seconds`,
`mean_window_seconds`) alongside the `gpu` counter deltas and a one-glance
`backend` capability snapshot.

---

## 6. Automatic NumPy fallback (the correctness guarantee)

The contract that makes all of this safe: **with CuPy absent or the GPU disabled,
every path produces the same valid corrections on the host.** The batched-BP
NumPy path is bit-identical to the single-shot `_bp_core` BP (because
`np.add.at` accumulates per column in the same edge order), and the OSD
post-process is identical GF(2) linear algebra either way.

The bundled example demonstrates this directly by forcing the fallback
(`examples/example_cupy_bp.py`, step 3):

```
3. Forced NumPy fallback (set_prefer_gpu(False)):
   H·c == s on ALL 256 shots: True
   GPU telemetry (should show no GPU calls): {'h2d': 0, 'd2h': 0, 'gpu_calls': 0, 'fallbacks': 0}
```

Same validity, zero GPU traffic. Code written against `gpu_backend` runs
unchanged on a CPU-only box.

---

## 7. Worked examples

Three runnable scripts under `examples/` (run with `PYTHONPATH=python`):

- **`example_cupy_bp.py`** — batched GPU BP-OSD on the `[[72,12,6]]`
  bivariate-bicycle qLDPC code, with the NumPy fallback and telemetry printout.
- **`example_auto_routing.py`** — `recommend_decoder` + `AutoRouter` across
  surface / repetition / qLDPC, including the structural guard that forces BP-OSD
  on a non-graphlike code even when it is mislabelled `"surface"`.
- **`example_streaming_session.py`** — `StreamingSession` and
  `sliding_window_decode` over a multi-round synthetic stream, with real
  per-window timing and window-invariance for a stateless decoder.

---

## 8. Summary

- **Hybrid by design:** Rust native CUDA for big graphlike batches; CuPy for
  batched BP-OSD on qLDPC and Python-side orchestration math.
- **One detection surface:** `has_cupy` / `has_cuda_rust` / `gpu_available` +
  `get_backend().summary()`; policy via `set_prefer_gpu`.
- **GPU for throughput, CPU for latency:** single-shot decodes stay on the CPU by
  design; batches amortise onto the GPU.
- **Always falls back:** no CuPy / no device ⇒ NumPy host path, same validity,
  zero GPU traffic. Every committed correction satisfies `H·c == s (mod 2)`.
