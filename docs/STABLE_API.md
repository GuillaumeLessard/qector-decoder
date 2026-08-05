# QECTOR v1.0.0 — Stable Public API

> **Status of this document:** Scaffold for the v1.0.0 release.
> **Cross-references:** `docs/API_STABILITY.md` (the long-form stability policy), `docs/API_SURFACES.md` (the per-symbol status table).
> **Audience:** Users picking a QECTOR version, and the v1.0.0 release reviewer who needs to confirm the stability promises before tagging.

This file is the **positive** v1.0.0 commitment: what is stable, what is provisional, and what is internal — for *this release* specifically. It is a focused, frozen-at-1.0.0 view that supplements the rolling `API_STABILITY.md`.

---

## 1. What "stable" means at v1.0.0

A symbol marked **Stable** at v1.0.0 is committed to the following guarantees, valid for the entire `1.x` line:

- **No removal** in any 1.x release. Deprecations are allowed (with a 2-release sunset at minimum), but the symbol will keep working.
- **No signature-breaking change** in any 1.x release. New keyword arguments with defaults are allowed. Renames are not.
- **No behaviour change in the documented contract.** The contract for each stable symbol is listed in §3.
- **Bug fixes that change observable output** are allowed (because the contract is the contract, not the output), but they are documented in the changelog with a "Behaviour" note and a one-line justification.

A symbol marked **Provisional** at v1.0.0 is supported but may change in a minor release (1.1, 1.2, …) without the 2-release deprecation sunset. It is on the path to Stable but has not earned that yet.

A symbol marked **Internal** is implementation detail. It may be removed, renamed, or have its signature change in any release. Documenting it here is a courtesy, not a promise.

---

## 2. The three stability tiers in v1.0.0

| Tier | SemVer promise within 1.x | Examples |
|---|---|---|
| **Stable** | None-removal, no signature breaks, documented contract | Core decoder classes, code generators, license-key API, public CLI subcommands |
| **Provisional** | May change signature/behaviour in a minor release, with a changelog note | BP-OSD tuning kwargs, GPU batch constructors, MCP server tool names |
| **Internal** | No promise; may change without notice | Rust module layout, private `_` helpers, harness internals |

---

## 3. Stable public API at v1.0.0

### 3.1 Top-level import

```python
import qector_decoder_v3 as q
```

- `q.__version__` is a string of the form `"1.x.y"`. **Stable.** (Bump-format only; never removed.)
- `q.cuda_is_available()` → `bool`. **Stable.** Contract: returns `True` iff a CUDA driver and a working NVRTC-compiled kernel are both reachable from the loaded wheel. `False` on machines without CUDA; **never raises** on a healthy install.
- `q.opencl_is_available()` → `bool`. **Stable.** Same contract as CUDA, for OpenCL.

### 3.2 Core decoders — Stable

All decoders in this section share one contract:

```text
decode(syndrome) -> np.ndarray[uint8, shape=(n_qubits,)]
H @ correction == syndrome (mod 2)   # syndrome-faithfulness, always
```

| Symbol | Tier | Contract additions |
|---|---|---|
| `q.UnionFindDecoder(check_to_qubits, n_qubits=None)` | **Stable** | Optional `edge_weights=...` (the DEM's `log((1-p)/p)`) — when present, weighted growth; when absent, unweighted (bit-identical to integer growth, no-weights). |
| `q.FastUnionFindDecoder(check_to_qubits, n_qubits=None)` | **Stable** | Same contract as UnionFind. Lower-overhead; same accuracy by construction. |
| `q.BlossomDecoder(check_to_qubits, n_qubits=None)` | **Stable** | Exact weighted MWPM. Reference for correctness. Not a latency leader; prefer `SparseBlossomDecoder` for throughput. |
| `q.SparseBlossomDecoder(check_to_qubits, n_qubits=None)` | **Stable** | Region-growing sparse MWPM. Always syndrome-faithful; not always exact-MWPM-optimal (≥99% on small codes). Bit-identical to `BlossomDecoder` on the tested `ring_code` and `repetition_code` syndromes. |
| `q.NativeAutoDecoder(check_to_qubits, n_qubits=None)` | **Stable** | Native CPU/GPU routing. License tier is enforced; on Community tier, GPU paths are skipped. |

#### 3.2.1 Provisional decoder surface

| Symbol | Tier | Why provisional |
|---|---|---|
| `q.CPUBatchDecoder` / `q.BatchDecoder` | **Provisional** | Constructor signature stable; `batch_decode()` input shape contract is stable for `(N, n_checks)` uint8 batches. Output shape and dtype are stable. Performance claims are workload-sensitive (see `docs/API_STABILITY.md`). |
| `q.AutoDecoder` | **Provisional** | The 7-tier fallback is a behaviour, not a contract — exact tier ordering may shift in a minor release. The "always returns a valid correction or raises" guarantee is Stable. |
| `q.StreamingDecoder` / `q.SlidingWindowDecoder` | **Provisional** | Constructor and `commit(window=)` API are stable; the internal buffer-growth strategy is not. |
| `q.BPOSDDecoder` / `q.BpOsdDecoder` | **Provisional** | `decode(syndrome)` and `batch_decode(syndromes)` are stable. Tuning kwargs (`bp_method`, `osd_order`, `damping`, `osd_lambda`) are stable in name; their default values and effect-window may shift in a minor release (e.g. switching the BP default from `min_sum` to `sum_product` would be a `1.x` change with a changelog note, not a 2.0.0 break). |

### 3.3 Code generators — Stable

| Symbol | Tier | Contract |
|---|---|---|
| `q.generate_repetition_code_checks(d, ...)` | **Stable** | Returns a 2D list-of-lists suitable for the `check_to_qubits` argument of every Stable decoder. |
| `q.generate_ring_code_checks(n, ...)` | **Stable** | Same. |
| `q.generate_surface_code_checks(d, ...)` | **Stable** | Returns a graphlike (weight-≤2) check matrix; explicit on this property. |
| `q.generate_toric_code_checks(d, ...)` | **Provisional** | Returns a periodic-boundary check matrix. Output is stable in shape; the per-distance boundary handling has been a source of bugs in earlier releases and is on a 1.x watchlist. |

### 3.4 License-key API — Stable

| Symbol | Tier | Contract |
|---|---|---|
| `q.set_license_key(key: str)` | **Stable** | Sets the active license from a v1 or v2 Ed25519-signed token. **Raises `ValueError`** on invalid (never silently accepts) — this is a hard contract at v1.0.0. Idempotent; safe to call on every startup. |
| `q.get_license_info()` | **Stable** | Returns a dict with at least the keys `sub`, `tier`, `exp`, `max_distance`, `gpu_enabled`, `gnn_enabled`, `is_expired`. Extra keys may be added in 1.x but documented ones will not be removed. |
| `QECTOR_LICENSE_KEY` / `QECTOR_LICENSE_FILE` env vars | **Stable** | Resolution order is `QECTOR_LICENSE_KEY` → `QECTOR_LICENSE_FILE` → `~/.qector/license.key`. A set-but-unreadable `QECTOR_LICENSE_FILE` is **invalid** (raises), not a silent downgrade. |
| `QECTOR_SILENT=1` | **Stable** | Suppresses the startup banner. |

### 3.5 CLI — Stable

| Command | Tier | Contract |
|---|---|---|
| `qector decode <input> [options]` | **Stable** | Reads a `.npy` check matrix (or a `.dem` with `--dem`), decodes a `.npy` syndrome batch, writes a `.npy` correction. All 7 `--decoder` choices return a syndrome-faithful correction. Exit code `0` on success, non-zero on error. |
| `qector bench <input> [options]` | **Provisional** | The CLI surface is stable; the *reported numbers* are workload-sensitive and should not be cited without the artifact. |
| `qector serve [options]` | **Provisional** | Boots the local REST service. Defaults to `127.0.0.1`. **Not a hosted service contract** — see `docs/SECURITY_DEPLOYMENT.md` before exposing externally. |
| `qector-doctor` | **Stable** | 14-PASS / 1-WARN / 0-FAIL diagnostic. Exit code is `0` iff all backend-availability checks pass. |

### 3.6 Telemetry — Stable

| Symbol | Tier | Contract |
|---|---|---|
| `q.record_shots(n: int)` | **Stable** | Increments the metered-billing counter. Idempotent across process restarts (the counter is in-process only at v1.0.0; persistence is a 1.x roadmap item, not a v1.0.0 promise). |
| `q.get_accumulated_shots() -> int` | **Stable** | Returns the current counter. Process-local. |

### 3.7 Result types — Stable

| Symbol | Tier | Contract |
|---|---|---|
| `q.DecodeResult` | **Stable** | A structured result type. Fields: `correction: np.ndarray[uint8]`, `syndrome: np.ndarray[uint8]`, `weight: int`, `decoding_time_s: float`, `backend: str`. `to_json()` and `explain()` are Stable. |

---

## 4. Provisional public API at v1.0.0

These are real, supported, and tested — but their exact surface may shift in a 1.x release. Treat them as "supported, but pin a version range, not an equality, when depending on them."

### 4.1 GPU batch decoders

| Symbol | Tier | Why provisional |
|---|---|---|
| `q.CUDABatchDecoder(check_to_qubits, n_qubits=None, edge_weights=None, precision="f32")` | **Provisional** | Constructor kwargs `edge_weights` and `precision` (one of `"f32"` / `"f64"`) are stable in name; the f64 path is a v0.7.x → v1.0.0 addition. `batch_decode()` shape contract is stable. **Bit-identity claim** (vs the CPU `UnionFindDecoder` in the no-weights case) is verified on tested graphlike codes — workload-sensitive. |
| `q.OpenCLBatchDecoder(...)` | **Provisional** | Same surface as `CUDABatchDecoder`. Driver/runtime dependent. |
| `q.CUDABpOsdDecoder` | **Provisional** | Construction requires a working CUDA driver and a compiled kernel; the kernel now compiles (per v0.7.0 fix). Single-shot `decode()` is **not** in v1.0.0 — use `batch_decode(s.reshape(1, -1))`. |

### 4.2 BP-OSD kwargs

| Symbol | Tier | Why provisional |
|---|---|---|
| `BpOsdDecoder(damping=0.0, osd_lambda=None, ...)` | **Provisional** | Both kwargs are stable in name. The CS-OSD(λ, w) sweep behaviour, the LLR damping formula `m ← (1-d)·m_new + d·m_old`, and the default `osd_lambda=24` may be tuned in a 1.x release. |

### 4.3 Sinter / qiskit-qec / pymatching compatibility

| Surface | Tier | Note |
|---|---|---|
| `sinter.collect(..., custom_decoders=[qector_blossom, ...])` | **Stable** | The 5 entry-point names (`qector_blossom`, `qector_belief`, `qector_unionfind`, `qector_bposd`, `qector_unionfind_unweighted`) are Stable — entry-point names are an interface contract. |
| `qiskit.qec` plugin | **Stable** | Plugin entry-point name is Stable. |
| `q.pymatching.Matching` | **Stable** | The submodule spelling (the attribute form already worked pre-v0.7.0). |

### 4.4 Experimental research surfaces

These are documented in the API but explicitly **not** in the v1.0.0 Stable tier. They will not be removed in 1.x, but they may gain features, change defaults, or have their exact behaviour tuned. Cross-reference `docs/API_STABILITY.md` for the longer list.

| Symbol | Tier | Note |
|---|---|---|
| `q.AmbiguityClusterDecoder` | **Provisional** | On the path to Stable. Tested; the cluster-DFS implementation is correct. Defaults may be tuned. |
| `q.TwoStageDecoder` | **Provisional** | Requires `check_types`. Behaviour on hyperedge codes is correct; default stage-pair selection may shift. |
| `q.ColourCodeDecoder` | **Provisional** | BP-OSD on the **undecomposed** hypergraph DEM. **`method` default needs to be reconciled with the v0.7.x changelog before v1.0.0 ships** — see `docs/unreleased_audit.md`. |
| `q.GNNBeliefMatcher` | **Provisional** | GNN-guided MWPM. Requires a trained checkpoint; training is out of scope for v1.0.0. |
| `q.NeuralPredecoder` / `q.HybridCascadeDecoder` | **Provisional** | Pre-decoder; the Cascade fix in v0.7.0 made the weights thread through; behaviour may be tuned. |
| `q.Workbench` | **Provisional** | Local-validation workstation. Not a hosted service. |

### 4.5 Network-facing surfaces

These are explicitly **Provisional** in 1.0.0 and will not be promoted to **Stable** without a deployment-review entry in `docs/API_STABILITY.md`. They are useful for demos and partner evaluation; they require a separate hardening pass before customer-facing use.

- REST service (`qector serve`, default `127.0.0.1`)
- gRPC service (optional, `--features grpc`)
- MCP server (stdio JSON-RPC, 9 real tools as of v0.7.0)
- Prometheus metrics exporter (default `127.0.0.1:9090`)

---

## 5. Internal API at v1.0.0 (no promise)

The following are implementation detail. They are documented for code-readers and for the maintainer's own bookkeeping. **No SemVer promise.**

- Rust module layout under `src/` — the `pub` items exposed to PyO3 are the contract; everything else is internal.
- The `_bp_core` private module.
- The `_native_module` symbol in `__init__.py`.
- The `_guard("ClassName")` callable stubs (they exist to make `import` always succeed; calling a guarded-stub raises `RuntimeError`).
- The `decoder_changes["0.x.y"]` history dict in `__init__.py`.
- The `benchmarks_session/`, `audit/`, `vs.py`, and other ad-hoc scratch directories in the working tree.
- The `RUST_SRC_B64_1..12` GitHub Actions secrets (build-time mechanism; not an API).

---

## 6. What will require a v2.0.0

The following are explicitly **not** promised in 1.x. Touching any of them is a breaking change that ships in 2.0.0:

- The `H · correction == syndrome (mod 2)` contract. (This is the foundational invariant; "breaking" the contract means the project is no longer a QEC decoder.)
- The decoder-class names listed in §3.2 and §3.2.1. Renames require a deprecation cycle of at least two 1.x releases.
- The `set_license_key` raising-on-invalid contract. (This was specifically hardened because silent-accept was the worst version of a hard-to-diagnose failure mode; relaxing it would invite the bug back.)
- The Stable-import names in §3.1.

Things that **can** change in a 1.x without a 2.0.0:

- Provisional-API signatures, with a changelog note.
- Internal Rust module layout.
- Performance numbers (must always be backed by a new artifact).
- New optional kwargs with defaults.
- New decoder families.
- A GPU backend dropping support for an EOL CUDA toolkit (this would be a 1.x with a deprecation notice, not a 2.0.0).

---

## 7. Promotion path from Provisional to Stable

To promote a symbol from §4 to §3 in a future 1.x release:

1. Add a `Promotion:` entry in `docs/API_STABILITY.md` with the date, the symbol, the surface reviewed, and the specific review that justified the promotion.
2. Pass the same bar as a Stable symbol on test coverage (a property test, a regression test, and an example) — per `docs/API_STABILITY.md` "Required before promoting".
3. Move the symbol from §4 to §3 in `STABLE_API.md`.
4. Note the promotion in `CHANGELOG.md` under the relevant 1.x release.

---

## 8. The v1.0.0 readiness checklist (sign-off)

Before tagging v1.0.0, confirm:

- [x] The `ColourCodeDecoder` default-method question is resolved (see `docs/unreleased_audit.md`). — default stays `method="bposd"` (accuracy-first); `cluster_bposd` remains opt-in, matching the code docstring.
- [x] The `CUDABpOsdDecoder.decode` single-shot convenience is either implemented or cut from the changelog. — implemented (`src/cuda_python.rs:194-211`).
- [x] All `0.7.x` strings in the repo are bumped to `1.0.0` (per the checklist in `docs/unreleased_audit.md`). — `__fallback_version__ = "1.0.0"`, Cargo/pyproject/CITATION/codemeta all 1.0.0, `rust_core.sha256` refreshed to the SEC-02 repack.
- [ ] The two pre-publish fixes (bench_community.json hygiene, the 9-warning spam) are applied and the report regenerated.
- [ ] The verification harness passes 2×2 with 0 failures and 0 flaky.
- [ ] `cargo test --no-default-features` and `cargo test --features full` both green.
- [ ] The v1.0.0 wheel is built via `release-build.yml` (Sigstore Trusted Publishing) and the attestation bundle is preserved.
- [ ] This document is reviewed and merged.

If any item is open, the v1.0.0 tag should be delayed, not pushed.
