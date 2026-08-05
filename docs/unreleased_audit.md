# [Unreleased] Section Audit — v0.7.1 → v1.0.0 Cut

**Date:** 2026-08-05
**Source tree:** `C:\Users\Clinque du Batiment\Downloads\qector-decoder-clone`
**Compiled wheel reported by verification harness:** `package_version = "1.0.0"` (per `results/results_enterprise.json`), but Python-side `__fallback_version__ = "0.7.1"` in `python/qector_decoder_v3/__init__.py:412`
**Doc: `CHANGELOG.md` `[Unreleased]` section, lines 8–64**

This file is the answer to *"which of the [Unreleased] items are actually in the current wheel vs. staged for the next cycle?"* — going line-by-line through the changelog and grepping the source.

---

## ✅ Items confirmed in the source (9 of 10)

| Changelog claim | File:line evidence | Status |
|---|---|---|
| Formal CS-OSD(λ, w) + LLR damping in `bposd.py` | `python/qector_decoder_v3/bposd.py:61-87` — `damping: float = 0.0` and `osd_lambda: Any = None` constructor kwargs, `_check_damping` import, threading to both `min_sum_bp` and `sum_product_bp` calls at lines 124, 136; the `osd_lambda`-capped sweep at line 277 | In source, threaded through to BP kernels |
| Verification-harness credibility fixes — `--run 0\|1` + sandbox | `run_full_verification.py:99-113` — `run_pytest(mode, run_index)` writes `junit_<mode>_run{0,1}.xml` and `pytest_<mode>_run{0,1}.log`; `sandbox_env_community()` at lines 36-47 remaps `HOME`/`USERPROFILE` to a fresh temp dir | In source |
| External LER gate tightened — seeded 12 000 shots, real `+1e-3` bound | `tests/test_12_benchmark_xcheck.py` (run on both tiers, passed in the verification harness); the `ler_rep_d3_p001_qector_blossom = 0.00392` vs `ler_rep_d3_p001_pymatching = 0.00317` measurement (delta 0.00075) is in `report/qector_benchmarks.csv` | In source, verified end-to-end |
| License-pinning test fixtures (`test_enforcement_matrix.py`, `test_license_rust_bridge.py` remap HOME/USERPROFILE) | `python/tests/test_enforcement_matrix.py:72` — `HOME/USERPROFILE are remapped to an empty temp dir so the pure-Python [fallback does not re-read the machine's key]` | In source |
| `CUDABatchDecoder(precision="f64")` — `uf_decode_batch_cuda_f64` kernel + `UfGraph::edge_lengths_f64()` | `src/cuda_batch.rs:18-92` — `pub enum CudaPrecision { F32, F64 }`; `:118-126` — `new_weighted_precision(...)` constructor; `:298-302` — `cuModuleGetFunction(uf_decode_batch_cuda_f64)`; `:345-381` — separate `edge_len_f64` upload; `:727-728` — public `precision()` getter | In source |
| Dual-stream weighted CUDA race fix — `sf` base offset per stream | `src/cuda_batch.rs:611` — `d_sf + (half * self.n_edges * self.precision.elem_bytes()) as u64` (the per-stream offset the fix describes) | In source |
| `CUDABatchDecoder` leaked `edge_len` device buffer on drop | `src/cuda_batch.rs:775-778` — Drop impl now frees both `edge_len` (F32) and `edge_len_f64` (F64) buffers | In source |
| `SparseBlossomDecoder` zero-allocation hot path (thread-local `SbScratch`) | `src/sparse_blossom.rs:330, 407-525, 1303, 1517, 1895` — `pub(crate) struct SbScratch`, `thread_local! { static TL_SB_SCRATCH: ... }`, `decode_core(... sc: &mut SbScratch)`, `solve_mwpm_blossom(sc: &mut SbScratch)`, tests reference `SbScratch::new()` | In source, internal-only |
| `CUDABpOsdDecoder.decode` single-shot convenience | `src/cuda_python.rs:194-211` — `#[pymethods] fn decode<'py>(&self, py, syndrome: PyReadonlyArray1<u8>) -> PyResult<Bound<'py, PyArray1<u8>>>`; docstring: "Single-shot decode: a 1-D syndrome of length `n_checks` decoded to a 1-D correction of length `n_qubits`. Implemented as a one-row `batch_decode` — the CUDA BP-OSD kernel is batch-oriented, so for latency-critical single shots prefer the CPU `BpOsdDecoder`." | In source, exposed to Python automatically via the `#[pyclass]` registration. **Note:** this item was initially misclassified as "not in source" by a grep that only matched `pub fn decode` in the inner Rust struct; the actual Python binding is in `cuda_python.rs` and uses the `#[pymethods]` attribute, which is invisible to a `pub fn` grep. |

## ⚠️ Partially in source — needs reconciliation before v1.0.0 (1 of 10)

### Hyperedge cluster expansion for colour codes

| What the changelog says | What the source actually does |
|---|---|
| "`ColourCodeDecoder(method="cluster_bposd")` (**new default**) runs a weighted union-find growth over the undecomposed hypergraph DEM with verified spanning-tree peeling, and hands only unresolved residual syndromes to BP-OSD. `method="bposd"` reproduces the previous behaviour bit-for-bit." | `python/qector_decoder_v3/colour_code.py:260` — `def __init__(self, dem, max_iter=30, osd_order=0, method: str = "bposd"):` — **the default is `bposd`, not `cluster_bposd`**. The `cluster_bposd` path exists (lines 16, 36, 269, 322-354) and is opt-in. |

There's a **three-way disagreement** between the changelog, the module docstring (line 16: "the default (`method="cluster_bposd"`) runs both"), and the constructor signature (line 260: default `"bposd"`).

**Recommendation:** pick one of the two before tagging v1.0.0:

1. **Change the source to match the changelog** (flip the default to `cluster_bposd`). This is a behaviour change for any user currently passing `method="bposd"` implicitly — they get a different decoder. Acceptable because v0.x is "anything can change" and v1.0.0 is the freeze point.
2. **Change the changelog and the docstring to match the source** (describe `cluster_bposd` as opt-in, not the default). Lower risk for the cut, but ships a more limited colour-code improvement than the changelog promises.

## ❌ Not in source (0 of 10)

This section is empty. The previous draft of this audit reported `CUDABpOsdDecoder.decode` as missing; that was a false negative from a `pub fn` grep that ignored `#[pymethods]` items. The method is in `src/cuda_python.rs:194-211`.

---

## Versioning state — resolved (was a `__fallback_version__` mismatch)

The version-string mismatch flagged in the original audit is **fixed**: `python/qector_decoder_v3/__init__.py:412` now reads `__fallback_version__ = "1.0.0"`, matching the wheel, `Cargo.toml`, `pyproject.toml`, `CITATION.cff`, and `codemeta.json`.

| Source | What it says | Where |
|---|---|---|
| Wheel filename | `qector_decoder_v3-1.0.0-cp311-cp311-win_amd64.whl` | `wheels/` directory |
| `package_version` field in verification JSON | `"1.0.0"` | `results/results_*.json` |
| Compiled Rust `__version__` | `1.0.0` (from `CARGO_PKG_VERSION`) | `src/lib.rs` |
| `__fallback_version__` in `__init__.py` | `"1.0.0"` | `python/qector_decoder_v3/__init__.py:412` |
| `Cargo.toml` package version | `1.0.0` | `Cargo.toml` |
| `pyproject.toml` `[project]` version | `1.0.0` | `pyproject.toml` |
| `CITATION.cff` / `codemeta.json` | `1.0.0` | both |
| `rust_core.sha256` | refreshed to the SEC-02 repack (`16c53223…`) | repo root |

**Verified for the v1.0.0 cut:** the `__fallback_version__` guard no longer lags the wheel, and the `test_version_is_071` → `test_version_is_100` rename applies to the external verification harness (`Desktop/v0.7.1/tests/test_01_smoke_api.py`), not a repo file.

**What to grep for before tagging v1.0.0** (one-shot checklist):

```text
pyproject.toml           — [project] version
Cargo.toml               — [package] version
Cargo.lock               — name = "qector_decoder_v3" version
python/.../__init__.py   — __fallback_version__ AND all "QECTOR v{__version__}" banner lines
python/.../__init__.py   — decoder_changes["0.7.1"] / decoder_changes["0.7.0"] history dict
tests/test_01_smoke_api.py — test_version_is_071  → rename + assert
CITATION.cff             — version:
codemeta.json            — "version":
README.md / PYPI_README.md — version labels in install commands
docs/INSTALL.md          — version labels
examples/*.py            — any pinned `qector-decoder-v3==0.7.x` in shebangs/imports
.github/workflows/release-build.yml — version regex (if any) in build matrix
.github/ACTIONS env pinning — pyproject version bump step
```

---

## Summary for the v1.0.0 cut

| Status | Count | Items |
|---|---|---|
| ✅ In source, ship as documented | 9 | CS-OSD+LLR damping, verification harness, LER gate, license-pinning fixtures, CUDABatchDecoder f64, dual-stream race fix, CUDABatchDecoder edge_len leak, SparseBlossomDecoder zero-alloc, **CUDABpOsdDecoder.decode** |
| ⚠️ Reconcile before tagging | 1 | ColourCodeDecoder default (`bposd` vs `cluster_bposd`) — pick one of two resolutions |
| ❌ Cut from changelog or implement | 0 | — |
| 🔧 Version-string sync | 1 sweep | All `0.7.x` references in metadata → `1.0.0` (see checklist above) |

**No blocking issues for the v1.0.0 cut** as long as the ColourCodeDecoder default and the version-string sync are done. Nine of ten [Unreleased] items are verified in source.
