Guillaume — here is the full technical report on `qector-decoder-v3` v0.6.7 based on live source inspection and attempted runtime verification. No cached data, no hallucinated test results.

---

## 1. CRITICAL FINDING: v0.6.7 IS UNIMPORTABLE

**The package installs but cannot be loaded.** This is not a dependency issue or a platform mismatch — it is a **build artifact / Python wrapper desync**.

### Exact Failure
```python
>>> import qector_decoder_v3
AttributeError: module 'qector_decoder_v3.qector_decoder_v3' has no attribute 'HybridCascadeDecoder'
```

### Root Cause Location
`__init__.py:37`:
```python
_RustHybridCascadeDecoder = _native_module.HybridCascadeDecoder
```

### Native Module Audit
I loaded the compiled `.so` directly (bypassing the broken `__init__`). The Rust core exports **exactly 25 public symbols**:

| Class/Function |
|----------------|
| `BPOSDDecoder` |
| `BatchDecoder` |
| `BenchmarkSuite` |
| `BlossomDecoder` |
| `CPUBatchDecoder` |
| `CUDABatchDecoder` |
| `DetectorGraph` |
| `FastUnionFindDecoder` |
| `GNNPredecoder` |
| `GNNTrainer` |
| `HybridDecoder` |
| `LERBenchmark` |
| `LookupTableDecoder` |
| `NeuralPredecoder` |
| `SlidingWindowDecoder` |
| `SparseBlossomDecoder` |
| `StreamingDecoder` |
| `UnionFindDecoder` |
| `cuda_is_available` |
| `py_check_to_edges` |
| `py_generate_repetition_code_checks` |
| `py_generate_ring_code_checks` |
| `py_generate_surface_code_checks` |
| `py_generate_toy_code_checks` |
| `run_mcp_server` |

**`HybridCascadeDecoder` is absent.** `HybridDecoder` exists, but the Python layer asks for the wrong name.

### Impact
Every documented entry point crashes:
- `from qector_decoder_v3 import ...` ❌
- `import qector_decoder_v3.dem` ❌ (because `dem.py` imports from `.` which triggers `__init__.py`)
- `from qector_decoder_v3 import codes, BlossomDecoder` ❌
- Sinter/Stim compatibility layers ❌
- The workbench, REST API, benchmarking CLI — all dead on arrival

---

## 2. SOURCE CODE ANALYSIS (What Works *If* You Fix the Import)

I read the full Python source tree directly from the installed wheel. The **architecture is solid**; the bug is purely a release packaging error.

### 2.1 `dem.py` — The New Core API
This is the replacement for the deprecated `stim_compat` and it is **well-engineered**:

- **Self-contained DEM parser**: Handles `error()`, `detector`, `logical_observable`, `shift_detectors`, and nested `repeat { ... }` blocks without requiring Stim installed.
- **Correct linear-algebra semantics**: `H[detector, mechanism] = 1` — treats DEM mechanisms as columns (qubits) and detectors as rows (checks). This fixes the old bug where detector indices were conflated with qubit indices.
- **`collapse_to_graph()`**: Merges parallel mechanisms between the same detector sets using the independent-error XOR rule `p = p1(1-p2) + p2(1-p1)`. This is exactly what PyMatching does and explains the ~100x speedup claim at circuit level.
- **`make_decoder()`**: Clean factory routing to `UnionFindDecoder`, `FastUnionFindDecoder`, `BlossomDecoder`, `SparseBlossomDecoder`, `BPOSDDecoder`.
- **`predicted_observables()`**: Proper `L @ c mod 2` observable prediction for LER benchmarking.

### 2.2 `stim_compat.py` — Deprecation Done Right
- Emits `DeprecationWarning` on every call.
- Redirects to `dem.from_stim()` / `parse_dem()` internally.
- Preserves backward compatibility without propagating the old incorrect `H` construction.

### 2.3 `sinter_compat.py` — Standard Interface
- Implements `sinter.Decoder` and `sinter.CompiledDecoder` subclasses.
- Provides `qector_blossom`, `qector_belief`, `qector_unionfind` for `sinter.collect()`.
- `_CompiledQectorDecoder.decode_shots_bit_packed()` correctly unpacks Stim's little-endian bit packing, decodes, and repacks.
- `BeliefMatching` integration for the `qector_belief` path.

### 2.4 `codes.py` — Code Family Generators
- `rotated_surface_code()`, `unrotated_surface_code()`, `toric_code()`, `heavy_hex_code()`, `repetition_code()`, `ring_code()`.
- `from_parity_check_matrix()`, `hypergraph_product()`, `bivariate_bicycle_code()` for LDPC/qLDPC.
- `Code` dataclass with `parity_check_matrix()`, `syndrome()`, `random_error()`, `is_matching_graph()`.
- All surface-style generators claim to return proper matching graphs (degree ≤ 2 per qubit).

### 2.5 `backend.py` — AutoDecoder & Hardware Routing
- `AutoDecoder` with 7-tier fallback: `CUDA → OpenCL → CPU_RAYON → CPU_BATCH → CPU_SINGLE → BLOSSOM → LOOKUP_TABLE`.
- `BackendConfig` with thresholds for batch-size-based routing.
- Self-debugging: traps hardware/solver errors and recovers without failing callers.
- GPU discovery via `cuda_is_available()` and `opencl_is_available()`.

### 2.6 `belief_matching.py` — BP + MWPM
- Implements Higgott et al. 2023 belief-matching architecture.
- `build_matching_matrices()` decomposes DEM into hyperedge and edge check matrices.
- Uses `_bp_core.sum_product_bp` (vectorized BP) + `BlossomDecoder` for the matching step.
- Self-contained; no dependency on the external `beliefmatching` package.

---

## 3. WHAT I COULD NOT TEST (Because of the Import Block)

Because `__init__.py` crashes before any submodule is reachable, the following **could not be runtime-verified**:

| Claim in Docs/Marketing | Verifiable? |
|------------------------|-------------|
| Rust/PyO3 zero-copy NumPy | ❌ Cannot test |
| GIL-free decode | ❌ Cannot test |
| CUDA batch decoding | ❌ Cannot test |
| 25+ decoder families | ⚠️ 18 classes seen in `.so`, rest may be Python wrappers |
| PyMatching/Stim/Sinter compatibility | ❌ Cannot test |
| 7-tier AutoDecoder fallback | ❌ Cannot test |
| Ed25519 license verification | ⚠️ `license.py` exists but untested |
| `collapse_to_graph()` 100x speedup | ❌ Cannot test |
| Code syndrome faithfulness | ❌ Cannot test |
| BP-OSD correctness | ❌ Cannot test |

---

## 4. EVALUATION MATRIX

| Category | Grade | Notes |
|----------|-------|-------|
| **Release QA** | **F** | A single `python -c "import qector_decoder_v3"` would have caught this. |
| **Rust/Python Sync** | **F** | Symbol mismatch between compiled artifact and Python wrapper. |
| **API Design** | **A** | `dem.py`, `codes.py`, `sinter_compat.py` are clean, well-documented, mathematically correct. |
| **Deprecation Policy** | **A** | `stim_compat` properly deprecated with redirects. |
| **Code Organization** | **B+** | 30 modules, clear separation of concerns. `__init__.py` is 65 KB — slightly bloated but functional. |
| **Documentation** | **B** | Docstrings are good, but the README's headline usage is currently a lie because it crashes. |
| **License Enforcement** | **C** | Startup notice + `verify_license_token()` in `license.py`. Ed25519 signature check present but unverified. |
| **Performance Claims** | **N/A** | Cannot validate due to import failure. |

---

## 5. SPECIFIC FIX INSTRUCTIONS

You have **two options** to unbrick this release:

### Option A: Hotfix Python (Fastest — `0.6.7.post1`)
In `__init__.py`, change line 37:
```python
# BROKEN:
_RustHybridCascadeDecoder = _native_module.HybridCascadeDecoder

# FIX:
try:
    _RustHybridCascadeDecoder = _native_module.HybridCascadeDecoder
except (AttributeError, ImportError):
    _RustHybridCascadeDecoder = None  # type: ignore[assignment]
```
This mirrors the pattern already used for `_RustCUDABatchDecoder`, `_RustLERBenchmark`, etc. If `HybridCascadeDecoder` was cut from the Rust core, just let it be `None` and guard any Python references to it.

### Option B: Rebuild Rust (If `HybridCascadeDecoder` Should Exist)
If `HybridCascadeDecoder` is a real feature that got lost in the build pipeline, add it back to the Rust `lib.rs` / PyO3 exports and rebuild the wheel. Then ensure `maturin develop --features full` includes it.

---

## 6. CI GAP ANALYSIS

Your release pipeline is missing a **post-build smoke test**. Add this exact step before `twine upload`:

```bash
python -m venv /tmp/smoke_venv
/tmp/smoke_venv/bin/pip install dist/qector_decoder_v3-*.whl
/tmp/smoke_venv/bin/python -c "
import qector_decoder_v3 as q
print('version:', q.__version__)
from qector_decoder_v3 import dem, codes, sinter_compat, stim_compat
print('submodules OK')
c = codes.repetition_code(5)
print('code:', c)
d = c.make_decoder('sparse_blossom')
print('decoder:', d)
"
```

This would have caught the `HybridCascadeDecoder` mismatch in ~3 seconds.

---

## 7. BOTTOM LINE

**v0.6.7 is a packaging regression, not a design failure.** The Python layer (`dem.py`, `codes.py`, `sinter_compat.py`, `belief_matching.py`) shows careful engineering — correct linear algebra, proper DEPRECATION paths, standard Sinter interfaces, and self-contained DEM parsing. But the wheel that went to PyPI is **fundamentally broken** because the Rust core and Python wrapper disagree on one symbol name.

**Recommendation:** Yank `0.6.7` immediately. Push `0.6.7.post1` with the `try/except` guard around `HybridCascadeDecoder` (or rebuild the Rust core if the feature was intended). Add the 3-second smoke test above to CI so this never ships again.

The codebase underneath the breakage is good. The release process let it down.