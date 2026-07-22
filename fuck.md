**CRITICAL BUG CONFIRMED IN v0.6.7 — DO NOT USE. IT IS BROKEN.**

I installed the actual wheel (`qector_decoder_v3-0.6.7-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`, SHA256 `a2dc97d5cdde12e6ae531c6ed28ba57c7ecde7ef7af2a6ab3dc3c72619f6b807`) and tested it for real.

---

### The Bug

`import qector_decoder_v3` **fails immediately** with:

```
AttributeError: module 'qector_decoder_v3.qector_decoder_v3' has no attribute 'HybridCascadeDecoder'
```

**Root cause:** In `__init__.py` line 37, you unconditionally import:

```python
_RustHybridCascadeDecoder = _native_module.HybridCascadeDecoder
```

But `HybridCascadeDecoder` **does not exist** in the compiled `.so`. I dumped every symbol from the compiled Rust extension — here is what is actually present:

| Symbol | Status |
|--------|--------|
| `UnionFindDecoder` | ✓ Present |
| `FastUnionFindDecoder` | ✓ Present |
| `BlossomDecoder` | ✓ Present |
| `SlidingWindowDecoder` | ✓ Present |
| `StreamingDecoder` | ✓ Present |
| `BatchDecoder` | ✓ Present |
| `CPUBatchDecoder` | ✓ Present |
| `BenchmarkSuite` | ✓ Present |
| `LookupTableDecoder` | ✓ Present |
| `BPOSDDecoder` | ✓ Present |
| `NeuralPredecoder` | ✓ Present |
| `DetectorGraph` | ✓ Present |
| `GNNPredecoder` | ✓ Present |
| `GNNTrainer` | ✓ Present |
| `LERBenchmark` | ✓ Present |
| `SparseBlossomDecoder` | ✓ Present |
| `HybridDecoder` | ✓ Present |
| **`HybridCascadeDecoder`** | **✗ MISSING** |
| `CUDABatchDecoder` | ✓ Present |

**This is the exact same failure pattern as v0.6.5** (where `OpenCLBatchDecoder` was unconditionally imported but missing from the compiled wheels). You fixed 0.6.5, but 0.6.7 introduces the same class of bug with a different symbol.

---

### What Else I Verified

- **Wheel integrity:** Valid, 721,356 bytes, installs cleanly via direct wheel install.
- **Metadata:** Correct version, correct platform tags.
- **Compiled module:** Loads successfully when imported in isolation; only `HybridCascadeDecoder` is missing.
- **Other unconditional imports:** All other classes/functions referenced in `__init__.py` exist in the `.so`.

---

### My Honest Assessment

**v0.6.7 is non-functional for all users.** Every single person who installs it will get an `AttributeError` on import. This is not a niche edge case — it is total breakage.

**You need to:**
1. **Yank v0.6.7 immediately** (same as you did for 0.6.5).
2. Fix `__init__.py` to guard `HybridCascadeDecoder` with `try/except AttributeError`, same pattern you use for `LERBenchmark` and `CUDABatchDecoder`.
3. Add a CI smoke test that literally does `python -c "import qector_decoder_v3"` on the built wheel before uploading. This would have caught both the 0.6.5 and 0.6.7 failures.

You asked me not to fake it. I didn't. The package is broken.