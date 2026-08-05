# GEMINI.md — MANDATORY AI AGENT RULES (ENFORCED)

Read this **BEFORE** touching any file. Violation of these rules is grounds for immediate session termination.

---

## 0. Prime Directive: NO FAKING

**NEVER claim a feature is implemented when it is not.** This is the single most important rule.

Previous AI sessions have been caught:
- Marking checklist items `[x]` when **zero code** was written
- Claiming "zero-allocation state machine" when the code does `Vec::new()` on every decode call
- Claiming "f64 CUDA kernel" when the `.cu` file contains only `float` operations
- Claiming "pinned host memory pipelining" when the Python file has zero mentions of `pinned`, `stream`, or `async`
- Writing docstrings that say "zero-alloc" while the function body allocates on every call

**If you cannot implement something, say "NOT IMPLEMENTED" and move on. Do NOT write a docstring claiming it works.**

---

## 1. Verification Protocol (MANDATORY before marking ANY item done)

Before marking **any** checklist item as `[x]`, you MUST:

1. **Show the actual code** you wrote (not just the docstring)
2. **Run a test** that exercises the specific feature claimed
3. **grep for the implementation** — if the feature claims "f64 kernel", grep for `f64` or `double` in the `.cu` file
4. **If the grep returns 0 matches, the feature is NOT implemented** — do NOT mark it done

### Specific verification commands:

```bash
# Verify f64 CUDA kernel actually exists
grep -c "f64\|double" src/cuda_kernels.cu    # must be > 0

# Verify pinned memory actually exists
grep -c "pinned\|cudaHostAlloc\|PinnedMemory" python/qector_decoder_v3/gpu_backend.py  # must be > 0

# Verify zero-alloc in blossom.rs hot path
grep -c "Vec::new\|HashMap::new\|BinaryHeap::new" src/blossom.rs  # count in decode() functions

# Verify CS-OSD exists
grep -c "cs_osd\|combination_sweep\|OSD-E\|damping" python/qector_decoder_v3/bposd.py  # must be > 0

# Verify Rust manifest integrity after ANY src/*.rs edit
python scripts/pack_rust_core.py check-manifest
```

---

## 2. Known Fakes from Previous Sessions (MUST BE FIXED, NOT RE-FAKED)

The following items were previously marked as complete but **have zero or inadequate implementation**. They must be either **genuinely implemented** or **honestly marked as `[ ]` NOT DONE**:

### 🔴 CONFIRMED FAKE (Zero code exists)
| # | Fake Claim | File | Reality |
|---|---|---|---|
| 1 | Pinned host memory `allocate_pinned_array()` | `gpu_backend.py` | File is a NumPy/CuPy selector. Zero pinned memory code. |
| 2 | `precision="f64"` CUDA kernel | `cuda_kernels.cu` | All computation uses `float` (f32). Zero f64 code. |
| 3 | Zero-alloc Sparse Blossom state machine | `blossom.rs`, `sparse_blossom.rs` | 4+ `Vec::new()` per decode in blossom.rs, 10+ in sparse_blossom.rs, 2 `HashMap::new()`. |
| 4 | Hyperedge colour code cluster expansion | `colour_code.py` | `build_matching_matrices` used only for data extraction. Zero cluster expansion. |

### 🟡 PARTIALLY FAKE (Exaggerated claims)
| # | Exaggerated Claim | File | Reality |
|---|---|---|---|
| 5 | "Dual-stream pinned memory pipelining" | `cuda_batch.rs` | Pseudo dual-stream exists but uses pageable memory → async is synchronous. Streams created/destroyed per call. |
| 6 | "CS-OSD / Combination Sweep OSD-E with damping" | `bposd.py` | Basic OSD-w combo search with itertools, NOT the formal CS-OSD. No damping. |
| 7 | "AVX2 SIMD bit-clearing" | `uf_core.rs` | Uses dirty-list tracking + epoch counters. Effective, but NOT AVX2/SIMD. |
| 8 | "Local Blossom tie-breaking" | `uf_core.rs` | Standard edge-index deterministic tiebreak. NOT Blossom tie-breaking. |

---

## 3. Absolute Prohibitions

| # | NEVER DO THIS |
|---|---|
| 1 | Mark a todo item `[x]` without running a test that proves the specific feature works |
| 2 | Write a docstring claiming "zero-allocation" when the function body contains `Vec::new()` |
| 3 | Claim a CUDA f64 kernel exists when `cuda_kernels.cu` only contains `float` |
| 4 | Claim "pinned memory" when no `cudaHostAlloc` / `cuMemHostAlloc` / CuPy `PinnedMemory` call exists |
| 5 | Claim "CS-OSD" when the code uses basic `itertools.combinations` brute force |
| 6 | Claim "AVX2/SIMD" when the code uses scalar Python/Rust loops |
| 7 | Push git tags (`v*`) or publish to PyPI without explicit human instruction |
| 8 | Skip `python scripts/pack_rust_core.py check-manifest` after editing any `src/*.rs` file |
| 9 | Claim a benchmark result you didn't actually run |
| 10 | "Fix" a failing test by weakening the assertion instead of fixing the code |

---

## 4. How to Honestly Report Status

Use these markers and ONLY these markers:

- `[x]` — **DONE**: Code exists, test passes, grep confirms implementation
- `[/]` — **IN PROGRESS**: Partially implemented, specific gaps documented
- `[ ]` — **NOT DONE**: Zero implementation exists
- `[!]` — **FAKE/REVERTED**: Previously claimed done, audit found zero/inadequate implementation

Example of honest reporting:
```
- [x] UfScratch thread-local reuse — `grep UfScratch src/fast_uf.rs` returns 12 matches, `thread_local!` confirmed on line 47, test_competitive_ler 5/5 PASS
- [!] f64 GPU kernel — PREVIOUSLY CLAIMED DONE, audit found 0 matches for f64/double in cuda_kernels.cu. MUST BE IMPLEMENTED.
- [ ] Pinned host memory — NOT STARTED
```

---

## 5. Rust Source Editing Rules (from CLAUDE.md, STILL ENFORCED)

- `src/*` is gitignored. After editing ANY `src/*.rs` file:
  1. Run `python scripts/pack_rust_core.py check-manifest`
  2. If manifest is stale, run `python scripts/pack_rust_core.py update-manifest`
  3. Rebuild wheel: `maturin build --release --no-default-features --features cuda --out dist`
  4. Force-reinstall: `pip install --force-reinstall --no-deps dist/*.whl`
  5. Run pytest to verify

---

## 6. Build & Test Checklist (run after EVERY change)

```bash
# 1. Manifest integrity
python scripts/pack_rust_core.py check-manifest

# 2. Version agreement
grep -m1 "^version" Cargo.toml pyproject.toml

# 3. Build wheel
maturin build --release --no-default-features --features cuda --out dist

# 4. Install
pip install --force-reinstall --no-deps dist/*.whl

# 5. Verify version
python -c "import qector_decoder_v3; print(qector_decoder_v3.__version__)"

# 6. Run tests (77 must pass, 0 skipped)
pytest python/tests/
```
