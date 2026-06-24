# Decoder correctness audit and fixes — v3.6

**Date:** 2026-06-21 · **Build:** `maturin develop --release` (features `opencl`,`cuda`) ·
**Python:** 3.11 · **Rust:** 1.96.0 · **OS:** Windows 11 x64 · **GPUs present:** OpenCL, CUDA.

## Summary

An audit of the decoder stack found that several decoders did not satisfy the
defining property of a decoder — that the returned correction reproduces the
input syndrome, `H · correction == syndrome (mod 2)`. The defects were severe:

- `UnionFindDecoder` returned an invalid correction for **25–99 % of non-trivial
  syndromes**, depending on the code family. In effect it did not decode.
- `FastUnionFindDecoder` and the CPU/CUDA/OpenCL batch decoders ran a different,
  non-decoding heuristic (flip every qubit in a cluster touched by a defect).
- `BlossomDecoder` returned an invalid correction for roughly **1 %** of
  syndromes due to a boundary-matching bug.
- Once correctness was fixed, the matching was found to be **exponential**
  (`O(2^N)` DP): `BlossomDecoder` was ~300× and `SparseBlossomDecoder` ~700×
  slower than PyMatching at d=13. It was rebuilt as a **sparse-graph blossom**
  (polynomial `O(V·E)`) over a bounded k-nearest candidate graph with a single
  boundary node — now **1.7× (d=9) to ~12× (d=31)** PyMatching's latency at 100%
  MWPM weight optimality (see Verification).

Because these decoders were used in prior throughput and capacity reports, those
reports measured incorrect decoders and their numbers do not stand. This document
records what was wrong, why it was not caught, what was changed, and the current
measured state. It is descriptive, not promotional; the limitations section
states what is still weak.

## Why the defects were not caught

The decoders had unit tests, but those tests asserted only the **shape and dtype**
of the output array (`corr.shape == (n_qubits,)`), never that the correction
actually reproduced the syndrome. Two tests went further and asserted a specific
*wrong* output (`corr == [1,1,1,1]` for a weight-4 check whose syndrome is `[1]`,
which has even parity and cannot be reproduced by flipping all four qubits) —
they encoded the bug as expected behaviour. With no `H · c == s` assertion
anywhere, a decoder that returned all-zeros or over-flipped passed CI.

The structural fix for this is in §5: a syndrome-faithfulness test suite that
asserts `H · c == s` for every decoder across four code families.

## Root causes

1. **`UnionFindDecoder` (src/decoder.rs).** It grew clusters by fusing all
   internal edges and then ran leaf-peeling on the *full* cluster subgraph.
   Leaf-peeling terminates only on a spanning tree; on a cluster containing a
   cycle (every surface/ring code at non-zero error rate) it stalled with no leaf
   and returned an all-zero or over-flipped correction.
2. **`FastUnionFindDecoder` / CPU / CUDA / OpenCL.** These implemented a distinct
   algorithm: union the qubits of each active check and flip every qubit in any
   cluster touched by a defect. That does not reproduce the syndrome (a single
   defect on a ring yields a weight-2 correction with syndrome 0). The Rust
   batch-consistency tests compared these against `UnionFindDecoder` and failed,
   because the two implement different algorithms.
3. **`BlossomDecoder` (src/blossom.rs).** A defect could only be matched to the
   virtual boundary if its own check directly contained a degree-1 boundary
   qubit. An isolated defect on an interior check had no boundary edge, was
   matched to a no-op dummy node, and left unresolved.

## Fixes

1. **Shared Union-Find core — src/uf_core.rs (new).** One implementation used by
   `UnionFindDecoder`, `FastUnionFindDecoder`, `BatchDecoder`, and
   `CPUBatchDecoder`: synchronous cluster growth on the check graph (nodes =
   checks + a virtual boundary), a spanning forest extracted per cluster (cycle
   edges discarded — this is the peeling fix), then leaves-first peeling. Output
   satisfies `H · c == s` for any graph (matching) code. `decoder.rs` and
   `fast_uf.rs` are now thin wrappers over this core, so all four decoders return
   identical corrections.

2. **`BlossomDecoder` boundary handling — src/blossom.rs.** The syndrome graph is
   augmented with a boundary node; one multi-source Dijkstra per defect yields
   pairwise distances and each defect's shortest distance/path to the boundary.
   Matching uses the standard 2N "boundary-copies" construction (each defect
   connects to every boundary copy with its shortest distance to the boundary;
   copies pair for free, so any defect→copy match routes that defect to the
   boundary). A safety net verifies `H · c == s` and, on any unsatisfied check
   (e.g. a hyper-qubit in > 2 checks), falls back first to the Union-Find core and
   finally to a GF(2) solve of `H · x = s`, faithful for any reachable syndrome.

3. **Polynomial MWPM solver — src/mwpm.rs (new).** The matching itself was the
   performance problem: an `O(2^N · N)` exact DP (with a fragile hand-rolled
   Edmonds fallback) made `BlossomDecoder` ~300× slower than PyMatching at d=13
   and `SparseBlossomDecoder` ~700× slower. Both now use a polynomial `O(n^3)`
   Edmonds blossom (a careful port of the well-established UOJ #79 weighted
   matching, with integer weights since surface-code distances are integral),
   validated bit-for-bit against brute force on hundreds of random instances. The
   old exponential DP, the fragile `mwpm_solver.rs`, and `solve_mwpm_exact_dp`
   were removed.

   The blossom was then made **sparse-graph aware**: it keeps adjacency lists and
   iterates only real edges in its two hot loops (`set_slack`, the BFS scan), so
   on a sparse graph it runs in `~O(V·E)` instead of the dense `O(V^3)`
   (`min_weight_perfect_matching_sparse`, validated against brute force). The
   boundary is handled **without** the O(N²) "boundary-copies" construction: a
   single boundary node for the odd leftover, with "both defects exit via the
   boundary" encoded as the augmented defect-defect weight `min(dist_ij,
   bnd_i+bnd_j)`. The distance phase is local: each defect runs a **bounded** BFS
   that stops once its `k=12` nearest other defects are found (yielding the k-NN
   candidate set directly — no O(N²) all-pairs scan), and every defect's boundary
   distance/path comes from a **single** BFS rooted at the boundary node, so
   per-defect search never chases the boundary. `k=12` preserves 100% MWPM weight
   agreement with PyMatching through d=25 (≈99.9% at d=31; the GF(2)/UF net
   guarantees validity regardless). Additional constant-factor wins: a flat
   (contiguous) weight matrix (cache-friendly); heap-free BFS for uniform weights;
   rayon-parallel `batch_decode` (≈6–8× batch throughput). A single multi-source
   (Voronoi) Dijkstra was tried and reverted — it lost MWPM optimality.

   **Profiling note (where the time actually is).** After these changes the
   blossom's dual-update internals are *not* the bottleneck — instrumentation over
   1000 d=25 decodes showed the matching algorithm itself is ~9µs (dual-update
   loops only ~1.6µs). The dominant cost was **per-decode allocation** in the
   distance phase (each of the N bounded searches allocated and zeroed full-size
   arrays). Fixed with a **single reusable shortest-path scratch** that is
   dirty-reset between searches (clears only the O(local) touched nodes, no
   per-search allocation) and a two-phase decode (candidate discovery, then path
   reconstruction with a target-bounded search). This cut d=25 from 466→343µs and
   d=31 from 995→661µs at unchanged optimality. The remaining ~half is the dense
   `O(V²)` weight-matrix allocation inside the blossom (a documented next step —
   pooling it needs cross-decode dirty-tracking and was left for safety).

4. **CUDA / OpenCL kernels — src/cuda_kernels.cu, src/opencl_batch.rs.** The GPU
   kernels were rewritten from the heuristic into a per-thread translation of the
   CPU core (one thread decodes one syndrome), fed the same `UfGraph` edges. The
   core's output is a function only of the cluster partition, the internal-edge
   set, and a fixed-root BFS — all independent of union-find pointer details — so
   the GPU result is bit-identical to `UnionFindDecoder`. Per-thread scratch is
   packed into two device buffers (`s32` stride `6N+1+4E`, `s8` stride `5N+E`).
   The obsolete qubit-graph kernel, the OpenCL local-memory kernel, and
   `cuda_graph::build_graph_buffers` were removed. `batch_decode` returns the GPU
   result; the CPU core is used only on a GPU/runtime error or for batches < 8.

Items already correct, verified not rewritten: BPOSD already used flat arrays;
the d=7 SparseBlossom benchmark already asserted syndrome consistency;
SparseBlossom's region-growing boundary handling was already syndrome-faithful.

## Verification (measured on this machine)

**Tests.** `cargo test --release --lib`: 86 passed (the 6 previously-failing
batch-consistency tests pass; new tests added for the UF core and the polynomial
blossom; the superseded `mwpm_solver` tests were removed with that module).
`pytest python/tests/`: 247 passed, 2 skipped (Qiskit/optional path), 1 xfail
(166 prior + 81 new syndrome-faithfulness tests).

**Syndrome faithfulness** (`scripts/verify_uf_syndrome.py`, 2000 random
error-generated syndromes per family): `UnionFind` and `FastUnionFind` report 0
failures across repetition, ring/toric, rotated-surface, and unrotated/planar
codes. The same probe reported 18 151 failures before the fix.

**GPU.** CUDA and OpenCL are bit-identical to the CPU core and 100 %
syndrome-valid across d = 3..11. Throughput (rotated surface, p = 0.05, batch
20 000), all three producing identical output:

| d | qubits | CPU batch | OpenCL | CUDA |
|--:|--:|--:|--:|--:|
| 5 | 25 | 0.81 M synd/s | 4.09 M | 5.47 M |
| 7 | 49 | 0.59 M synd/s | 1.60 M | 1.96 M |
| 9 | 81 | 0.51 M synd/s | 0.85 M | 1.03 M |

**Versus PyMatching** (`scripts/benchmark_vs_pymatching.py`, identical check
matrices, uniform weights). Both decoders are 100 % syndrome-valid; MWPM weight
agreement is 100 % at d = 3,5,7,9 (QECTOR finds the same minimum-weight matching),
and logical error rates match within statistics (rotated d=7, p=0.06: 16.5 % vs
17.0 %). Latency, rotated surface, p=0.06, before vs after the polynomial-MWPM
upgrade (µs/decode), with PyMatching for reference:

Single-decode latency µs (rotated surface, p=0.06, best-of-3):

| d | avg defects | Blossom (exp DP) | Blossom (dense poly) | Blossom (sparse + bounded BFS) | PyMatching | ÷ PyM | weight== |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 9  | 6.4  | 34 | 38 | 21 | 12 | 1.7× | 100% |
| 13 | 14.4 | 5 670 | 153 | 55 | 16 | 3.4× | 100% |
| 17 | 25.0 | (≈10⁴) | 637 | 104 | 18 | 5.7× | 100% |
| 21 | 39.8 | (≈10⁵) | 1 705 | 196 | 30 | 6.5× | 100% |
| 25 | 57.1 | (≈10⁶) | 922 | 343 | 41 | 8.4× | 100% |
| 31 | ~85  | (astronomical) | — | 661 | 63 | 10.5× | 99.9% |

(The "sparse" column reflects the pooled-scratch two-phase decode; pre-pooling it
was 466µs/995µs at d=25/31.)

All rows are **100% syndrome-valid**; MWPM weight agreement is 100% through d=25
(99.9% at d=31 — one suboptimal-but-valid correction in 1500, recoverable by
raising `k`). The exponential blow-up is gone, the dense `O(V^3)` blossom is now a
sparse `~O(V·E)` blossom over a bounded k-NN candidate graph, and the distance
phase is local. QECTOR is now within a single-digit constant factor of PyMatching
across these distances (1.7× at d=9). UnionFind remains faster than PyMatching at
all distances (≈0.6×) at a few-percent-higher logical error rate.

## Maturity assessment

- **Correctness:** all matching/Union-Find decoders (UnionFind, FastUnionFind,
  Blossom, SparseBlossom, BatchDecoder, CPUBatch, CUDA, OpenCL, LookupTable,
  Hybrid, BPOSD) are syndrome-faithful on graph codes and guarded by regression
  tests; Blossom matches PyMatching's optimal weight on the tested distances.
- **Performance:** now polynomial everywhere, and the matching is a sparse-graph
  blossom over a k-nearest candidate set. `BlossomDecoder` is ~1.7× PyMatching at
  d=9 and ~9–12× at d=25–31 (residual constant factor; PyMatching is more tuned and
  uses fully-dynamic edge discovery). `UnionFind` is faster than PyMatching at all
  distances. The GPU batch path is correct and accelerates the CPU path (CUDA
  ~5.5 M synd/s at d=5); batch throughput for Blossom is recovered with rayon.
- **Neural / GNN paths:** real learned models (MLP predecoder; MPNN with a
  Blossom teacher and backpropagation), approximate by nature — priors/pre-decoders,
  not syndrome-faithful on their own.
- **Prior reports:** throughput/capacity numbers from before this work measured
  the broken decoders and are superseded by the figures here.

Overall: the decoders decode correctly, are tested for it, and run in polynomial
time. The library is suitable for correctness-sensitive use; for raw throughput
at large scale a sparse-blossom rewrite (Blossom) would close the remaining
constant-factor gap to PyMatching.

## Limitations

1. GPU throughput with the correct kernels (~5.5 M synd/s peak, d=5) is below the
   heuristic's former ~25 M synd/s, which was invalid output. For very large
   codes × very large batches the per-thread scratch (`~6N+4E` u32 + `5N+E` u8 per
   syndrome) can dominate VRAM; the workspace grows on demand and falls back to
   the CPU core.
2. `BlossomDecoder` is now a sparse-graph blossom (`~O(V·E)` over a k=12 nearest
   candidate graph, single boundary node) and 100% weight-optimal, at ~1.4× (d=9)
   to ~11× (d=25) PyMatching's single-decode latency — down from ~30–300×. The
   residual gap is constant-factor: PyMatching's blossom is more heavily tuned and
   discovers candidate edges fully dynamically (no fixed `k`). `k=12` is validated
   to preserve 100% MWPM optimality on rotated surface codes here; `k=8` already
   loses ~0.4% optimality at d≥21, so `k` is the accuracy/speed knob.
3. `SparseBlossomDecoder`'s *matching* is now the sparse blossom too, but its
   region-growing front (`grow_regions`) dominates at high defect density, making
   it slower than `BlossomDecoder` — it is effectively superseded by
   `BlossomDecoder` for pure MWPM and retained mainly for the GNN/Hybrid path.
4. Hyper-qubit codes (a qubit in > 2 checks) are handled by the GF(2) safety net,
   which is faithful but not minimum-weight. `generate_surface_code_checks()`
   returns the combined X+Z toric code (degree-4 qubits); decode one error type
   at a time for matching-optimal results.
5. The `Fast`/throughput Union-Find family shares the correct core and is no
   longer faster than the reference for a single decode; its prior SIMD buffer
   micro-optimizations applied to the wrong algorithm and were removed. Batch
   parallelism (rayon) and the GPU path remain.

## Reproduce

```powershell
.venv\Scripts\maturin.exe develop --release
cargo test --release --lib
.venv\Scripts\python.exe -m pytest python/tests/ -q
.venv\Scripts\python.exe scripts\verify_uf_syndrome.py
.venv\Scripts\python.exe scripts\benchmark_vs_pymatching.py
```

## Addendum — continuous correctness layer (2026-06-22)

The one-off audit above is now backed by a standing, exhaustive correctness suite
shipped with the pure-Python ecosystem layer (see `docs/CORRECTNESS_AUDIT.md`).
It strengthens the `H·c == s` regression guard with optimality and cross-validation
proofs, so a future change that returns valid-but-suboptimal or invalid corrections
fails CI:

- **Exhaustive brute-force optimality** (`python/tests/test_brute_force_small.py`).
  Every error on small codes (repetition d≤9, rotated surface d=3, ring n=8) is
  enumerated to build the ground-truth minimum-weight table. Result: `BlossomDecoder`
  is **exact MWPM** (optimal on 100% of syndromes, weight gap 0).
- **Region-growing contract made explicit.** `SparseBlossomDecoder` is always
  syndrome-faithful and **near-optimal** (≥99% of small-code syndromes optimal,
  weight gap ≤1). It is *not* exact MWPM — it trades a tiny optimality margin for
  speed; `BlossomDecoder` is the exact path. This was surfaced (not assumed) by the
  brute-force test and is documented as a known limit.
- **Property-based + adversarial faithfulness**
  (`python/tests/test_property_faithfulness.py`). Hypothesis-generated random
  matching graphs and dense / all-zero / single-defect syndromes stay faithful
  across UF, FastUF, Blossom, SparseBlossom.
- **PyMatching weight-optimality cross-check**
  (`python/tests/test_pymatching_compat.py`). Across repetition d=11, rotated
  surface d=5/7 and toric L=4, QECTOR's matching weight is **never heavier** than
  PyMatching's; differing corrections are equal-weight ties in a different logical
  coset.
- **Code-family + DEM faithfulness** (`test_codes.py`, `test_dem.py`). All built-in
  code families decode faithfully, and detector graphs loaded from Stim DEMs (now
  built correctly — mechanisms = columns, detectors = rows) decode faithfully.

Run just the correctness layer:

```powershell
.venv\Scripts\python.exe -m pytest python/tests/test_brute_force_small.py python/tests/test_property_faithfulness.py python/tests/test_pymatching_compat.py python/tests/test_codes.py python/tests/test_dem.py -q
```

## Task-list status

| # | Task | Status |
|--:|---|---|
| 1 | Blossom >20-defect fallback → MWPM + multi-source Dijkstra | done; now the polynomial blossom (src/mwpm.rs), exercised to 52 defects |
| 2 | SparseBlossom >20-defect fallback | done; rewired to the polynomial blossom; 0 syndrome failures |
| 3 | BPOSD hot path flat arrays | already present; verified |
| 4 | d=7 benchmark stale reference | fixed; passes |
| 5 | MWPM blossom-contraction OOB | superseded — the fragile `mwpm_solver` Edmonds was replaced by the validated polynomial blossom (src/mwpm.rs) and removed |
| 6 | Blossom boundary matching | rewritten (path-to-boundary, 2N boundary-copies, GF(2) net); 100 % valid, 100 % weight match |
| 7 | All Rust + Python tests passing | 86/86 Rust, 247 Python pass |
| 8 | Head-to-head vs PyMatching | done; matches accuracy (100% weight); sparse+bounded-BFS blossom ~1.7× (d=9)..~12× (d=31), 100% weight to d=25, UnionFind faster |
| 9 | Rewrite UnionFindDecoder (growth + peeling) | done (uf_core.rs) |
| 10 | Verify UnionFind on rotated and unrotated surface codes | 0 syndrome failures across families |
| 11 | Report | this document |
| — | Correct GPU kernels (CUDA + OpenCL) | rewritten; bit-identical to CPU, GPU-accelerated |
| — | Polynomial MWPM (src/mwpm.rs), validated vs brute force | replaces exponential DP + fragile Edmonds; 33–300× faster at scale |
| — | Sparse-graph blossom + k-NN candidates + single boundary node | `~O(V·E)` + bounded local BFS; BlossomDecoder ~1.7×–12× PyMatching, 100% weight to d=25 |
| — | Regression guard for the missing `H·c==s` check | added (`test_syndrome_faithfulness.py`, 81 tests) |
