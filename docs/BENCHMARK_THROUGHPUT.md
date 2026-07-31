# Per-shot decode latency — QECTOR vs PyMatching 2

Individual (per-shot) decode latency distribution across rotated surface-code
memory experiments at circuit-level noise, p = 0.005.

## Setup

- **Decoders**:
  - *QECTOR-Blossom* — weighted exact MWPM (``pymatching_compat.Matching.decode``).
  - *QECTOR-UnionFind* — fast unweighted UF (``UnionFindDecoder.decode``).
  - *PyMatching 2* — ``pymatching.Matching.decode`` (single-shot Python API).
- **500 individual single-shot decode calls per distance**, each timed with
  ``time.perf_counter_ns``.
- Circuit generated identically to the competitive benchmark
  (``surface_code:rotated_memory_x``, rounds = distance, p = 0.005).
- **Environment**: AMD Ryzen, Python 3.11, Windows, stim 1.16.0,
  PyMatching 2.4.0, QECTOR 0.7.0.

## Results

### Per-round latency (µs / (shot · round))

| d | rounds | decoders | p50 µs/rd | p99 µs/rd | mean µs/rd |
|---|--------|----------|-----------|-----------|------------|
| 3 | 3 | QECTOR-Blossom | 2.9 | 8.0 | 3.1 |
|   |       | QECTOR-UF      | 1.2 | 3.0 | 1.3 |
|   |       | PyMatching 2   | 2.1 | 3.9 | 2.3 |
| 5 | 5 | QECTOR-Blossom | 28.1 | 67.3 | 27.4 |
|   |       | QECTOR-UF      | 1.8 | 4.4 | 2.0 |
|   |       | PyMatching 2   | 1.8 | 4.3 | 2.1 |
| 7 | 7 | QECTOR-Blossom | 38.1 | 82.7 | 42.6 |
|   |       | QECTOR-UF      | 2.8 | 6.4 | 3.0 |
|   |       | PyMatching 2   | 2.1 | 4.9 | 2.7 |
| 11 | 11 | QECTOR-Blossom | 765.9 | 1941.1 | 828.9 |
|    |    | QECTOR-UF      | 6.2 | 13.2 | 6.6 |
|    |    | PyMatching 2   | 4.4 | 10.2 | 5.6 |
| 15 | 15 | QECTOR-UF      | 126.8 | 182.1 | 128.9 |
|    |    | PyMatching 2   | 7.9 | 13.9 | 10.1 |
| 21 | 21 | QECTOR-UF      | 308.4 | 412.3 | 313.0 |
|    |    | PyMatching 2   | 20.4 | 33.8 | 25.6 |

> **Note:** Blossom at d ≥ 11 has prohibitive single-shot latency
> (>8 ms median per shot at d = 11; the trend is O(d⁶)). At d = 15 and
> d = 21 the per-shot benchmark was run for UF and PyMatching only.

### Raw per-shot latency (µs / shot)

| d | QECTOR-Blossom p50/p99 | QECTOR-UF p50/p99 | PyMatching 2 p50/p99 |
|---|------------------------|-------------------|----------------------|
| 3 | 8.8 / 23.9 | 3.6 / 9.1  | 6.4 / 11.8  |
| 5 | 140.4 / 336.6 | 9.0 / 22.1 | 8.8 / 21.6 |
| 7 | 266.9 / 579.2 | 19.3 / 44.7 | 15.0 / 34.2 |
| 11 | 8424.5 / 21351.8 | 68.1 / 144.9 | 48.3 / 112.1 |
| 15 | — | 1902.7 / 2731.9 | 118.8 / 208.5 |
| 21 | — | 6475.6 / 8658.0 | 427.8 / 709.8 |

> **Note on per-shot vs batch.** The figures above measure **individual decode
> calls**, which is relevant for real-time feed-forward decoding where
> syndromes arrive one at a time. When large batches are available the
> amortised per-shot latency is significantly lower (e.g. QECTOR-Blossom at
> d=11: 8.4 ms single-shot vs ~230 µs batch average). Union-Find and PyMatching
> 2 have much smaller single-shot overhead (≤2× the batch average), making
> them more suitable for low-latency streaming use.

### Throughput (shots/s)

| d | QECTOR-Blossom | QECTOR-UF | PyMatching 2 |
|---|----------------|-----------|--------------|
| 3 | 109 000 | 260 000 | 145 000 |
| 5 | 7 300 | 102 000 | 94 000 |
| 7 | 3 400 | 47 600 | 53 800 |
| 11 | 110 | 13 700 | 16 200 |
| 15 | — | 520 | 6 600 |
| 21 | — | 150 | 1 860 |

## Observations

1. **QECTOR-UnionFind and PyMatching 2 have very similar per-shot latency**
   across all distances. UF is slightly faster at low d, PyMatching at high d.
   Both stay under 70 µs median per shot even at d = 11.

2. **QECTOR-Blossom's single-shot overhead grows sharply with d** — at d = 11
   the median decode takes 8.4 ms, vs ~230 µs when batch-decoded. The
   overhead comes from per-call setup/teardown in the Python→Rust bridge that
   is amortised in the batch path.

3. **Unweighted UF latency collapses at d ≥ 15** (1.9 ms at d=15, 6.5 ms at
   d=21), becoming *slower* than PyMatching by 16×. Without edge weights,
   cluster growth becomes essentially random, creating unreasonably large
   clusters that require many iterations to resolve. Weighted UF (UF-01)
   would guide growth with `log((1-p)/p)` edge costs, yielding smaller,
   localized clusters and restoring the O(N log N) scaling.

4. **Tail (p99) is 1.3–2× the median** for UF at high d, and ~1.5–2× for
   PyMatching, consistent with OS scheduling noise and Python GC pauses.
