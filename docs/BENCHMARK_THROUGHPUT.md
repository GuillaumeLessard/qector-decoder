# Decode latency — capability statement and benchmark methodology

> **Withdrawn: no performance figures are published for v0.7.0.** The measured
> tables that previously lived here were produced on 2026-07-30 against a core
> whose tracked fingerprint (`rust_core.sha256`) has moved twice since (Jul 31,
> Aug 1). A CUDA kernel optimisation lands directly on the numbers a latency
> benchmark measures, so those figures describe a different binary than the one
> this release ships. Any latency or throughput figure in this repository would
> have to come from a fresh run against the current core, archived with its
> artifact. The capability and methodology content below is retained; the tables
> are not.

## Decoder paths covered

- **QECTOR-Blossom** — weighted exact MWPM, callable through
  `pymatching_compat.Matching.decode` as well as `BlossomDecoder.decode`.
- **QECTOR-UnionFind** — fast unweighted Union-Find
  (`UnionFindDecoder.decode`). A weighted growth variant exists on the GPU
  backends (`edge_weights` argument) and in `uf_core::grow_weighted`.
- **PyMatching 2** — external reference decoder, used for comparison only.

Both per-shot (`decode`) and batched (`parallel_batch_decode`, `BatchDecoder`,
GPU `batch_decode`) entry points exist; the batched paths amortise per-call
setup that the single-shot path pays each call.

## Benchmark methodology (how a latency benchmark is run)

- 500 individual single-shot decode calls per distance, each timed with
  `time.perf_counter_ns`.
- Circuit generated identically to the competitive benchmark
  (`surface_code:rotated_memory_x`, rounds = distance, p = 0.005).
- Reference environment for the withdrawn run: AMD Ryzen, Python 3.11,
  Windows, stim 1.16.0, PyMatching 2.4.0, QECTOR 0.7.0.

Results are only citable once they are re-measured on the current core and the
raw output is archived with its environment block (see
`docs/METHODOLOGY.md`).
