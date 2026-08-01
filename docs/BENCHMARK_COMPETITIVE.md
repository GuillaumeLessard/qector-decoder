# Competitive benchmark — QECTOR vs external decoders (circuit-level)

> **Withdrawn: no accuracy or latency figures are published for v0.7.0.** The
> result tables that previously lived here were produced on 2026-07-30 against a
> core whose tracked fingerprint (`rust_core.sha256`) has moved twice since
> (Jul 31, Aug 1), and the LER column of the newer workbench dataset was found
> not to be attributable to the decoder named in each row (see the release
> notes). A figure taken from a different binary, or attributed to the wrong
> decoder, is worse than no figure. The methodology below is retained so the
> comparison can be re-run and archived properly.

## What the comparison covers

A circuit-level comparison of decoding behaviour on rotated-surface-code memory
experiments against external reference decoders:

- **Circuit**: `stim.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=d)`
- **Noise**: circuit-level, `after_clifford_depolarization = before_measure_flip =
  after_reset_flip = 0.005`
- **Decoding problem**: built from the Stim Detector Error Model
  (`decompose_errors=True`), then **collapsed** so parallel mechanisms between the
  same detector pair become one min-weight edge.
- **Decoders**:
  - *QECTOR-Blossom* — weighted exact polynomial MWPM (uses `log((1-p)/p)` edge weights).
  - *QECTOR-UF* — `UnionFindDecoder` (fast, unweighted; the weighted variant is
    available on the GPU backends and in `uf_core::grow_weighted`).
  - *PyMatching* — `pymatching.Matching.from_detector_error_model` (reference).
  - *ldpc BP-OSD* — external `ldpc.BpOsdDecoder`, OSD-CS order 0, 30 BP iterations.
  - *BeliefMatching* — external `beliefmatching.BeliefMatching`, product-sum BP, 20 iters.

## How results are produced

- LER is scored with Wilson 95% intervals; the shot counts per cell and the
  environment (Python version, NumPy, Stim, PyMatching, ldpc, beliefmatching
  versions, host CPU) are recorded with the artifact.
- The long-standing qualitative findings of the earlier run are unchanged and
  are not withdrawn: unweighted Union-Find operates above threshold at
  circuit level (ignoring edge weights is known to cost accuracy — the weighted
  variant exists to close that gap), and PyMatching remains the latency
  reference for plain exact MWPM. No magnitude is claimed here; the artifacts
  must be re-measured before any number is quoted.

## Reproduce

Requires `qector_decoder_v3` (built from source for the weighted UF updates),
plus `stim`, `pymatching`, `ldpc`, `beliefmatching`:

```bash
pip install stim pymatching ldpc beliefmatching
python scripts/competitive_stim_ler.py \
    --distances 3 5 7 --shots 20000 \
    --out benchmark_results/c1_05_extended
# Higher distances (QB/PM/UF only):
python scripts/competitive_stim_ler.py \
    --distances 9 11 --shots 40000 \
    --out benchmark_results/competitive_stim_ler
```
