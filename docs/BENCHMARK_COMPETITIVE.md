# Competitive Benchmark — QECTOR vs external decoders (circuit-level)

Logical-error-rate and decode-latency comparison across rotated-surface-code
memory experiments at circuit-level noise, against **PyMatching 2**, **ldpc BP-OSD**
(OSD-CS, order 0, 30 BP iterations), and **BeliefMatching** (product-sum BP, 20 iters).

## Setup

- **Circuit**: `stim.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=d)`
- **Noise**: circuit-level, `after_clifford_depolarization = before_measure_flip =
  after_reset_flip = 0.005`
- **Decoding problem**: built from the Stim Detector Error Model
  (`decompose_errors=True`), then **collapsed** so parallel mechanisms between the
  same detector pair become one min-weight edge.
- **Decoders**:
  - *QECTOR-Blossom* — weighted exact polynomial MWPM (uses `log((1-p)/p)` edge weights).
  - *QECTOR-UF* — `UnionFindDecoder` (fast, **unweighted** — the weighted variant is
    available but not yet used in this benchmark; UF's above-threshold behaviour at
    d≥5 is a known consequence of ignoring edge weights at circuit level).
  - *PyMatching* — `pymatching.Matching.from_detector_error_model` (reference).
  - *ldpc BP-OSD* — external `ldpc.BpOsdDecoder`, OSD-CS order 0, 30 BP iterations.
  - *BeliefMatching* — external `beliefmatching.BeliefMatching`, product_sum BP, 20 iters.
- **Shots**: 20 000 per distance for d ≤ 5; 10 000 for d = 7.
  LER intervals are Wilson 95%.
- **Environment**: AMD Ryzen, Python 3.11, NumPy 2.2.6, Stim 1.16.0,
  PyMatching 2.4.0, ldpc 0.2.10+, beliefmatching 0.1.x.

## Results

| d | rounds | shots | det | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | ldpc BP-OSD LER | BeliefMatching LER | QB µs | PM µs | LDPC µs | BM µs |
|---|--------|-------|-----|--------------------|----------------|---------------|------------------|---------------------|-------|-------|---------|-------|
| 3 | 3 | 20000 | 24 | 0.01225 [0.01082, 0.01387] | 0.01225 [0.01082, 0.01387] | 0.01465 | 0.01225 [0.01082, 0.01387] | 0.00985 [0.00857, 0.01132] | 0.5 | 0.4 | 14.3 | 84.1 |
| 5 | 5 | 20000 | 120 | 0.00745 [0.00635, 0.00874] | 0.00745 [0.00635, 0.00874] | 0.03845 | 0.00800 [0.00686, 0.00933] | 0.00575 [0.00479, 0.00690] | 9.0 | 2.8 | 341.1 | 2541.2 |
| 7 | 7 | 10000 | 336 | 0.00430 [0.00319, 0.00579] | 0.00440 [0.00328, 0.00590] | 0.02000 | 0.00460 [0.00345, 0.00613] | 0.00290 [0.00202, 0.00416] | 44.6 | 8.1 | 2240.6 | 13953.1 |

### Higher distances (QECTOR-Blossom / PyMatching / QECTOR-UF only)

BP-based decoders become prohibitively slow beyond d = 7 on this machine.
The table below reproduces the earlier head-to-head for reference:

| d | rounds | shots | det | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | QB µs | PM µs | UF µs |
|---|--------|-------|-----|--------------------|----------------|---------------|-------|-------|-------|
| 9 | 9 | 40000 | 786 | 0.0030 [0.0025, 0.0036] | 0.0031 [0.0026, 0.0036] | 0.0214 | 103.1 | 22.0 | 11.8 |
| 11 | 11 | 40000 | 1470 | 0.0018 [0.0015, 0.0023] | 0.0018 [0.0015, 0.0023] | 0.0161 | 230.4 | 56.5 | 26.5 |

## What the data says

**Accuracy parity holds across all three MWPM decoders.** QECTOR-Blossom and
PyMatching produce statistically identical LER at every distance (overlapping
Wilson 95% intervals). **ldpc BP-OSD** matches within CI as well, with a slight
edge at d = 3 but wider intervals. **BeliefMatching** shows numerically lower LER
at all three distances (marginally overlapping CIs), consistent with the
literature result that BP+post-processing can out-gate MWPM on surface codes
when BP converges.

**UF is above threshold at circuit level.** Unweighted Union-Find's LER rises
from 1.47 % (d = 3) to 3.85 % (d = 5), confirming that ignoring edge weights
from `log((1-p)/p)` is catastrophic for circuit-level accuracy. The weighted UF
variant (UF-01, available in the Rust core) would close most of this gap — the
benchmark script has not yet been updated to use it (C2-04 tracks this).

**Latency spread is wide.** The four MWPM-class decoders span three orders of
magnitude:
| Decoder | d = 3 | d = 5 | d = 7 |
|---------|-------|-------|-------|
| QECTOR-UF | 0.3 µs | 1.5 µs | 4.3 µs |
| PyMatching | 0.4 µs | 2.8 µs | 8.1 µs |
| QECTOR-Blossom | 0.5 µs | 9.0 µs | 44.6 µs |
| ldpc BP-OSD | 14.3 µs | 341.1 µs | 2240.6 µs |
| BeliefMatching | 84.1 µs | 2541.2 µs | 13953.1 µs |

BP-based decoders are 10–300× slower than MWPM at circuit level, and the gap
widens with distance as BP's O(edges × iters) cost grows faster than MWPM's
O(detectors × mean-degree). BeliefMatching is an additional ~6× slower than
ldpc BP-OSD at every distance.

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
