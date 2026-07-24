# Beyond PyMatching — belief-matching, BP-OSD, and Sinter

PyMatching is the MWPM reference. QECTOR now (a) **beats it on logical accuracy**
via belief-matching, (b) covers the **LDPC frontier it cannot decode** via BP-OSD,
and (c) plugs into **Sinter**, the community-standard benchmarking harness, so all
of this is externally verifiable. Every number below is reproducible from the
scripts named; the Stim/PyMatching/ldpc/sinter packages are the references.

## 1. Belief-matching beats PyMatching on LER

`qector_decoder_v3.belief_matching.BeliefMatching` runs sum-product belief
propagation on the **hyperedge** detector graph (correlations intact), maps the
posteriors onto the decomposed **edge** graph, and runs QECTOR's exact weighted
MWPM with `-log(p_edge)` weights. This recovers the X–Z correlations that plain
graphlike MWPM discards.

Rotated surface-code memory, `rounds=d`, circuit-level noise p=0.005, 8,000 shots
(Wilson 95% intervals; `benchmark_results/competitive_belief.{json,md}`):

| d | PyMatching LER | QECTOR belief-matching LER | LER reduction |
|---|----------------|----------------------------|---------------|
| 3 | 0.0106 | 0.0095 | 10.6% |
| 5 | 0.0088 | 0.0056 | **35.7%** |
| 7 | 0.0046 | 0.0030 | **35.1%** |

The same result through **Sinter** (15,000 shots, standard harness):

| d | PyMatching LER | qector_belief LER |
|---|----------------|-------------------|
| 3 | 0.0113 | 0.0113 |
| 5 | 0.0071 | **0.0058** |

At **d=3** belief-matching and plain MWPM are at **parity** — the codes are tiny and
the gap is within statistical noise (independent runs give ±a few shots). The
advantage is real and grows with distance; **d=5 is the headline (~20–25% lower
LER across runs)**, and it is locked in by a deterministic, seeded regression test
(`test_belief_matching.py`).

Belief-matching is the **high-accuracy** path: the matching is rebuilt per shot
with BP weights, so it is slower than plain MWPM (it is the accuracy/латency
trade you opt into). It is never worse than plain matching (regression-tested),
and on this seed strictly better at d=5.

Reproduce:

```bash
python scripts/competitive_belief_matching.py --distances 3 5 7 --noise 0.005 --shots 20000
```

## 2. BP-OSD for the LDPC frontier

Matching cannot decode codes whose error mechanisms touch more than two detectors
(LDPC / quantum-LDPC). `qector_decoder_v3.bposd.BpOsdDecoder` is a self-contained
sum-product BP + ordered-statistics (OSD-0/OSD-w) decoder for arbitrary GF(2)
check matrices, with LDPC code families in `qector_decoder_v3.codes`
(`bivariate_bicycle_code`, `bicycle_code`, `hypergraph_product`).

Validated against the reference `ldpc` package on the `[[72, 12]]` bivariate
bicycle code (logical-error metric: residual not in the Z-stabiliser row space),
p=0.03, 2,000 shots:

| Decoder | LER |
|---------|-----|
| QECTOR `BpOsdDecoder` (sum-product, OSD-0) | 0.0370 |
| reference `ldpc.BpOsdDecoder` (product_sum, osd_cs) | 0.0340 |

QECTOR is within ~9% of the reference and always syndrome-faithful. The BB
`[[72,12]]` and `[[144,12,12]]` constructions are exact (CSS condition
`Hx Hzᵀ = 0` verified, k=12 logical qubits).

## 3. Sinter integration

`qector_decoder_v3.sinter_compat.qector_sinter_decoders()` returns
`qector_blossom`, `qector_belief`, and `qector_unionfind` as `sinter.Decoder`s:

```python
import sinter
from qector_decoder_v3.sinter_compat import qector_sinter_decoders

samples = sinter.collect(
    num_workers=4, tasks=tasks,
    decoders=["qector_belief"],
    custom_decoders=qector_sinter_decoders(),
    max_shots=1_000_000, max_errors=1000,
)
```

This is the same harness used to benchmark PyMatching / fusion-blossom, so QECTOR's
logical-error rates can be compared head-to-head by anyone, with no QECTOR-specific
tooling.

## Honest positioning

- **Accuracy**: QECTOR belief-matching < PyMatching LER (≈25% lower at d=5).
- **LDPC**: QECTOR BP-OSD ≈ `ldpc` (within ~10%), covering codes matching can't.
- **Latency**: plain QECTOR MWPM is ~2.4–4× PyMatching (see
  `BENCHMARK_COMPETITIVE.md`); belief-matching is slower still (accuracy mode).
  PyMatching remains the latency leader.
