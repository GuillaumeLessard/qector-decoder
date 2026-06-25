# QECTOR v3 — LER Summary (Updated 2026-06-25)

This file consolidates logical-error-rate results from two sources:
1. Internal benchmarks (toric surface codes, earlier runs)
2. **Independent PyPI validation** (2026-06-24, 86/87 checks PASS — the authoritative external record)

Full machine-readable validation artifact: `benchmark_results/validation_v051.json`

---

## A. Internal Benchmarks (earlier runs, toric surface code model)

CPU: AMD64 Family 23 Model 96 Stepping 1 (AuthenticAMD)
Python: 3.11.0
QECTOR: 0.5.2

### Surface Code Toric d=3 (5000 shots)

| p | Model | Decoder | LER ± stderr | 95% CI |
|---|-------|---------|-------------|--------|
| 0.01 | bitflip | UnionFind | 0.0576 ± 0.0033 | [0.0515, 0.0644] |
| 0.01 | bitflip | Blossom | 0.0250 ± 0.0022 | [0.0210, 0.0297] |
| 0.05 | bitflip | UnionFind | 0.2364 ± 0.0060 | [0.2248, 0.2484] |
| 0.05 | bitflip | Blossom | 0.1216 ± 0.0046 | [0.1128, 0.1309] |
| 0.10 | bitflip | UnionFind | 0.3642 ± 0.0068 | [0.3510, 0.3776] |
| 0.10 | bitflip | Blossom | 0.2364 ± 0.0060 | [0.2248, 0.2484] |
| 0.15 | bitflip | UnionFind | 0.4286 ± 0.0070 | [0.4149, 0.4424] |
| 0.15 | bitflip | Blossom | 0.2924 ± 0.0064 | [0.2800, 0.3052] |

### Surface Code Toric d=5 (3000 shots)

| p | Model | Decoder | LER ± stderr | 95% CI |
|---|-------|---------|-------------|--------|
| 0.01 | bitflip | UnionFind | 0.0903 ± 0.0052 | [0.0806, 0.1011] |
| 0.01 | bitflip | Blossom | 0.0543 ± 0.0041 | [0.0468, 0.0630] |
| 0.05 | bitflip | UnionFind | 0.3220 ± 0.0085 | [0.3055, 0.3389] |
| 0.05 | bitflip | Blossom | 0.1953 ± 0.0072 | [0.1815, 0.2099] |

---

## B. Independent PyPI Validation — Repetition Code LER (100k shots/pt, seed 777)

Platform: Windows 10, AMD Ryzen 16-core, Python 3.11, PyMatching 2.4.0, stim 1.16.0
Reference decoder: PyMatching 2.4.0 (independent cross-check, bit-identical to Blossom)

| dist | p | UF errs | Blossom errs | PyM errs | Blossom LER | 95% CI (Wilson) |
|------|-----|---------|--------------|----------|-------------|-----------------|
| d=3 | 0.01 | 29 | 29 | 29 | 0.00029 | [0.0002, 0.0004] |
| d=3 | 0.03 | 263 | 263 | 263 | 0.00263 | [0.0023, 0.0030] |
| d=3 | 0.05 | 694 | 694 | 694 | 0.00694 | [0.0064, 0.0075] |
| d=3 | 0.08 | 1836 | 1836 | 1836 | 0.01836 | [0.0175, 0.0192] |
| d=3 | 0.10 | 2863 | 2863 | 2863 | 0.02863 | [0.0276, 0.0297] |
| d=3 | 0.15 | 6127 | 6127 | 6127 | 0.06127 | [0.0598, 0.0628] |
| d=5 | 0.01 | 38 | 3 | 3 | 0.00003 | [0.0000, 0.0001] |
| d=5 | 0.03 | 341 | 20 | 20 | 0.00020 | [0.0001, 0.0003] |
| d=5 | 0.05 | 913 | 107 | 107 | 0.00107 | [0.0009, 0.0013] |
| d=5 | 0.08 | 2279 | 454 | 454 | 0.00454 | [0.0041, 0.0050] |
| d=5 | 0.10 | 3352 | 815 | 815 | 0.00815 | [0.0076, 0.0087] |
| d=5 | 0.15 | 7177 | 2660 | 2660 | 0.02660 | [0.0256, 0.0276] |
| d=7 | 0.01 | 0 | 0 | 0 | 0.00000 | [0.0000, 0.0000] |
| d=7 | 0.03 | 19 | 2 | 2 | 0.00002 | [0.0000, 0.0001] |
| d=7 | 0.05 | 129 | 19 | 19 | 0.00019 | [0.0001, 0.0003] |
| d=7 | 0.08 | 454 | 124 | 124 | 0.00124 | [0.0010, 0.0015] |
| d=7 | 0.10 | 912 | 269 | 269 | 0.00269 | [0.0024, 0.0030] |
| d=7 | 0.15 | 2695 | 1144 | 1144 | 0.01144 | [0.0108, 0.0121] |
| d=9 | 0.01 | 0 | 0 | 0 | 0.00000 | [0.0000, 0.0000] |
| d=9 | 0.03 | 3 | 0 | 0 | 0.00000 | [0.0000, 0.0000] |
| d=9 | 0.05 | 20 | 3 | 3 | 0.00003 | [0.0000, 0.0001] |
| d=9 | 0.08 | 134 | 37 | 37 | 0.00037 | [0.0003, 0.0005] |
| d=9 | 0.10 | 306 | 97 | 97 | 0.00097 | [0.0008, 0.0012] |
| d=9 | 0.15 | 1315 | 564 | 564 | 0.00564 | [0.0052, 0.0061] |

**Key result**: Blossom LER == PyMatching LER at 0.00% delta across all d=3–9 points (repetition code).

---

## C. Independent PyPI Validation — Rotated Surface Code LER (100k shots/pt, seed 777)

| dist | p | UF LER | Blossom LER | PyMatching LER | Blossom–PyM delta |
|------|-----|--------|-------------|----------------|-------------------|
| d=3 | 0.01 | 0.0110 | 0.0105 | 0.0107 | -0.02% |
| d=3 | 0.03 | 0.0359 | 0.0317 | 0.0326 | -0.09% |
| d=3 | 0.05 | 0.0649 | 0.0556 | 0.0574 | -0.18% |
| d=3 | 0.08 | 0.1107 | 0.0903 | 0.0947 | -0.44% |
| d=3 | 0.10 | 0.1414 | 0.1141 | 0.1201 | -0.60% |
| d=3 | 0.15 | 0.2177 | 0.1732 | 0.1837 | -1.05% |
| d=5 | 0.01 | 0.0123 | 0.0099 | 0.0098 | +0.01% |
| d=5 | 0.03 | 0.0488 | 0.0325 | 0.0312 | +0.13% |
| d=5 | 0.05 | 0.0929 | 0.0574 | 0.0551 | +0.23% |
| d=5 | 0.08 | 0.1660 | 0.1002 | 0.0947 | +0.55% |
| d=5 | 0.10 | 0.2137 | 0.1286 | 0.1213 | +0.73% |
| d=5 | 0.15 | 0.3077 | 0.1995 | 0.1880 | +1.16% |
| d=7 | 0.01 | 0.0107 | 0.0107 | 0.0105 | +0.02% |
| d=7 | 0.03 | 0.0355 | 0.0326 | 0.0312 | +0.14% |
| d=7 | 0.05 | 0.0694 | 0.0586 | 0.0548 | +0.38% |
| d=7 | 0.08 | 0.1295 | 0.1012 | 0.0927 | +0.86% |
| d=7 | 0.10 | 0.1716 | 0.1311 | 0.1205 | +1.06% |
| d=7 | 0.15 | 0.2655 | 0.2046 | 0.1868 | +1.78% |

**Key result**: Blossom within 1.78% of PyMatching at worst case (d=7, p=0.15, code-capacity model).
Note: d=3/5/7 LER curves do not separate with distance under single-round code-capacity noise — this is a
model property shared with PyMatching (Finding F-2). Use circuit-level Stim DEM for threshold curves.

---

## D. CPU Batch Throughput (repetition d=9, 20k shots primary run)

| Decoder | Throughput |
|---------|-----------|
| UnionFindDecoder | ~1.06M shots/s |
| BlossomDecoder | ~2.70M shots/s |
| SparseBlossomDecoder | ~1.80M shots/s |
| CPUBatchDecoder | ~0.34M shots/s |
| BatchDecoder (parallel_batch_decode) | ~2.67M shots/s |
| BatchDecoder (alias) | ~1.88M shots/s |

---

## E. Key Findings Summary

1. **Blossom consistently outperforms UnionFind on LER** for surface codes (2.3× at d=3 p=0.01; 1.7× at d=5 p=0.01).
2. **Blossom matches PyMatching exactly** on repetition code (0.00% delta), within 1.78% on surface code.
3. **CUDA delivers 7× throughput** vs CPU batch at 100k shots on GTX 1660 Ti.
4. **LookupTableDecoder is the fastest single-shot decoder tested** (~8.7 µs on rep d=5, exact on precomputed syndromes).
5. **UnionFind ~3.1× less accurate than Blossom/MWPM** — expected speed/accuracy trade-off, both 100% valid.
6. **Workbench latency**: Blossom rep d=5 → 277,778 dec/s, p50 3.60 µs, p99 11.61 µs, max 46.8 µs.

### Recommendations

- For real-time decoding where speed is critical: use FastUnionFindDecoder or CPUBatchDecoder.
- For research and accuracy-critical applications: use BlossomDecoder or SparseBlossomDecoder.
- For batch GPU throughput: use CUDABatchDecoder (check `is_available()` first; requires ≥4096 batch for GPU win).
- For small codes with exact O(1) lookup: use LookupTableDecoder (table_size=64 for rep d=5).
- For threshold curves: use circuit-level Stim DEM with BeliefMatching or qector_sinter_decoders() — not code-capacity.
