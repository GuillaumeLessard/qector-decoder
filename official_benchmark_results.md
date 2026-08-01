# QECTOR v0.7.0 — circuit-level decoder comparison

Every row below is one `ler.estimate_ler_circuit_level` measurement: the same Stim rotated-surface-code circuit, the same decomposed DEM, the same detector/observable samples and the same `decode_batch` resolver for every decoder, scored against the circuit's own logical observables. `ler.assert_comparable` gated these rows before writing.

- **Noise model**: circuit-level, p = 0.005 (gate, reset and measurement noise over d rounds of syndrome extraction)
- **Per-cell decode budget**: 25s. Cells projected to exceed it appear under *Not measured* — nothing is extrapolated.
- **Git commit**: `07966e4da5` (tree dirty: True)
- **Platform**: Windows-10-10.0.26100-SP0 · Python 3.11.9
- **Versions**: qector 0.7.0, pymatching 2.4.0, ldpc ?, stim 1.16.0


## Measured results (185 cells)

| Decoder | d | Shots | Errors | LER | 95% CI | Throughput (dec/s) | vs PyMatching |
|:---|:---:|---:|---:|---:|:---:|---:|---:|
| PyMatching v2 (C++) | 3 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 2,076,843.5 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 1,219,066.3 | 0.59x |
| QECTOR CUDA Batch (GPU, weighted) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 1,587,553.5 | 0.76x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 441,267.3 | 0.21x |
| QECTOR OpenCL Batch (GPU, weighted) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 358,808.8 | 0.17x |
| QECTOR Sparse Blossom (CPU) | 3 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 728,597.5 | 0.35x |
| QECTOR Union-Find (CPU) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 887,862.9 | 0.43x |
| ldpc BP-OSD | 3 | 1,000 | 12 | 0.01200 | [0.00688, 0.02086] | 2,434.5 | 0.001x |
| PyMatching v2 (C++) | 3 | 5,000 | 84 | 0.01680 | [0.01359, 0.02075] | 2,371,241.6 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 1,670,564.7 | 0.70x |
| QECTOR CUDA Batch (GPU, weighted) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 1,369,637.9 | 0.58x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 1,518,187.9 | 0.64x |
| QECTOR OpenCL Batch (GPU, weighted) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 1,189,343.5 | 0.50x |
| QECTOR Sparse Blossom (CPU) | 3 | 5,000 | 84 | 0.01680 | [0.01359, 0.02075] | 1,491,335.3 | 0.63x |
| QECTOR Union-Find (CPU) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 2,458,693.9 | 1.04x |
| ldpc BP-OSD | 3 | 5,000 | 87 | 0.01740 | [0.01413, 0.02141] | 2,359.6 | 0.001x |
| PyMatching v2 (C++) | 3 | 10,000 | 202 | 0.02020 | [0.01762, 0.02315] | 2,436,825.3 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 10,000 | 235 | 0.02350 | [0.02071, 0.02666] | 1,802,808.8 | 0.74x |
| QECTOR CUDA Batch (GPU, weighted) | 3 | 10,000 | 238 | 0.02380 | [0.02099, 0.02698] | 1,408,867.4 | 0.58x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 10,000 | 235 | 0.02350 | [0.02071, 0.02666] | 1,602,127.6 | 0.66x |
| QECTOR OpenCL Batch (GPU, weighted) | 3 | 10,000 | 238 | 0.02380 | [0.02099, 0.02698] | 1,432,008.3 | 0.59x |
| QECTOR Sparse Blossom (CPU) | 3 | 10,000 | 202 | 0.02020 | [0.01762, 0.02315] | 1,616,553.5 | 0.66x |
| QECTOR Union-Find (CPU) | 3 | 10,000 | 233 | 0.02330 | [0.02052, 0.02644] | 2,809,225.4 | 1.15x |
| ldpc BP-OSD | 3 | 10,000 | 199 | 0.01990 | [0.01734, 0.02283] | 2,371.1 | 0.001x |
| PyMatching v2 (C++) | 3 | 50,000 | 960 | 0.01920 | [0.01803, 0.02044] | 2,501,288.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 50,000 | 1131 | 0.02262 | [0.02135, 0.02396] | 1,363,174.1 | 0.55x |
| QECTOR CUDA Batch (GPU, weighted) | 3 | 50,000 | 1129 | 0.02258 | [0.02131, 0.02392] | 1,206,188.2 | 0.48x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 50,000 | 1131 | 0.02262 | [0.02135, 0.02396] | 1,281,886.9 | 0.51x |
| QECTOR OpenCL Batch (GPU, weighted) | 3 | 50,000 | 1129 | 0.02258 | [0.02131, 0.02392] | 1,328,900.1 | 0.53x |
| QECTOR Sparse Blossom (CPU) | 3 | 50,000 | 960 | 0.01920 | [0.01803, 0.02044] | 2,174,518.1 | 0.87x |
| QECTOR Union-Find (CPU) | 3 | 50,000 | 1128 | 0.02256 | [0.02129, 0.02390] | 4,510,274.4 | 1.80x |
| ldpc BP-OSD | 3 | 50,000 | 969 | 0.01938 | [0.01821, 0.02063] | 2,355.5 | 0.001x |
| PyMatching v2 (C++) | 3 | 100,000 | 1891 | 0.01891 | [0.01808, 0.01977] | 2,488,546.5 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 100,000 | 2215 | 0.02215 | [0.02126, 0.02308] | 1,391,074.0 | 0.56x |
| QECTOR CUDA Batch (GPU, weighted) | 3 | 100,000 | 2201 | 0.02201 | [0.02112, 0.02294] | 1,252,127.1 | 0.50x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 100,000 | 2215 | 0.02215 | [0.02126, 0.02308] | 1,186,666.1 | 0.48x |
| QECTOR OpenCL Batch (GPU, weighted) | 3 | 100,000 | 2201 | 0.02201 | [0.02112, 0.02294] | 1,494,096.1 | 0.60x |
| QECTOR Sparse Blossom (CPU) | 3 | 100,000 | 1891 | 0.01891 | [0.01808, 0.01977] | 2,259,953.4 | 0.91x |
| QECTOR Union-Find (CPU) | 3 | 100,000 | 2210 | 0.02210 | [0.02121, 0.02303] | 4,816,445.3 | 1.94x |
| PyMatching v2 (C++) | 5 | 1,000 | 19 | 0.01900 | [0.01220, 0.02948] | 273,687.7 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 1,000 | 60 | 0.06000 | [0.04690, 0.07647] | 86,111.9 | 0.32x |
| QECTOR CUDA Batch (GPU, weighted) | 5 | 1,000 | 31 | 0.03100 | [0.02192, 0.04367] | 36,620.4 | 0.13x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 1,000 | 60 | 0.06000 | [0.04690, 0.07647] | 123,236.2 | 0.45x |
| QECTOR OpenCL Batch (GPU, weighted) | 5 | 1,000 | 31 | 0.03100 | [0.02192, 0.04367] | 28,726.4 | 0.10x |
| QECTOR Sparse Blossom (CPU) | 5 | 1,000 | 19 | 0.01900 | [0.01220, 0.02948] | 97,116.6 | 0.35x |
| QECTOR Union-Find (CPU) | 5 | 1,000 | 26 | 0.02600 | [0.01780, 0.03782] | 319,335.8 | 1.17x |
| ldpc BP-OSD | 5 | 1,000 | 21 | 0.02100 | [0.01378, 0.03189] | 102.7 | 0x |
| PyMatching v2 (C++) | 5 | 5,000 | 73 | 0.01460 | [0.01163, 0.01832] | 298,064.4 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 5,000 | 336 | 0.06720 | [0.06059, 0.07448] | 233,925.8 | 0.79x |
| QECTOR CUDA Batch (GPU, weighted) | 5 | 5,000 | 181 | 0.03620 | [0.03137, 0.04174] | 106,413.3 | 0.36x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 5,000 | 336 | 0.06720 | [0.06059, 0.07448] | 211,317.3 | 0.71x |
| QECTOR OpenCL Batch (GPU, weighted) | 5 | 5,000 | 181 | 0.03620 | [0.03137, 0.04174] | 72,852.0 | 0.24x |
| QECTOR Sparse Blossom (CPU) | 5 | 5,000 | 73 | 0.01460 | [0.01163, 0.01832] | 103,809.6 | 0.35x |
| QECTOR Union-Find (CPU) | 5 | 5,000 | 148 | 0.02960 | [0.02525, 0.03467] | 526,642.9 | 1.77x |
| PyMatching v2 (C++) | 5 | 10,000 | 166 | 0.01660 | [0.01428, 0.01930] | 302,400.8 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 10,000 | 600 | 0.06000 | [0.05551, 0.06483] | 244,247.4 | 0.81x |
| QECTOR CUDA Batch (GPU, weighted) | 5 | 10,000 | 316 | 0.03160 | [0.02835, 0.03521] | 90,121.8 | 0.30x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 10,000 | 600 | 0.06000 | [0.05551, 0.06483] | 185,941.7 | 0.61x |
| QECTOR OpenCL Batch (GPU, weighted) | 5 | 10,000 | 316 | 0.03160 | [0.02835, 0.03521] | 58,503.5 | 0.19x |
| QECTOR Sparse Blossom (CPU) | 5 | 10,000 | 166 | 0.01660 | [0.01428, 0.01930] | 113,170.7 | 0.37x |
| QECTOR Union-Find (CPU) | 5 | 10,000 | 268 | 0.02680 | [0.02381, 0.03015] | 529,234.9 | 1.75x |
| PyMatching v2 (C++) | 5 | 50,000 | 809 | 0.01618 | [0.01511, 0.01732] | 305,101.9 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 50,000 | 3041 | 0.06082 | [0.05876, 0.06295] | 151,108.2 | 0.49x |
| QECTOR CUDA Batch (GPU, weighted) | 5 | 50,000 | 1561 | 0.03122 | [0.02973, 0.03278] | 68,196.0 | 0.22x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 50,000 | 3041 | 0.06082 | [0.05876, 0.06295] | 137,817.5 | 0.45x |
| QECTOR OpenCL Batch (GPU, weighted) | 5 | 50,000 | 1561 | 0.03122 | [0.02973, 0.03278] | 49,724.2 | 0.16x |
| QECTOR Sparse Blossom (CPU) | 5 | 50,000 | 810 | 0.01620 | [0.01513, 0.01734] | 101,140.7 | 0.33x |
| QECTOR Union-Find (CPU) | 5 | 50,000 | 1356 | 0.02712 | [0.02573, 0.02858] | 585,514.4 | 1.92x |
| PyMatching v2 (C++) | 5 | 100,000 | 1596 | 0.01596 | [0.01520, 0.01676] | 276,642.6 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 100,000 | 6094 | 0.06094 | [0.05947, 0.06244] | 157,099.4 | 0.57x |
| QECTOR CUDA Batch (GPU, weighted) | 5 | 100,000 | 3182 | 0.03182 | [0.03075, 0.03293] | 70,473.1 | 0.26x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 100,000 | 6094 | 0.06094 | [0.05947, 0.06244] | 138,965.1 | 0.50x |
| QECTOR OpenCL Batch (GPU, weighted) | 5 | 100,000 | 3182 | 0.03182 | [0.03075, 0.03293] | 51,241.1 | 0.18x |
| QECTOR Sparse Blossom (CPU) | 5 | 100,000 | 1596 | 0.01596 | [0.01520, 0.01676] | 102,728.0 | 0.37x |
| QECTOR Union-Find (CPU) | 5 | 100,000 | 2645 | 0.02645 | [0.02547, 0.02746] | 630,492.8 | 2.28x |
| PyMatching v2 (C++) | 7 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 90,950.4 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 20,547.8 | 0.23x |
| QECTOR CUDA Batch (GPU, weighted) | 7 | 1,000 | 25 | 0.02500 | [0.01699, 0.03665] | 4,149.1 | 0.05x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 40,396.5 | 0.44x |
| QECTOR OpenCL Batch (GPU, weighted) | 7 | 1,000 | 25 | 0.02500 | [0.01699, 0.03665] | 4,691.7 | 0.05x |
| QECTOR Sparse Blossom (CPU) | 7 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 17,369.7 | 0.19x |
| QECTOR Union-Find (CPU) | 7 | 1,000 | 18 | 0.01800 | [0.01142, 0.02827] | 70,580.6 | 0.78x |
| PyMatching v2 (C++) | 7 | 5,000 | 74 | 0.01480 | [0.01181, 0.01854] | 97,446.1 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 5,000 | 218 | 0.04360 | [0.03828, 0.04962] | 57,195.5 | 0.59x |
| QECTOR CUDA Batch (GPU, weighted) | 7 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 14,833.4 | 0.15x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 5,000 | 218 | 0.04360 | [0.03828, 0.04962] | 57,889.7 | 0.59x |
| QECTOR OpenCL Batch (GPU, weighted) | 7 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 9,637.5 | 0.10x |
| QECTOR Sparse Blossom (CPU) | 7 | 5,000 | 75 | 0.01500 | [0.01198, 0.01876] | 17,735.2 | 0.18x |
| QECTOR Union-Find (CPU) | 7 | 5,000 | 107 | 0.02140 | [0.01774, 0.02579] | 142,736.8 | 1.47x |
| PyMatching v2 (C++) | 7 | 10,000 | 130 | 0.01300 | [0.01096, 0.01541] | 100,601.6 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 10,000 | 409 | 0.04090 | [0.03719, 0.04496] | 56,318.9 | 0.56x |
| QECTOR CUDA Batch (GPU, weighted) | 7 | 10,000 | 180 | 0.01800 | [0.01557, 0.02080] | 12,679.6 | 0.13x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 10,000 | 409 | 0.04090 | [0.03719, 0.04496] | 50,969.7 | 0.51x |
| QECTOR OpenCL Batch (GPU, weighted) | 7 | 10,000 | 180 | 0.01800 | [0.01557, 0.02080] | 8,063.1 | 0.08x |
| QECTOR Sparse Blossom (CPU) | 7 | 10,000 | 133 | 0.01330 | [0.01123, 0.01574] | 17,978.8 | 0.18x |
| QECTOR Union-Find (CPU) | 7 | 10,000 | 209 | 0.02090 | [0.01827, 0.02389] | 133,876.3 | 1.33x |
| PyMatching v2 (C++) | 7 | 50,000 | 619 | 0.01238 | [0.01145, 0.01339] | 100,970.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 50,000 | 2197 | 0.04394 | [0.04218, 0.04577] | 40,030.9 | 0.40x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 50,000 | 2197 | 0.04394 | [0.04218, 0.04577] | 223,596.7 | 2.21x |
| QECTOR Sparse Blossom (CPU) | 7 | 50,000 | 624 | 0.01248 | [0.01154, 0.01349] | 17,993.9 | 0.18x |
| QECTOR Union-Find (CPU) | 7 | 50,000 | 1055 | 0.02110 | [0.01988, 0.02240] | 150,366.7 | 1.49x |
| PyMatching v2 (C++) | 7 | 100,000 | 1220 | 0.01220 | [0.01154, 0.01290] | 101,498.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 100,000 | 4274 | 0.04274 | [0.04150, 0.04401] | 42,491.7 | 0.42x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 100,000 | 4274 | 0.04274 | [0.04150, 0.04401] | 237,312.3 | 2.34x |
| QECTOR Sparse Blossom (CPU) | 7 | 100,000 | 1233 | 0.01233 | [0.01166, 0.01303] | 17,017.7 | 0.17x |
| QECTOR Union-Find (CPU) | 7 | 100,000 | 2042 | 0.02042 | [0.01956, 0.02132] | 152,985.1 | 1.51x |
| PyMatching v2 (C++) | 9 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 35,063.5 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 1,000 | 56 | 0.05600 | [0.04337, 0.07202] | 8,592.1 | 0.24x |
| QECTOR CUDA Batch (GPU, weighted) | 9 | 1,000 | 17 | 0.01700 | [0.01064, 0.02706] | 923.5 | 0.03x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 1,000 | 56 | 0.05600 | [0.04337, 0.07202] | 15,118.6 | 0.43x |
| QECTOR OpenCL Batch (GPU, weighted) | 9 | 1,000 | 17 | 0.01700 | [0.01064, 0.02706] | 1,000.0 | 0.03x |
| QECTOR Sparse Blossom (CPU) | 9 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 2,321.9 | 0.07x |
| QECTOR Union-Find (CPU) | 9 | 1,000 | 14 | 0.01400 | [0.00836, 0.02336] | 17,950.1 | 0.51x |
| PyMatching v2 (C++) | 9 | 5,000 | 36 | 0.00720 | [0.00521, 0.00995] | 39,197.4 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 5,000 | 224 | 0.04480 | [0.03941, 0.05089] | 28,983.8 | 0.74x |
| QECTOR CUDA Batch (GPU, weighted) | 9 | 5,000 | 57 | 0.01140 | [0.00881, 0.01474] | 3,765.1 | 0.10x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 5,000 | 224 | 0.04480 | [0.03941, 0.05089] | 22,434.3 | 0.57x |
| QECTOR OpenCL Batch (GPU, weighted) | 9 | 5,000 | 57 | 0.01140 | [0.00881, 0.01474] | 1,983.8 | 0.05x |
| QECTOR Sparse Blossom (CPU) | 9 | 5,000 | 36 | 0.00720 | [0.00521, 0.00995] | 2,568.9 | 0.07x |
| QECTOR Union-Find (CPU) | 9 | 5,000 | 85 | 0.01700 | [0.01377, 0.02097] | 44,132.5 | 1.13x |
| PyMatching v2 (C++) | 9 | 10,000 | 78 | 0.00780 | [0.00625, 0.00972] | 40,425.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 10,000 | 431 | 0.04310 | [0.03929, 0.04726] | 24,069.6 | 0.59x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 10,000 | 431 | 0.04310 | [0.03929, 0.04726] | 18,874.5 | 0.47x |
| QECTOR Sparse Blossom (CPU) | 9 | 10,000 | 79 | 0.00790 | [0.00634, 0.00983] | 2,569.4 | 0.06x |
| QECTOR Union-Find (CPU) | 9 | 10,000 | 156 | 0.01560 | [0.01335, 0.01822] | 37,363.1 | 0.92x |
| PyMatching v2 (C++) | 9 | 50,000 | 450 | 0.00900 | [0.00821, 0.00987] | 39,995.1 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 50,000 | 2324 | 0.04648 | [0.04467, 0.04836] | 15,441.5 | 0.39x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 50,000 | 2324 | 0.04648 | [0.04467, 0.04836] | 101,861.5 | 2.55x |
| QECTOR Sparse Blossom (CPU) | 9 | 50,000 | 453 | 0.00906 | [0.00827, 0.00993] | 2,342.5 | 0.06x |
| QECTOR Union-Find (CPU) | 9 | 50,000 | 827 | 0.01654 | [0.01546, 0.01770] | 40,492.1 | 1.01x |
| PyMatching v2 (C++) | 9 | 100,000 | 878 | 0.00878 | [0.00822, 0.00938] | 40,581.5 | 1.00x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 100,000 | 4663 | 0.04663 | [0.04534, 0.04795] | 101,976.3 | 2.51x |
| QECTOR Union-Find (CPU) | 9 | 100,000 | 1732 | 0.01732 | [0.01653, 0.01815] | 42,073.4 | 1.04x |
| PyMatching v2 (C++) | 11 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 18,583.6 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 5,091.3 | 0.27x |
| QECTOR CUDA Batch (GPU, weighted) | 11 | 1,000 | 10 | 0.01000 | [0.00544, 0.01831] | 276.9 | 0.01x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 7,159.7 | 0.39x |
| QECTOR OpenCL Batch (GPU, weighted) | 11 | 1,000 | 10 | 0.01000 | [0.00544, 0.01831] | 285.8 | 0.01x |
| QECTOR Sparse Blossom (CPU) | 11 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 700.7 | 0.04x |
| QECTOR Union-Find (CPU) | 11 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 4,544.3 | 0.24x |
| PyMatching v2 (C++) | 11 | 5,000 | 24 | 0.00480 | [0.00323, 0.00713] | 20,328.9 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 5,000 | 229 | 0.04580 | [0.04035, 0.05195] | 14,372.8 | 0.71x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 5,000 | 229 | 0.04580 | [0.04035, 0.05195] | 10,940.6 | 0.54x |
| QECTOR Sparse Blossom (CPU) | 11 | 5,000 | 24 | 0.00480 | [0.00323, 0.00713] | 720.6 | 0.04x |
| QECTOR Union-Find (CPU) | 11 | 5,000 | 71 | 0.01420 | [0.01127, 0.01787] | 8,186.0 | 0.40x |
| PyMatching v2 (C++) | 11 | 10,000 | 77 | 0.00770 | [0.00617, 0.00961] | 19,444.1 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 10,000 | 440 | 0.04400 | [0.04015, 0.04820] | 11,861.8 | 0.61x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 10,000 | 440 | 0.04400 | [0.04015, 0.04820] | 9,328.2 | 0.48x |
| QECTOR Sparse Blossom (CPU) | 11 | 10,000 | 78 | 0.00780 | [0.00625, 0.00972] | 757.0 | 0.04x |
| QECTOR Union-Find (CPU) | 11 | 10,000 | 172 | 0.01720 | [0.01483, 0.01994] | 8,019.1 | 0.41x |
| PyMatching v2 (C++) | 11 | 50,000 | 334 | 0.00668 | [0.00600, 0.00743] | 18,836.4 | 1.00x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 50,000 | 2111 | 0.04222 | [0.04049, 0.04402] | 50,184.4 | 2.66x |
| QECTOR Union-Find (CPU) | 11 | 50,000 | 839 | 0.01678 | [0.01569, 0.01794] | 8,681.4 | 0.46x |
| PyMatching v2 (C++) | 11 | 100,000 | 647 | 0.00647 | [0.00599, 0.00699] | 19,880.1 | 1.00x |
| PyMatching v2 (C++) | 13 | 1,000 | 3 | 0.00300 | [0.00102, 0.00878] | 7,678.8 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 1,000 | 39 | 0.03900 | [0.02866, 0.05287] | 2,747.3 | 0.36x |
| QECTOR CUDA Batch (GPU, weighted) | 13 | 1,000 | 8 | 0.00800 | [0.00406, 0.01571] | 123.4 | 0.02x |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 1,000 | 39 | 0.03900 | [0.02866, 0.05287] | 4,002.0 | 0.52x |
| QECTOR OpenCL Batch (GPU, weighted) | 13 | 1,000 | 8 | 0.00800 | [0.00406, 0.01571] | 137.4 | 0.02x |
| QECTOR Sparse Blossom (CPU) | 13 | 1,000 | 3 | 0.00300 | [0.00102, 0.00878] | 219.7 | 0.03x |
| QECTOR Union-Find (CPU) | 13 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 1,065.6 | 0.14x |
| PyMatching v2 (C++) | 13 | 5,000 | 18 | 0.00360 | [0.00228, 0.00568] | 9,294.9 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 5,000 | 201 | 0.04020 | [0.03510, 0.04601] | 7,927.1 | 0.85x |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 5,000 | 201 | 0.04020 | [0.03510, 0.04601] | 6,133.6 | 0.66x |
| QECTOR Sparse Blossom (CPU) | 13 | 5,000 | 19 | 0.00380 | [0.00243, 0.00593] | 226.6 | 0.02x |
| QECTOR Union-Find (CPU) | 13 | 5,000 | 98 | 0.01960 | [0.01611, 0.02383] | 1,623.3 | 0.17x |
| PyMatching v2 (C++) | 13 | 10,000 | 37 | 0.00370 | [0.00269, 0.00510] | 7,786.5 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 10,000 | 410 | 0.04100 | [0.03729, 0.04507] | 6,599.7 | 0.85x |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 10,000 | 410 | 0.04100 | [0.03729, 0.04507] | 24,690.3 | 3.17x |
| QECTOR Union-Find (CPU) | 13 | 10,000 | 140 | 0.01400 | [0.01188, 0.01650] | 2,378.7 | 0.30x |
| PyMatching v2 (C++) | 13 | 50,000 | 221 | 0.00442 | [0.00388, 0.00504] | 7,768.7 | 1.00x |
| PyMatching v2 (C++) | 13 | 100,000 | 445 | 0.00445 | [0.00406, 0.00488] | 10,008.5 | 1.00x |
| PyMatching v2 (C++) | 15 | 1,000 | 2 | 0.00200 | [0.00055, 0.00726] | 5,999.4 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 1,000 | 41 | 0.04100 | [0.03036, 0.05515] | 1,654.0 | 0.28x |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 1,000 | 41 | 0.04100 | [0.03036, 0.05515] | 2,411.9 | 0.40x |
| QECTOR Sparse Blossom (CPU) | 15 | 1,000 | 2 | 0.00200 | [0.00055, 0.00726] | 59.1 | 0.01x |
| QECTOR Union-Find (CPU) | 15 | 1,000 | 12 | 0.01200 | [0.00688, 0.02086] | 316.9 | 0.05x |
| PyMatching v2 (C++) | 15 | 5,000 | 12 | 0.00240 | [0.00137, 0.00419] | 5,784.1 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 5,000 | 170 | 0.03400 | [0.02932, 0.03939] | 4,779.8 | 0.83x |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 5,000 | 170 | 0.03400 | [0.02932, 0.03939] | 12,703.5 | 2.20x |
| PyMatching v2 (C++) | 15 | 10,000 | 44 | 0.00440 | [0.00328, 0.00590] | 6,862.9 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 10,000 | 376 | 0.03760 | [0.03405, 0.04151] | 3,971.2 | 0.58x |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 10,000 | 376 | 0.03760 | [0.03405, 0.04151] | 11,531.4 | 1.68x |
| PyMatching v2 (C++) | 15 | 50,000 | 171 | 0.00342 | [0.00295, 0.00397] | 6,773.2 | 1.00x |
| PyMatching v2 (C++) | 15 | 100,000 | 314 | 0.00314 | [0.00281, 0.00351] | 6,720.9 | 1.00x |

## Not measured (95 cells)

These cells were **not run** and carry no numbers. They are listed so the gaps in the grid are explicit rather than quietly filled in.

| Decoder | d | Shots | Reason | Probe rate (dec/s) | Projected decode |
|:---|:---:|---:|:---|---:|---:|
| ldpc BP-OSD | 3 | 100,000 | over per-cell decode budget | 2,505.6 | 40s |
| ldpc BP-OSD | 5 | 5,000 | over per-cell decode budget | 104.1 | 48s |
| ldpc BP-OSD | 5 | 10,000 | over per-cell decode budget | 104.1 | 96s |
| ldpc BP-OSD | 5 | 50,000 | over per-cell decode budget | 104.1 | 480s |
| ldpc BP-OSD | 5 | 100,000 | over per-cell decode budget | 104.1 | 960s |
| QECTOR CUDA Batch (GPU, weighted) | 7 | 50,000 | over per-cell decode budget | 1,191.3 | 42s |
| QECTOR CUDA Batch (GPU, weighted) | 7 | 100,000 | over per-cell decode budget | 1,191.3 | 84s |
| QECTOR OpenCL Batch (GPU, weighted) | 7 | 50,000 | over per-cell decode budget | 912.8 | 55s |
| QECTOR OpenCL Batch (GPU, weighted) | 7 | 100,000 | over per-cell decode budget | 912.8 | 110s |
| ldpc BP-OSD | 7 | 1,000 | over per-cell decode budget | 27.8 | 36s |
| ldpc BP-OSD | 7 | 5,000 | over per-cell decode budget | 27.8 | 180s |
| ldpc BP-OSD | 7 | 10,000 | over per-cell decode budget | 27.8 | 359s |
| ldpc BP-OSD | 7 | 50,000 | over per-cell decode budget | 27.8 | 1,797s |
| ldpc BP-OSD | 7 | 100,000 | over per-cell decode budget | 27.8 | 3,594s |
| QECTOR Sparse Blossom (CPU) | 9 | 100,000 | over per-cell decode budget | 2,066.8 | 48s |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 100,000 | over per-cell decode budget | 2,589.2 | 39s |
| QECTOR CUDA Batch (GPU, weighted) | 9 | 10,000 | over per-cell decode budget | 330.9 | 30s |
| QECTOR CUDA Batch (GPU, weighted) | 9 | 50,000 | over per-cell decode budget | 330.9 | 151s |
| QECTOR CUDA Batch (GPU, weighted) | 9 | 100,000 | over per-cell decode budget | 330.9 | 302s |
| QECTOR OpenCL Batch (GPU, weighted) | 9 | 10,000 | over per-cell decode budget | 329.0 | 30s |
| QECTOR OpenCL Batch (GPU, weighted) | 9 | 50,000 | over per-cell decode budget | 329.0 | 152s |
| QECTOR OpenCL Batch (GPU, weighted) | 9 | 100,000 | over per-cell decode budget | 329.0 | 304s |
| ldpc BP-OSD | 9 | 1,000 | over per-cell decode budget | 10.1 | 99s |
| ldpc BP-OSD | 9 | 5,000 | over per-cell decode budget | 10.1 | 496s |
| ldpc BP-OSD | 9 | 10,000 | over per-cell decode budget | 10.1 | 993s |
| ldpc BP-OSD | 9 | 50,000 | over per-cell decode budget | 10.1 | 4,963s |
| ldpc BP-OSD | 9 | 100,000 | over per-cell decode budget | 10.1 | 9,926s |
| QECTOR Sparse Blossom (CPU) | 11 | 50,000 | over per-cell decode budget | 714.4 | 70s |
| QECTOR Sparse Blossom (CPU) | 11 | 100,000 | over per-cell decode budget | 714.4 | 140s |
| QECTOR Union-Find (CPU) | 11 | 100,000 | over per-cell decode budget | 2,446.9 | 41s |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 50,000 | over per-cell decode budget | 1,189.6 | 42s |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 100,000 | over per-cell decode budget | 1,189.6 | 84s |
| QECTOR CUDA Batch (GPU, weighted) | 11 | 5,000 | over per-cell decode budget | 102.1 | 49s |
| QECTOR CUDA Batch (GPU, weighted) | 11 | 10,000 | over per-cell decode budget | 102.1 | 98s |
| QECTOR CUDA Batch (GPU, weighted) | 11 | 50,000 | over per-cell decode budget | 102.1 | 490s |
| QECTOR CUDA Batch (GPU, weighted) | 11 | 100,000 | over per-cell decode budget | 102.1 | 980s |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 100,000 | over per-cell decode budget | 2,237.3 | 45s |
| QECTOR OpenCL Batch (GPU, weighted) | 11 | 5,000 | over per-cell decode budget | 106.2 | 47s |
| QECTOR OpenCL Batch (GPU, weighted) | 11 | 10,000 | over per-cell decode budget | 106.2 | 94s |
| QECTOR OpenCL Batch (GPU, weighted) | 11 | 50,000 | over per-cell decode budget | 106.2 | 471s |
| QECTOR OpenCL Batch (GPU, weighted) | 11 | 100,000 | over per-cell decode budget | 106.2 | 942s |
| ldpc BP-OSD | 11 | 1,000 | over per-cell decode budget | 4.8 | 210s |
| ldpc BP-OSD | 11 | 5,000 | over per-cell decode budget | 4.8 | 1,048s |
| ldpc BP-OSD | 11 | 10,000 | over per-cell decode budget | 4.8 | 2,096s |
| ldpc BP-OSD | 11 | 50,000 | over per-cell decode budget | 4.8 | 10,479s |
| ldpc BP-OSD | 11 | 100,000 | over per-cell decode budget | 4.8 | 20,957s |
| QECTOR Sparse Blossom (CPU) | 13 | 10,000 | over per-cell decode budget | 209.3 | 48s |
| QECTOR Sparse Blossom (CPU) | 13 | 50,000 | over per-cell decode budget | 209.3 | 239s |
| QECTOR Sparse Blossom (CPU) | 13 | 100,000 | over per-cell decode budget | 209.3 | 478s |
| QECTOR Union-Find (CPU) | 13 | 50,000 | over per-cell decode budget | 638.6 | 78s |
| QECTOR Union-Find (CPU) | 13 | 100,000 | over per-cell decode budget | 638.6 | 157s |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 50,000 | over per-cell decode budget | 740.3 | 68s |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 100,000 | over per-cell decode budget | 740.3 | 135s |
| QECTOR CUDA Batch (GPU, weighted) | 13 | 5,000 | over per-cell decode budget | 49.6 | 101s |
| QECTOR CUDA Batch (GPU, weighted) | 13 | 10,000 | over per-cell decode budget | 49.6 | 202s |
| QECTOR CUDA Batch (GPU, weighted) | 13 | 50,000 | over per-cell decode budget | 49.6 | 1,009s |
| QECTOR CUDA Batch (GPU, weighted) | 13 | 100,000 | over per-cell decode budget | 49.6 | 2,017s |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 50,000 | over per-cell decode budget | 1,228.4 | 41s |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 100,000 | over per-cell decode budget | 1,228.4 | 81s |
| QECTOR OpenCL Batch (GPU, weighted) | 13 | 5,000 | over per-cell decode budget | 50.7 | 99s |
| QECTOR OpenCL Batch (GPU, weighted) | 13 | 10,000 | over per-cell decode budget | 50.7 | 197s |
| QECTOR OpenCL Batch (GPU, weighted) | 13 | 50,000 | over per-cell decode budget | 50.7 | 986s |
| QECTOR OpenCL Batch (GPU, weighted) | 13 | 100,000 | over per-cell decode budget | 50.7 | 1,973s |
| ldpc BP-OSD | 13 | 1,000 | over per-cell decode budget | 2.8 | 362s |
| ldpc BP-OSD | 13 | 5,000 | over per-cell decode budget | 2.8 | 1,808s |
| ldpc BP-OSD | 13 | 10,000 | over per-cell decode budget | 2.8 | 3,616s |
| ldpc BP-OSD | 13 | 50,000 | over per-cell decode budget | 2.8 | 18,080s |
| ldpc BP-OSD | 13 | 100,000 | over per-cell decode budget | 2.8 | 36,160s |
| QECTOR Sparse Blossom (CPU) | 15 | 5,000 | over per-cell decode budget | 56.1 | 89s |
| QECTOR Sparse Blossom (CPU) | 15 | 10,000 | over per-cell decode budget | 56.1 | 178s |
| QECTOR Sparse Blossom (CPU) | 15 | 50,000 | over per-cell decode budget | 56.1 | 891s |
| QECTOR Sparse Blossom (CPU) | 15 | 100,000 | over per-cell decode budget | 56.1 | 1,782s |
| QECTOR Union-Find (CPU) | 15 | 5,000 | over per-cell decode budget | 191.1 | 26s |
| QECTOR Union-Find (CPU) | 15 | 10,000 | over per-cell decode budget | 191.1 | 52s |
| QECTOR Union-Find (CPU) | 15 | 50,000 | over per-cell decode budget | 191.1 | 262s |
| QECTOR Union-Find (CPU) | 15 | 100,000 | over per-cell decode budget | 191.1 | 523s |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 50,000 | over per-cell decode budget | 437.7 | 114s |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 100,000 | over per-cell decode budget | 437.7 | 228s |
| QECTOR CUDA Batch (GPU, weighted) | 15 | 1,000 | over per-cell decode budget | 22.5 | 44s |
| QECTOR CUDA Batch (GPU, weighted) | 15 | 5,000 | over per-cell decode budget | 22.5 | 222s |
| QECTOR CUDA Batch (GPU, weighted) | 15 | 10,000 | over per-cell decode budget | 22.5 | 445s |
| QECTOR CUDA Batch (GPU, weighted) | 15 | 50,000 | over per-cell decode budget | 22.5 | 2,223s |
| QECTOR CUDA Batch (GPU, weighted) | 15 | 100,000 | over per-cell decode budget | 22.5 | 4,446s |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 50,000 | over per-cell decode budget | 703.5 | 71s |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 100,000 | over per-cell decode budget | 703.5 | 142s |
| QECTOR OpenCL Batch (GPU, weighted) | 15 | 1,000 | over per-cell decode budget | 29.4 | 34s |
| QECTOR OpenCL Batch (GPU, weighted) | 15 | 5,000 | over per-cell decode budget | 29.4 | 170s |
| QECTOR OpenCL Batch (GPU, weighted) | 15 | 10,000 | over per-cell decode budget | 29.4 | 340s |
| QECTOR OpenCL Batch (GPU, weighted) | 15 | 50,000 | over per-cell decode budget | 29.4 | 1,701s |
| QECTOR OpenCL Batch (GPU, weighted) | 15 | 100,000 | over per-cell decode budget | 29.4 | 3,403s |
| ldpc BP-OSD | 15 | 1,000 | over per-cell decode budget | 1.8 | 557s |
| ldpc BP-OSD | 15 | 5,000 | over per-cell decode budget | 1.8 | 2,783s |
| ldpc BP-OSD | 15 | 10,000 | over per-cell decode budget | 1.8 | 5,566s |
| ldpc BP-OSD | 15 | 50,000 | over per-cell decode budget | 1.8 | 27,830s |
| ldpc BP-OSD | 15 | 100,000 | over per-cell decode budget | 1.8 | 55,660s |

## How to read this

- Throughput figures are only meaningful on an otherwise-idle machine.
- LER figures are subject to binomial error; at low p and low shot counts the confidence interval can exceed the difference between decoders. Check ci95_lo/ci95_hi per row.
- Accuracy and speed are independent axes: a lower LER at the same (d, p) is more accurate, a higher throughput is faster. This table deliberately does not collapse them into one score.
- Throughput counts decode time only — `LerResult.seconds` wraps the single `decode_batch` call. Circuit construction and sampling are excluded for every decoder equally.
- A cell with 0 errors is not evidence of a zero error rate; read its `ci95_hi` as an upper bound. The LER chart plots those as open downward markers.
