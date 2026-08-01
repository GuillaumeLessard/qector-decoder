# QECTOR v0.7.0 — circuit-level decoder comparison

Every row below is one `ler.estimate_ler_circuit_level` measurement: the same Stim rotated-surface-code circuit, the same decomposed DEM, the same detector/observable samples and the same `decode_batch` resolver for every decoder, scored against the circuit's own logical observables. `ler.assert_comparable` gated these rows before writing.

- **Noise model**: circuit-level, p = 0.005 (gate, reset and measurement noise over d rounds of syndrome extraction)
- **Per-cell decode budget**: 25s. Cells projected to exceed it appear under *Not measured* — nothing is extrapolated.
- **Git commit**: `b436f04e3b` (tree dirty: True)
- **Platform**: Windows-10-10.0.26100-SP0 · Python 3.11.9
- **Versions**: qector 0.7.0, pymatching 2.4.0, ldpc ?, stim 1.16.0


## Measured results (187 cells)

| Decoder | d | Shots | Errors | LER | 95% CI | Throughput (dec/s) | vs PyMatching |
|:---|:---:|---:|---:|---:|:---:|---:|---:|
| PyMatching v2 (C++) | 3 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 2,059,308.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 1,243,935.9 | 0.60x |
| QECTOR CUDA Batch (GPU, weighted) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 1,014,816.4 | 0.49x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 477,623.3 | 0.23x |
| QECTOR OpenCL Batch (GPU, weighted) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 323,300.2 | 0.16x |
| QECTOR Sparse Blossom (CPU) | 3 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 706,563.9 | 0.34x |
| QECTOR Union-Find (CPU) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 1,077,934.7 | 0.52x |
| ldpc BP-OSD | 3 | 1,000 | 12 | 0.01200 | [0.00688, 0.02086] | 2,411.4 | 0.001x |
| PyMatching v2 (C++) | 3 | 5,000 | 84 | 0.01680 | [0.01359, 0.02075] | 2,387,090.6 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 1,364,442.6 | 0.57x |
| QECTOR CUDA Batch (GPU, weighted) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 1,288,892.3 | 0.54x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 1,409,125.5 | 0.59x |
| QECTOR OpenCL Batch (GPU, weighted) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 1,242,977.2 | 0.52x |
| QECTOR Sparse Blossom (CPU) | 3 | 5,000 | 84 | 0.01680 | [0.01359, 0.02075] | 1,526,298.1 | 0.64x |
| QECTOR Union-Find (CPU) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 2,156,938.9 | 0.90x |
| ldpc BP-OSD | 3 | 5,000 | 87 | 0.01740 | [0.01413, 0.02141] | 2,320.1 | 0.001x |
| PyMatching v2 (C++) | 3 | 10,000 | 202 | 0.02020 | [0.01762, 0.02315] | 2,414,176.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 10,000 | 235 | 0.02350 | [0.02071, 0.02666] | 2,027,369.5 | 0.84x |
| QECTOR CUDA Batch (GPU, weighted) | 3 | 10,000 | 238 | 0.02380 | [0.02099, 0.02698] | 1,444,961.4 | 0.60x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 10,000 | 235 | 0.02350 | [0.02071, 0.02666] | 1,681,774.6 | 0.70x |
| QECTOR OpenCL Batch (GPU, weighted) | 3 | 10,000 | 238 | 0.02380 | [0.02099, 0.02698] | 1,311,750.7 | 0.54x |
| QECTOR Sparse Blossom (CPU) | 3 | 10,000 | 202 | 0.02020 | [0.01762, 0.02315] | 1,918,686.1 | 0.80x |
| QECTOR Union-Find (CPU) | 3 | 10,000 | 233 | 0.02330 | [0.02052, 0.02644] | 3,129,596.6 | 1.30x |
| ldpc BP-OSD | 3 | 10,000 | 199 | 0.01990 | [0.01734, 0.02283] | 2,354.7 | 0.001x |
| PyMatching v2 (C++) | 3 | 50,000 | 960 | 0.01920 | [0.01803, 0.02044] | 2,483,336.8 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 50,000 | 1131 | 0.02262 | [0.02135, 0.02396] | 982,559.6 | 0.40x |
| QECTOR CUDA Batch (GPU, weighted) | 3 | 50,000 | 1129 | 0.02258 | [0.02131, 0.02392] | 1,158,896.3 | 0.47x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 50,000 | 1131 | 0.02262 | [0.02135, 0.02396] | 1,192,998.5 | 0.48x |
| QECTOR OpenCL Batch (GPU, weighted) | 3 | 50,000 | 1129 | 0.02258 | [0.02131, 0.02392] | 1,340,209.5 | 0.54x |
| QECTOR Sparse Blossom (CPU) | 3 | 50,000 | 960 | 0.01920 | [0.01803, 0.02044] | 2,285,098.0 | 0.92x |
| QECTOR Union-Find (CPU) | 3 | 50,000 | 1128 | 0.02256 | [0.02129, 0.02390] | 4,478,521.0 | 1.80x |
| ldpc BP-OSD | 3 | 50,000 | 969 | 0.01938 | [0.01821, 0.02063] | 2,399.3 | 0.001x |
| PyMatching v2 (C++) | 3 | 100,000 | 1891 | 0.01891 | [0.01808, 0.01977] | 2,478,781.6 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 100,000 | 2215 | 0.02215 | [0.02126, 0.02308] | 1,446,123.2 | 0.58x |
| QECTOR CUDA Batch (GPU, weighted) | 3 | 100,000 | 2201 | 0.02201 | [0.02112, 0.02294] | 1,201,287.8 | 0.48x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 100,000 | 2215 | 0.02215 | [0.02126, 0.02308] | 1,286,158.7 | 0.52x |
| QECTOR OpenCL Batch (GPU, weighted) | 3 | 100,000 | 2201 | 0.02201 | [0.02112, 0.02294] | 1,554,013.6 | 0.63x |
| QECTOR Sparse Blossom (CPU) | 3 | 100,000 | 1891 | 0.01891 | [0.01808, 0.01977] | 2,337,448.0 | 0.94x |
| QECTOR Union-Find (CPU) | 3 | 100,000 | 2210 | 0.02210 | [0.02121, 0.02303] | 4,953,928.5 | 2.00x |
| PyMatching v2 (C++) | 5 | 1,000 | 19 | 0.01900 | [0.01220, 0.02948] | 295,203.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 1,000 | 60 | 0.06000 | [0.04690, 0.07647] | 86,729.5 | 0.29x |
| QECTOR CUDA Batch (GPU, weighted) | 5 | 1,000 | 31 | 0.03100 | [0.02192, 0.04367] | 36,514.6 | 0.12x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 1,000 | 60 | 0.06000 | [0.04690, 0.07647] | 127,962.3 | 0.43x |
| QECTOR OpenCL Batch (GPU, weighted) | 5 | 1,000 | 31 | 0.03100 | [0.02192, 0.04367] | 36,462.3 | 0.12x |
| QECTOR Sparse Blossom (CPU) | 5 | 1,000 | 19 | 0.01900 | [0.01220, 0.02948] | 89,467.0 | 0.30x |
| QECTOR Union-Find (CPU) | 5 | 1,000 | 26 | 0.02600 | [0.01780, 0.03782] | 320,625.9 | 1.09x |
| ldpc BP-OSD | 5 | 1,000 | 21 | 0.02100 | [0.01378, 0.03189] | 112.8 | 0x |
| PyMatching v2 (C++) | 5 | 5,000 | 73 | 0.01460 | [0.01163, 0.01832] | 322,584.8 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 5,000 | 336 | 0.06720 | [0.06059, 0.07448] | 215,810.3 | 0.67x |
| QECTOR CUDA Batch (GPU, weighted) | 5 | 5,000 | 181 | 0.03620 | [0.03137, 0.04174] | 106,718.8 | 0.33x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 5,000 | 336 | 0.06720 | [0.06059, 0.07448] | 235,303.0 | 0.73x |
| QECTOR OpenCL Batch (GPU, weighted) | 5 | 5,000 | 181 | 0.03620 | [0.03137, 0.04174] | 82,005.7 | 0.25x |
| QECTOR Sparse Blossom (CPU) | 5 | 5,000 | 73 | 0.01460 | [0.01163, 0.01832] | 106,067.0 | 0.33x |
| QECTOR Union-Find (CPU) | 5 | 5,000 | 148 | 0.02960 | [0.02525, 0.03467] | 536,814.8 | 1.66x |
| PyMatching v2 (C++) | 5 | 10,000 | 166 | 0.01660 | [0.01428, 0.01930] | 319,171.7 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 10,000 | 600 | 0.06000 | [0.05551, 0.06483] | 240,079.3 | 0.75x |
| QECTOR CUDA Batch (GPU, weighted) | 5 | 10,000 | 316 | 0.03160 | [0.02835, 0.03521] | 90,402.9 | 0.28x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 10,000 | 600 | 0.06000 | [0.05551, 0.06483] | 179,482.0 | 0.56x |
| QECTOR OpenCL Batch (GPU, weighted) | 5 | 10,000 | 316 | 0.03160 | [0.02835, 0.03521] | 66,580.1 | 0.21x |
| QECTOR Sparse Blossom (CPU) | 5 | 10,000 | 166 | 0.01660 | [0.01428, 0.01930] | 112,330.6 | 0.35x |
| QECTOR Union-Find (CPU) | 5 | 10,000 | 268 | 0.02680 | [0.02381, 0.03015] | 590,576.8 | 1.85x |
| PyMatching v2 (C++) | 5 | 50,000 | 809 | 0.01618 | [0.01511, 0.01732] | 339,589.9 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 50,000 | 3041 | 0.06082 | [0.05876, 0.06295] | 152,165.5 | 0.45x |
| QECTOR CUDA Batch (GPU, weighted) | 5 | 50,000 | 1561 | 0.03122 | [0.02973, 0.03278] | 67,282.4 | 0.20x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 50,000 | 3041 | 0.06082 | [0.05876, 0.06295] | 140,526.1 | 0.41x |
| QECTOR OpenCL Batch (GPU, weighted) | 5 | 50,000 | 1561 | 0.03122 | [0.02973, 0.03278] | 60,052.2 | 0.18x |
| QECTOR Sparse Blossom (CPU) | 5 | 50,000 | 810 | 0.01620 | [0.01513, 0.01734] | 105,863.7 | 0.31x |
| QECTOR Union-Find (CPU) | 5 | 50,000 | 1356 | 0.02712 | [0.02573, 0.02858] | 672,617.1 | 1.98x |
| PyMatching v2 (C++) | 5 | 100,000 | 1596 | 0.01596 | [0.01520, 0.01676] | 340,202.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 100,000 | 6094 | 0.06094 | [0.05947, 0.06244] | 150,505.2 | 0.44x |
| QECTOR CUDA Batch (GPU, weighted) | 5 | 100,000 | 3182 | 0.03182 | [0.03075, 0.03293] | 69,156.1 | 0.20x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 100,000 | 6094 | 0.06094 | [0.05947, 0.06244] | 144,886.3 | 0.43x |
| QECTOR OpenCL Batch (GPU, weighted) | 5 | 100,000 | 3182 | 0.03182 | [0.03075, 0.03293] | 61,420.5 | 0.18x |
| QECTOR Sparse Blossom (CPU) | 5 | 100,000 | 1596 | 0.01596 | [0.01520, 0.01676] | 111,843.2 | 0.33x |
| QECTOR Union-Find (CPU) | 5 | 100,000 | 2645 | 0.02645 | [0.02547, 0.02746] | 694,191.8 | 2.04x |
| PyMatching v2 (C++) | 7 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 87,829.5 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 24,776.0 | 0.28x |
| QECTOR CUDA Batch (GPU, weighted) | 7 | 1,000 | 25 | 0.02500 | [0.01699, 0.03665] | 4,379.4 | 0.05x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 41,416.4 | 0.47x |
| QECTOR OpenCL Batch (GPU, weighted) | 7 | 1,000 | 25 | 0.02500 | [0.01699, 0.03665] | 5,076.0 | 0.06x |
| QECTOR Sparse Blossom (CPU) | 7 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 17,804.5 | 0.20x |
| QECTOR Union-Find (CPU) | 7 | 1,000 | 18 | 0.01800 | [0.01142, 0.02827] | 85,774.3 | 0.98x |
| PyMatching v2 (C++) | 7 | 5,000 | 74 | 0.01480 | [0.01181, 0.01854] | 95,176.6 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 5,000 | 218 | 0.04360 | [0.03828, 0.04962] | 68,090.1 | 0.71x |
| QECTOR CUDA Batch (GPU, weighted) | 7 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 16,141.2 | 0.17x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 5,000 | 218 | 0.04360 | [0.03828, 0.04962] | 60,767.4 | 0.64x |
| QECTOR OpenCL Batch (GPU, weighted) | 7 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 10,049.1 | 0.11x |
| QECTOR Sparse Blossom (CPU) | 7 | 5,000 | 75 | 0.01500 | [0.01198, 0.01876] | 20,658.1 | 0.22x |
| QECTOR Union-Find (CPU) | 7 | 5,000 | 107 | 0.02140 | [0.01774, 0.02579] | 147,940.4 | 1.55x |
| PyMatching v2 (C++) | 7 | 10,000 | 130 | 0.01300 | [0.01096, 0.01541] | 97,230.9 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 10,000 | 409 | 0.04090 | [0.03719, 0.04496] | 64,826.2 | 0.67x |
| QECTOR CUDA Batch (GPU, weighted) | 7 | 10,000 | 180 | 0.01800 | [0.01557, 0.02080] | 13,551.9 | 0.14x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 10,000 | 409 | 0.04090 | [0.03719, 0.04496] | 52,408.1 | 0.54x |
| QECTOR OpenCL Batch (GPU, weighted) | 7 | 10,000 | 180 | 0.01800 | [0.01557, 0.02080] | 8,692.5 | 0.09x |
| QECTOR Sparse Blossom (CPU) | 7 | 10,000 | 133 | 0.01330 | [0.01123, 0.01574] | 20,408.5 | 0.21x |
| QECTOR Union-Find (CPU) | 7 | 10,000 | 209 | 0.02090 | [0.01827, 0.02389] | 153,667.0 | 1.58x |
| PyMatching v2 (C++) | 7 | 50,000 | 619 | 0.01238 | [0.01145, 0.01339] | 96,788.6 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 50,000 | 2197 | 0.04394 | [0.04218, 0.04577] | 44,879.6 | 0.46x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 50,000 | 2197 | 0.04394 | [0.04218, 0.04577] | 236,796.6 | 2.45x |
| QECTOR Sparse Blossom (CPU) | 7 | 50,000 | 624 | 0.01248 | [0.01154, 0.01349] | 20,334.0 | 0.21x |
| QECTOR Union-Find (CPU) | 7 | 50,000 | 1055 | 0.02110 | [0.01988, 0.02240] | 164,348.0 | 1.70x |
| PyMatching v2 (C++) | 7 | 100,000 | 1220 | 0.01220 | [0.01154, 0.01290] | 97,940.8 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 100,000 | 4274 | 0.04274 | [0.04150, 0.04401] | 45,353.3 | 0.46x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 100,000 | 4274 | 0.04274 | [0.04150, 0.04401] | 244,318.4 | 2.50x |
| QECTOR Sparse Blossom (CPU) | 7 | 100,000 | 1233 | 0.01233 | [0.01166, 0.01303] | 20,975.4 | 0.21x |
| QECTOR Union-Find (CPU) | 7 | 100,000 | 2042 | 0.02042 | [0.01956, 0.02132] | 173,669.0 | 1.77x |
| PyMatching v2 (C++) | 9 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 38,464.6 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 1,000 | 56 | 0.05600 | [0.04337, 0.07202] | 10,066.7 | 0.26x |
| QECTOR CUDA Batch (GPU, weighted) | 9 | 1,000 | 17 | 0.01700 | [0.01064, 0.02706] | 1,120.4 | 0.03x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 1,000 | 56 | 0.05600 | [0.04337, 0.07202] | 17,223.2 | 0.45x |
| QECTOR OpenCL Batch (GPU, weighted) | 9 | 1,000 | 17 | 0.01700 | [0.01064, 0.02706] | 1,205.0 | 0.03x |
| QECTOR Sparse Blossom (CPU) | 9 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 2,696.0 | 0.07x |
| QECTOR Union-Find (CPU) | 9 | 1,000 | 14 | 0.01400 | [0.00836, 0.02336] | 17,940.3 | 0.47x |
| PyMatching v2 (C++) | 9 | 5,000 | 36 | 0.00720 | [0.00521, 0.00995] | 42,068.1 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 5,000 | 224 | 0.04480 | [0.03941, 0.05089] | 29,601.4 | 0.70x |
| QECTOR CUDA Batch (GPU, weighted) | 9 | 5,000 | 57 | 0.01140 | [0.00881, 0.01474] | 4,494.8 | 0.11x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 5,000 | 224 | 0.04480 | [0.03941, 0.05089] | 25,800.8 | 0.61x |
| QECTOR OpenCL Batch (GPU, weighted) | 9 | 5,000 | 57 | 0.01140 | [0.00881, 0.01474] | 2,371.3 | 0.06x |
| QECTOR Sparse Blossom (CPU) | 9 | 5,000 | 36 | 0.00720 | [0.00521, 0.00995] | 2,719.3 | 0.07x |
| QECTOR Union-Find (CPU) | 9 | 5,000 | 85 | 0.01700 | [0.01377, 0.02097] | 46,386.6 | 1.10x |
| PyMatching v2 (C++) | 9 | 10,000 | 78 | 0.00780 | [0.00625, 0.00972] | 42,397.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 10,000 | 431 | 0.04310 | [0.03929, 0.04726] | 27,672.0 | 0.65x |
| QECTOR CUDA Batch (GPU, weighted) | 9 | 10,000 | 140 | 0.01400 | [0.01188, 0.01650] | 3,381.9 | 0.08x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 10,000 | 431 | 0.04310 | [0.03929, 0.04726] | 21,899.4 | 0.52x |
| QECTOR OpenCL Batch (GPU, weighted) | 9 | 10,000 | 140 | 0.01400 | [0.01188, 0.01650] | 2,042.9 | 0.05x |
| QECTOR Sparse Blossom (CPU) | 9 | 10,000 | 79 | 0.00790 | [0.00634, 0.00983] | 2,606.2 | 0.06x |
| QECTOR Union-Find (CPU) | 9 | 10,000 | 156 | 0.01560 | [0.01335, 0.01822] | 41,347.3 | 0.97x |
| PyMatching v2 (C++) | 9 | 50,000 | 450 | 0.00900 | [0.00821, 0.00987] | 42,084.7 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 50,000 | 2324 | 0.04648 | [0.04467, 0.04836] | 18,272.6 | 0.43x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 50,000 | 2324 | 0.04648 | [0.04467, 0.04836] | 106,560.0 | 2.53x |
| QECTOR Sparse Blossom (CPU) | 9 | 50,000 | 453 | 0.00906 | [0.00827, 0.00993] | 2,649.0 | 0.06x |
| QECTOR Union-Find (CPU) | 9 | 50,000 | 827 | 0.01654 | [0.01546, 0.01770] | 43,803.1 | 1.04x |
| PyMatching v2 (C++) | 9 | 100,000 | 878 | 0.00878 | [0.00822, 0.00938] | 42,294.9 | 1.00x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 100,000 | 4663 | 0.04663 | [0.04534, 0.04795] | 107,528.9 | 2.54x |
| QECTOR Union-Find (CPU) | 9 | 100,000 | 1732 | 0.01732 | [0.01653, 0.01815] | 44,420.0 | 1.05x |
| PyMatching v2 (C++) | 11 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 19,696.1 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 5,679.7 | 0.29x |
| QECTOR CUDA Batch (GPU, weighted) | 11 | 1,000 | 10 | 0.01000 | [0.00544, 0.01831] | 330.1 | 0.02x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 7,614.2 | 0.39x |
| QECTOR OpenCL Batch (GPU, weighted) | 11 | 1,000 | 10 | 0.01000 | [0.00544, 0.01831] | 338.5 | 0.02x |
| QECTOR Sparse Blossom (CPU) | 11 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 734.4 | 0.04x |
| QECTOR Union-Find (CPU) | 11 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 3,708.8 | 0.19x |
| PyMatching v2 (C++) | 11 | 5,000 | 24 | 0.00480 | [0.00323, 0.00713] | 21,163.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 5,000 | 229 | 0.04580 | [0.04035, 0.05195] | 16,031.5 | 0.76x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 5,000 | 229 | 0.04580 | [0.04035, 0.05195] | 12,561.3 | 0.59x |
| QECTOR Sparse Blossom (CPU) | 11 | 5,000 | 24 | 0.00480 | [0.00323, 0.00713] | 856.6 | 0.04x |
| QECTOR Union-Find (CPU) | 11 | 5,000 | 71 | 0.01420 | [0.01127, 0.01787] | 8,764.9 | 0.41x |
| PyMatching v2 (C++) | 11 | 10,000 | 77 | 0.00770 | [0.00617, 0.00961] | 21,335.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 10,000 | 440 | 0.04400 | [0.04015, 0.04820] | 13,621.4 | 0.64x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 10,000 | 440 | 0.04400 | [0.04015, 0.04820] | 10,652.5 | 0.50x |
| QECTOR Sparse Blossom (CPU) | 11 | 10,000 | 78 | 0.00780 | [0.00625, 0.00972] | 889.6 | 0.04x |
| QECTOR Union-Find (CPU) | 11 | 10,000 | 172 | 0.01720 | [0.01483, 0.01994] | 8,294.2 | 0.39x |
| PyMatching v2 (C++) | 11 | 50,000 | 334 | 0.00668 | [0.00600, 0.00743] | 21,597.8 | 1.00x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 50,000 | 2111 | 0.04222 | [0.04049, 0.04402] | 53,218.4 | 2.46x |
| QECTOR Union-Find (CPU) | 11 | 50,000 | 839 | 0.01678 | [0.01569, 0.01794] | 9,722.0 | 0.45x |
| PyMatching v2 (C++) | 11 | 100,000 | 647 | 0.00647 | [0.00599, 0.00699] | 21,510.9 | 1.00x |
| PyMatching v2 (C++) | 13 | 1,000 | 3 | 0.00300 | [0.00102, 0.00878] | 10,351.3 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 1,000 | 39 | 0.03900 | [0.02866, 0.05287] | 3,172.4 | 0.31x |
| QECTOR CUDA Batch (GPU, weighted) | 13 | 1,000 | 8 | 0.00800 | [0.00406, 0.01571] | 151.4 | 0.01x |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 1,000 | 39 | 0.03900 | [0.02866, 0.05287] | 4,715.0 | 0.46x |
| QECTOR OpenCL Batch (GPU, weighted) | 13 | 1,000 | 8 | 0.00800 | [0.00406, 0.01571] | 166.3 | 0.02x |
| QECTOR Sparse Blossom (CPU) | 13 | 1,000 | 3 | 0.00300 | [0.00102, 0.00878] | 290.5 | 0.03x |
| QECTOR Union-Find (CPU) | 13 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 1,364.0 | 0.13x |
| PyMatching v2 (C++) | 13 | 5,000 | 18 | 0.00360 | [0.00228, 0.00568] | 11,752.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 5,000 | 201 | 0.04020 | [0.03510, 0.04601] | 9,104.2 | 0.78x |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 5,000 | 201 | 0.04020 | [0.03510, 0.04601] | 7,151.2 | 0.61x |
| QECTOR Sparse Blossom (CPU) | 13 | 5,000 | 19 | 0.00380 | [0.00243, 0.00593] | 292.8 | 0.03x |
| QECTOR Union-Find (CPU) | 13 | 5,000 | 98 | 0.01960 | [0.01611, 0.02383] | 1,786.2 | 0.15x |
| PyMatching v2 (C++) | 13 | 10,000 | 37 | 0.00370 | [0.00269, 0.00510] | 11,976.1 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 10,000 | 410 | 0.04100 | [0.03729, 0.04507] | 7,633.9 | 0.64x |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 10,000 | 410 | 0.04100 | [0.03729, 0.04507] | 25,937.0 | 2.17x |
| QECTOR Union-Find (CPU) | 13 | 10,000 | 140 | 0.01400 | [0.01188, 0.01650] | 2,570.9 | 0.21x |
| PyMatching v2 (C++) | 13 | 50,000 | 221 | 0.00442 | [0.00388, 0.00504] | 12,078.9 | 1.00x |
| PyMatching v2 (C++) | 13 | 100,000 | 445 | 0.00445 | [0.00406, 0.00488] | 12,096.1 | 1.00x |
| PyMatching v2 (C++) | 15 | 1,000 | 2 | 0.00200 | [0.00055, 0.00726] | 5,755.5 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 1,000 | 41 | 0.04100 | [0.03036, 0.05515] | 1,934.9 | 0.34x |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 1,000 | 41 | 0.04100 | [0.03036, 0.05515] | 2,488.1 | 0.43x |
| QECTOR Sparse Blossom (CPU) | 15 | 1,000 | 2 | 0.00200 | [0.00055, 0.00726] | 73.2 | 0.01x |
| QECTOR Union-Find (CPU) | 15 | 1,000 | 12 | 0.01200 | [0.00688, 0.02086] | 369.5 | 0.06x |
| PyMatching v2 (C++) | 15 | 5,000 | 12 | 0.00240 | [0.00137, 0.00419] | 5,161.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 5,000 | 170 | 0.03400 | [0.02932, 0.03939] | 5,276.2 | 1.02x |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 5,000 | 170 | 0.03400 | [0.02932, 0.03939] | 14,110.8 | 2.73x |
| QECTOR Union-Find (CPU) | 15 | 5,000 | 72 | 0.01440 | [0.01145, 0.01809] | 511.0 | 0.10x |
| PyMatching v2 (C++) | 15 | 10,000 | 44 | 0.00440 | [0.00328, 0.00590] | 5,465.1 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 10,000 | 376 | 0.03760 | [0.03405, 0.04151] | 4,397.7 | 0.81x |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 10,000 | 376 | 0.03760 | [0.03405, 0.04151] | 14,808.9 | 2.71x |
| PyMatching v2 (C++) | 15 | 50,000 | 171 | 0.00342 | [0.00295, 0.00397] | 4,846.7 | 1.00x |

## Not measured (93 cells)

These cells were **not run** and carry no numbers. They are listed so the gaps in the grid are explicit rather than quietly filled in.

| Decoder | d | Shots | Reason | Probe rate (dec/s) | Projected decode |
|:---|:---:|---:|:---|---:|---:|
| ldpc BP-OSD | 3 | 100,000 | over per-cell decode budget | 2,499.2 | 40s |
| ldpc BP-OSD | 5 | 5,000 | over per-cell decode budget | 109.5 | 46s |
| ldpc BP-OSD | 5 | 10,000 | over per-cell decode budget | 109.5 | 91s |
| ldpc BP-OSD | 5 | 50,000 | over per-cell decode budget | 109.5 | 456s |
| ldpc BP-OSD | 5 | 100,000 | over per-cell decode budget | 109.5 | 913s |
| QECTOR CUDA Batch (GPU, weighted) | 7 | 50,000 | over per-cell decode budget | 1,286.7 | 39s |
| QECTOR CUDA Batch (GPU, weighted) | 7 | 100,000 | over per-cell decode budget | 1,286.7 | 78s |
| QECTOR OpenCL Batch (GPU, weighted) | 7 | 50,000 | over per-cell decode budget | 1,078.7 | 46s |
| QECTOR OpenCL Batch (GPU, weighted) | 7 | 100,000 | over per-cell decode budget | 1,078.7 | 93s |
| ldpc BP-OSD | 7 | 1,000 | over per-cell decode budget | 28.3 | 35s |
| ldpc BP-OSD | 7 | 5,000 | over per-cell decode budget | 28.3 | 177s |
| ldpc BP-OSD | 7 | 10,000 | over per-cell decode budget | 28.3 | 353s |
| ldpc BP-OSD | 7 | 50,000 | over per-cell decode budget | 28.3 | 1,767s |
| ldpc BP-OSD | 7 | 100,000 | over per-cell decode budget | 28.3 | 3,534s |
| QECTOR Sparse Blossom (CPU) | 9 | 100,000 | over per-cell decode budget | 2,450.9 | 41s |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 100,000 | over per-cell decode budget | 3,311.4 | 30s |
| QECTOR CUDA Batch (GPU, weighted) | 9 | 50,000 | over per-cell decode budget | 414.8 | 121s |
| QECTOR CUDA Batch (GPU, weighted) | 9 | 100,000 | over per-cell decode budget | 414.8 | 241s |
| QECTOR OpenCL Batch (GPU, weighted) | 9 | 50,000 | over per-cell decode budget | 420.6 | 119s |
| QECTOR OpenCL Batch (GPU, weighted) | 9 | 100,000 | over per-cell decode budget | 420.6 | 238s |
| ldpc BP-OSD | 9 | 1,000 | over per-cell decode budget | 10.2 | 98s |
| ldpc BP-OSD | 9 | 5,000 | over per-cell decode budget | 10.2 | 492s |
| ldpc BP-OSD | 9 | 10,000 | over per-cell decode budget | 10.2 | 985s |
| ldpc BP-OSD | 9 | 50,000 | over per-cell decode budget | 10.2 | 4,923s |
| ldpc BP-OSD | 9 | 100,000 | over per-cell decode budget | 10.2 | 9,846s |
| QECTOR Sparse Blossom (CPU) | 11 | 50,000 | over per-cell decode budget | 694.2 | 72s |
| QECTOR Sparse Blossom (CPU) | 11 | 100,000 | over per-cell decode budget | 694.2 | 144s |
| QECTOR Union-Find (CPU) | 11 | 100,000 | over per-cell decode budget | 2,405.1 | 42s |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 50,000 | over per-cell decode budget | 1,462.9 | 34s |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 100,000 | over per-cell decode budget | 1,462.9 | 68s |
| QECTOR CUDA Batch (GPU, weighted) | 11 | 5,000 | over per-cell decode budget | 125.7 | 40s |
| QECTOR CUDA Batch (GPU, weighted) | 11 | 10,000 | over per-cell decode budget | 125.7 | 80s |
| QECTOR CUDA Batch (GPU, weighted) | 11 | 50,000 | over per-cell decode budget | 125.7 | 398s |
| QECTOR CUDA Batch (GPU, weighted) | 11 | 100,000 | over per-cell decode budget | 125.7 | 795s |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 100,000 | over per-cell decode budget | 2,645.0 | 38s |
| QECTOR OpenCL Batch (GPU, weighted) | 11 | 5,000 | over per-cell decode budget | 128.8 | 39s |
| QECTOR OpenCL Batch (GPU, weighted) | 11 | 10,000 | over per-cell decode budget | 128.8 | 78s |
| QECTOR OpenCL Batch (GPU, weighted) | 11 | 50,000 | over per-cell decode budget | 128.8 | 388s |
| QECTOR OpenCL Batch (GPU, weighted) | 11 | 100,000 | over per-cell decode budget | 128.8 | 777s |
| ldpc BP-OSD | 11 | 1,000 | over per-cell decode budget | 5.1 | 197s |
| ldpc BP-OSD | 11 | 5,000 | over per-cell decode budget | 5.1 | 983s |
| ldpc BP-OSD | 11 | 10,000 | over per-cell decode budget | 5.1 | 1,965s |
| ldpc BP-OSD | 11 | 50,000 | over per-cell decode budget | 5.1 | 9,827s |
| ldpc BP-OSD | 11 | 100,000 | over per-cell decode budget | 5.1 | 19,654s |
| QECTOR Sparse Blossom (CPU) | 13 | 10,000 | over per-cell decode budget | 254.9 | 39s |
| QECTOR Sparse Blossom (CPU) | 13 | 50,000 | over per-cell decode budget | 254.9 | 196s |
| QECTOR Sparse Blossom (CPU) | 13 | 100,000 | over per-cell decode budget | 254.9 | 392s |
| QECTOR Union-Find (CPU) | 13 | 50,000 | over per-cell decode budget | 761.9 | 66s |
| QECTOR Union-Find (CPU) | 13 | 100,000 | over per-cell decode budget | 761.9 | 131s |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 50,000 | over per-cell decode budget | 841.1 | 59s |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 100,000 | over per-cell decode budget | 841.1 | 119s |
| QECTOR CUDA Batch (GPU, weighted) | 13 | 5,000 | over per-cell decode budget | 61.3 | 82s |
| QECTOR CUDA Batch (GPU, weighted) | 13 | 10,000 | over per-cell decode budget | 61.3 | 163s |
| QECTOR CUDA Batch (GPU, weighted) | 13 | 50,000 | over per-cell decode budget | 61.3 | 816s |
| QECTOR CUDA Batch (GPU, weighted) | 13 | 100,000 | over per-cell decode budget | 61.3 | 1,632s |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 50,000 | over per-cell decode budget | 1,503.3 | 33s |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 100,000 | over per-cell decode budget | 1,503.3 | 67s |
| QECTOR OpenCL Batch (GPU, weighted) | 13 | 5,000 | over per-cell decode budget | 62.9 | 79s |
| QECTOR OpenCL Batch (GPU, weighted) | 13 | 10,000 | over per-cell decode budget | 62.9 | 159s |
| QECTOR OpenCL Batch (GPU, weighted) | 13 | 50,000 | over per-cell decode budget | 62.9 | 795s |
| QECTOR OpenCL Batch (GPU, weighted) | 13 | 100,000 | over per-cell decode budget | 62.9 | 1,589s |
| ldpc BP-OSD | 13 | 1,000 | over per-cell decode budget | 2.9 | 351s |
| ldpc BP-OSD | 13 | 5,000 | over per-cell decode budget | 2.9 | 1,753s |
| ldpc BP-OSD | 13 | 10,000 | over per-cell decode budget | 2.9 | 3,505s |
| ldpc BP-OSD | 13 | 50,000 | over per-cell decode budget | 2.9 | 17,526s |
| ldpc BP-OSD | 13 | 100,000 | over per-cell decode budget | 2.9 | 35,053s |
| QECTOR Sparse Blossom (CPU) | 15 | 5,000 | over per-cell decode budget | 67.8 | 74s |
| QECTOR Sparse Blossom (CPU) | 15 | 10,000 | over per-cell decode budget | 67.8 | 147s |
| QECTOR Sparse Blossom (CPU) | 15 | 50,000 | over per-cell decode budget | 67.8 | 737s |
| QECTOR Sparse Blossom (CPU) | 15 | 100,000 | over per-cell decode budget | 67.8 | 1,474s |
| QECTOR Union-Find (CPU) | 15 | 10,000 | over per-cell decode budget | 263.2 | 38s |
| QECTOR Union-Find (CPU) | 15 | 50,000 | over per-cell decode budget | 263.2 | 190s |
| QECTOR Union-Find (CPU) | 15 | 100,000 | over per-cell decode budget | 263.2 | 380s |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 50,000 | over per-cell decode budget | 494.0 | 101s |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 100,000 | over per-cell decode budget | 494.0 | 202s |
| QECTOR CUDA Batch (GPU, weighted) | 15 | 1,000 | over per-cell decode budget | 27.5 | 36s |
| QECTOR CUDA Batch (GPU, weighted) | 15 | 5,000 | over per-cell decode budget | 27.5 | 182s |
| QECTOR CUDA Batch (GPU, weighted) | 15 | 10,000 | over per-cell decode budget | 27.5 | 364s |
| QECTOR CUDA Batch (GPU, weighted) | 15 | 50,000 | over per-cell decode budget | 27.5 | 1,821s |
| QECTOR CUDA Batch (GPU, weighted) | 15 | 100,000 | over per-cell decode budget | 27.5 | 3,641s |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 50,000 | over per-cell decode budget | 833.9 | 60s |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 100,000 | over per-cell decode budget | 833.9 | 120s |
| QECTOR OpenCL Batch (GPU, weighted) | 15 | 1,000 | over per-cell decode budget | 28.0 | 36s |
| QECTOR OpenCL Batch (GPU, weighted) | 15 | 5,000 | over per-cell decode budget | 28.0 | 179s |
| QECTOR OpenCL Batch (GPU, weighted) | 15 | 10,000 | over per-cell decode budget | 28.0 | 357s |
| QECTOR OpenCL Batch (GPU, weighted) | 15 | 50,000 | over per-cell decode budget | 28.0 | 1,785s |
| QECTOR OpenCL Batch (GPU, weighted) | 15 | 100,000 | over per-cell decode budget | 28.0 | 3,571s |
| PyMatching v2 (C++) | 15 | 100,000 | over per-cell decode budget | 3,016.3 | 33s |
| ldpc BP-OSD | 15 | 1,000 | over per-cell decode budget | 1.8 | 564s |
| ldpc BP-OSD | 15 | 5,000 | over per-cell decode budget | 1.8 | 2,821s |
| ldpc BP-OSD | 15 | 10,000 | over per-cell decode budget | 1.8 | 5,641s |
| ldpc BP-OSD | 15 | 50,000 | over per-cell decode budget | 1.8 | 28,205s |
| ldpc BP-OSD | 15 | 100,000 | over per-cell decode budget | 1.8 | 56,411s |

## How to read this

- Throughput figures are only meaningful on an otherwise-idle machine.
- LER figures are subject to binomial error; at low p and low shot counts the confidence interval can exceed the difference between decoders. Check ci95_lo/ci95_hi per row.
- Accuracy and speed are independent axes: a lower LER at the same (d, p) is more accurate, a higher throughput is faster. This table deliberately does not collapse them into one score.
- Throughput counts decode time only — `LerResult.seconds` wraps the single `decode_batch` call. Circuit construction and sampling are excluded for every decoder equally.
- A cell with 0 errors is not evidence of a zero error rate; read its `ci95_hi` as an upper bound. The LER chart plots those as open downward markers.
