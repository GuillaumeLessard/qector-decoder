# QECTOR v0.7.0 — circuit-level decoder comparison

Every row below is one `ler.estimate_ler_circuit_level` measurement: the same Stim rotated-surface-code circuit, the same decomposed DEM, the same detector/observable samples and the same `decode_batch` resolver for every decoder, scored against the circuit's own logical observables. `ler.assert_comparable` gated these rows before writing.

- **Noise model**: circuit-level, p = 0.005 (gate, reset and measurement noise over d rounds of syndrome extraction)
- **Per-cell decode budget**: 25s. Cells projected to exceed it appear under *Not measured* — nothing is extrapolated.
- **Git commit**: `b057aaee34` (tree dirty: True)
- **Platform**: Windows-10-10.0.26100-SP0 · Python 3.11.9
- **Versions**: qector 0.7.0, pymatching 2.4.0, ldpc ?, stim 1.16.0


## Measured results (137 cells)

| Decoder | d | Shots | Errors | LER | 95% CI | Throughput (dec/s) | vs PyMatching |
|:---|:---:|---:|---:|---:|:---:|---:|---:|
| PyMatching v2 (C++) | 3 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 1,949,698.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 308,603.9 | 0.16x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 359,789.9 | 0.18x |
| QECTOR Sparse Blossom (CPU) | 3 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 250,645.4 | 0.13x |
| QECTOR Union-Find (CPU) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 382,614.0 | 0.20x |
| ldpc BP-OSD | 3 | 1,000 | 12 | 0.01200 | [0.00688, 0.02086] | 2,222.5 | 0.001x |
| PyMatching v2 (C++) | 3 | 5,000 | 84 | 0.01680 | [0.01359, 0.02075] | 2,394,865.3 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 1,572,772.2 | 0.66x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 1,311,647.5 | 0.55x |
| QECTOR Sparse Blossom (CPU) | 3 | 5,000 | 84 | 0.01680 | [0.01359, 0.02075] | 319,211.4 | 0.13x |
| QECTOR Union-Find (CPU) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 643,227.5 | 0.27x |
| ldpc BP-OSD | 3 | 5,000 | 87 | 0.01740 | [0.01413, 0.02141] | 2,344.9 | 0.001x |
| PyMatching v2 (C++) | 3 | 10,000 | 202 | 0.02020 | [0.01762, 0.02315] | 2,433,682.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 10,000 | 235 | 0.02350 | [0.02071, 0.02666] | 1,700,044.2 | 0.70x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 10,000 | 235 | 0.02350 | [0.02071, 0.02666] | 1,557,680.9 | 0.64x |
| QECTOR Sparse Blossom (CPU) | 3 | 10,000 | 202 | 0.02020 | [0.01762, 0.02315] | 339,989.2 | 0.14x |
| QECTOR Union-Find (CPU) | 3 | 10,000 | 233 | 0.02330 | [0.02052, 0.02644] | 835,491.7 | 0.34x |
| ldpc BP-OSD | 3 | 10,000 | 199 | 0.01990 | [0.01734, 0.02283] | 2,331.5 | 0.001x |
| PyMatching v2 (C++) | 3 | 50,000 | 960 | 0.01920 | [0.01803, 0.02044] | 2,513,232.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 50,000 | 1131 | 0.02262 | [0.02135, 0.02396] | 1,121,033.5 | 0.45x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 50,000 | 1131 | 0.02262 | [0.02135, 0.02396] | 1,239,695.0 | 0.49x |
| QECTOR Sparse Blossom (CPU) | 3 | 50,000 | 960 | 0.01920 | [0.01803, 0.02044] | 345,329.1 | 0.14x |
| QECTOR Union-Find (CPU) | 3 | 50,000 | 1128 | 0.02256 | [0.02129, 0.02390] | 1,188,436.0 | 0.47x |
| ldpc BP-OSD | 3 | 50,000 | 969 | 0.01938 | [0.01821, 0.02063] | 2,326.5 | 0.001x |
| PyMatching v2 (C++) | 3 | 100,000 | 1891 | 0.01891 | [0.01808, 0.01977] | 2,497,016.1 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 3 | 100,000 | 2215 | 0.02215 | [0.02126, 0.02308] | 1,285,173.6 | 0.52x |
| QECTOR OpenCL Batch (GPU, unweighted) | 3 | 100,000 | 2215 | 0.02215 | [0.02126, 0.02308] | 1,331,565.0 | 0.53x |
| QECTOR Sparse Blossom (CPU) | 3 | 100,000 | 1891 | 0.01891 | [0.01808, 0.01977] | 349,901.3 | 0.14x |
| QECTOR Union-Find (CPU) | 3 | 100,000 | 2210 | 0.02210 | [0.02121, 0.02303] | 1,314,711.8 | 0.53x |
| PyMatching v2 (C++) | 5 | 1,000 | 19 | 0.01900 | [0.01220, 0.02948] | 286,886.4 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 1,000 | 60 | 0.06000 | [0.04690, 0.07647] | 86,090.4 | 0.30x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 1,000 | 60 | 0.06000 | [0.04690, 0.07647] | 126,542.2 | 0.44x |
| QECTOR Sparse Blossom (CPU) | 5 | 1,000 | 19 | 0.01900 | [0.01220, 0.02948] | 11,668.7 | 0.04x |
| QECTOR Union-Find (CPU) | 5 | 1,000 | 26 | 0.02600 | [0.01780, 0.03782] | 95,464.5 | 0.33x |
| ldpc BP-OSD | 5 | 1,000 | 21 | 0.02100 | [0.01378, 0.03189] | 109.4 | 0x |
| PyMatching v2 (C++) | 5 | 5,000 | 73 | 0.01460 | [0.01163, 0.01832] | 314,724.7 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 5,000 | 336 | 0.06720 | [0.06059, 0.07448] | 230,917.0 | 0.73x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 5,000 | 336 | 0.06720 | [0.06059, 0.07448] | 231,072.8 | 0.73x |
| QECTOR Sparse Blossom (CPU) | 5 | 5,000 | 73 | 0.01460 | [0.01163, 0.01832] | 11,510.5 | 0.04x |
| QECTOR Union-Find (CPU) | 5 | 5,000 | 148 | 0.02960 | [0.02525, 0.03467] | 118,257.3 | 0.38x |
| PyMatching v2 (C++) | 5 | 10,000 | 166 | 0.01660 | [0.01428, 0.01930] | 322,491.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 10,000 | 600 | 0.06000 | [0.05551, 0.06483] | 243,179.4 | 0.75x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 10,000 | 600 | 0.06000 | [0.05551, 0.06483] | 172,102.2 | 0.53x |
| QECTOR Sparse Blossom (CPU) | 5 | 10,000 | 166 | 0.01660 | [0.01428, 0.01930] | 11,415.1 | 0.04x |
| QECTOR Union-Find (CPU) | 5 | 10,000 | 268 | 0.02680 | [0.02381, 0.03015] | 130,041.0 | 0.40x |
| PyMatching v2 (C++) | 5 | 50,000 | 809 | 0.01618 | [0.01511, 0.01732] | 327,439.4 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 50,000 | 3041 | 0.06082 | [0.05876, 0.06295] | 137,869.5 | 0.42x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 50,000 | 3041 | 0.06082 | [0.05876, 0.06295] | 141,548.6 | 0.43x |
| QECTOR Sparse Blossom (CPU) | 5 | 50,000 | 810 | 0.01620 | [0.01513, 0.01734] | 11,100.8 | 0.03x |
| QECTOR Union-Find (CPU) | 5 | 50,000 | 1356 | 0.02712 | [0.02573, 0.02858] | 136,080.4 | 0.42x |
| PyMatching v2 (C++) | 5 | 100,000 | 1596 | 0.01596 | [0.01520, 0.01676] | 325,420.7 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 5 | 100,000 | 6094 | 0.06094 | [0.05947, 0.06244] | 138,720.7 | 0.43x |
| QECTOR OpenCL Batch (GPU, unweighted) | 5 | 100,000 | 6094 | 0.06094 | [0.05947, 0.06244] | 143,444.5 | 0.44x |
| QECTOR Sparse Blossom (CPU) | 5 | 100,000 | 1596 | 0.01596 | [0.01520, 0.01676] | 11,121.1 | 0.03x |
| QECTOR Union-Find (CPU) | 5 | 100,000 | 2645 | 0.02645 | [0.02547, 0.02746] | 139,133.3 | 0.43x |
| PyMatching v2 (C++) | 7 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 90,436.4 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 21,781.7 | 0.24x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 39,889.6 | 0.44x |
| QECTOR Sparse Blossom (CPU) | 7 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 1,917.3 | 0.02x |
| QECTOR Union-Find (CPU) | 7 | 1,000 | 18 | 0.01800 | [0.01142, 0.02827] | 11,689.3 | 0.13x |
| PyMatching v2 (C++) | 7 | 5,000 | 74 | 0.01480 | [0.01181, 0.01854] | 96,754.5 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 5,000 | 218 | 0.04360 | [0.03828, 0.04962] | 69,309.2 | 0.72x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 5,000 | 218 | 0.04360 | [0.03828, 0.04962] | 62,728.6 | 0.65x |
| QECTOR Sparse Blossom (CPU) | 7 | 5,000 | 75 | 0.01500 | [0.01198, 0.01876] | 1,870.5 | 0.02x |
| QECTOR Union-Find (CPU) | 7 | 5,000 | 107 | 0.02140 | [0.01774, 0.02579] | 23,848.1 | 0.25x |
| PyMatching v2 (C++) | 7 | 10,000 | 130 | 0.01300 | [0.01096, 0.01541] | 99,587.8 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 10,000 | 409 | 0.04090 | [0.03719, 0.04496] | 63,533.3 | 0.64x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 10,000 | 409 | 0.04090 | [0.03719, 0.04496] | 52,989.1 | 0.53x |
| QECTOR Sparse Blossom (CPU) | 7 | 10,000 | 133 | 0.01330 | [0.01123, 0.01574] | 1,925.2 | 0.02x |
| QECTOR Union-Find (CPU) | 7 | 10,000 | 209 | 0.02090 | [0.01827, 0.02389] | 25,553.6 | 0.26x |
| PyMatching v2 (C++) | 7 | 50,000 | 619 | 0.01238 | [0.01145, 0.01339] | 98,837.3 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 50,000 | 2197 | 0.04394 | [0.04218, 0.04577] | 40,702.7 | 0.41x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 50,000 | 2197 | 0.04394 | [0.04218, 0.04577] | 37,017.2 | 0.38x |
| QECTOR Union-Find (CPU) | 7 | 50,000 | 1055 | 0.02110 | [0.01988, 0.02240] | 24,486.8 | 0.25x |
| PyMatching v2 (C++) | 7 | 100,000 | 1220 | 0.01220 | [0.01154, 0.01290] | 100,735.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 7 | 100,000 | 4274 | 0.04274 | [0.04150, 0.04401] | 41,777.3 | 0.41x |
| QECTOR OpenCL Batch (GPU, unweighted) | 7 | 100,000 | 4274 | 0.04274 | [0.04150, 0.04401] | 38,330.7 | 0.38x |
| QECTOR Union-Find (CPU) | 7 | 100,000 | 2042 | 0.02042 | [0.01956, 0.02132] | 27,570.8 | 0.27x |
| PyMatching v2 (C++) | 9 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 36,641.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 1,000 | 56 | 0.05600 | [0.04337, 0.07202] | 9,526.9 | 0.26x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 1,000 | 56 | 0.05600 | [0.04337, 0.07202] | 16,532.8 | 0.45x |
| QECTOR Sparse Blossom (CPU) | 9 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 247.9 | 0.007x |
| QECTOR Union-Find (CPU) | 9 | 1,000 | 14 | 0.01400 | [0.00836, 0.02336] | 1,852.1 | 0.05x |
| PyMatching v2 (C++) | 9 | 5,000 | 36 | 0.00720 | [0.00521, 0.00995] | 40,426.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 5,000 | 224 | 0.04480 | [0.03941, 0.05089] | 31,021.0 | 0.77x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 5,000 | 224 | 0.04480 | [0.03941, 0.05089] | 25,000.7 | 0.62x |
| QECTOR Sparse Blossom (CPU) | 9 | 5,000 | 36 | 0.00720 | [0.00521, 0.00995] | 250.5 | 0.006x |
| QECTOR Union-Find (CPU) | 9 | 5,000 | 85 | 0.01700 | [0.01377, 0.02097] | 6,284.9 | 0.15x |
| PyMatching v2 (C++) | 9 | 10,000 | 78 | 0.00780 | [0.00625, 0.00972] | 39,081.3 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 10,000 | 431 | 0.04310 | [0.03929, 0.04726] | 27,047.0 | 0.69x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 10,000 | 431 | 0.04310 | [0.03929, 0.04726] | 21,219.1 | 0.54x |
| QECTOR Union-Find (CPU) | 9 | 10,000 | 156 | 0.01560 | [0.01335, 0.01822] | 5,076.8 | 0.13x |
| PyMatching v2 (C++) | 9 | 50,000 | 450 | 0.00900 | [0.00821, 0.00987] | 40,958.8 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 50,000 | 2324 | 0.04648 | [0.04467, 0.04836] | 16,216.7 | 0.40x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 50,000 | 2324 | 0.04648 | [0.04467, 0.04836] | 16,127.3 | 0.39x |
| QECTOR Union-Find (CPU) | 9 | 50,000 | 827 | 0.01654 | [0.01546, 0.01770] | 5,360.9 | 0.13x |
| PyMatching v2 (C++) | 9 | 100,000 | 878 | 0.00878 | [0.00822, 0.00938] | 40,874.0 | 1.00x |
| QECTOR OpenCL Batch (GPU, unweighted) | 9 | 100,000 | 4663 | 0.04663 | [0.04534, 0.04795] | 16,555.5 | 0.41x |
| QECTOR Union-Find (CPU) | 9 | 100,000 | 1732 | 0.01732 | [0.01653, 0.01815] | 5,354.3 | 0.13x |
| PyMatching v2 (C++) | 11 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 18,454.8 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 5,662.9 | 0.31x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 1,000 | 44 | 0.04400 | [0.03294, 0.05855] | 7,898.3 | 0.43x |
| QECTOR Sparse Blossom (CPU) | 11 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 79.1 | 0.004x |
| QECTOR Union-Find (CPU) | 11 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 493.1 | 0.03x |
| PyMatching v2 (C++) | 11 | 5,000 | 24 | 0.00480 | [0.00323, 0.00713] | 19,863.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 5,000 | 229 | 0.04580 | [0.04035, 0.05195] | 15,865.4 | 0.80x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 5,000 | 229 | 0.04580 | [0.04035, 0.05195] | 12,607.9 | 0.64x |
| QECTOR Union-Find (CPU) | 11 | 5,000 | 71 | 0.01420 | [0.01127, 0.01787] | 956.9 | 0.05x |
| PyMatching v2 (C++) | 11 | 10,000 | 77 | 0.00770 | [0.00617, 0.00961] | 20,957.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 10,000 | 440 | 0.04400 | [0.04015, 0.04820] | 13,405.5 | 0.64x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 10,000 | 440 | 0.04400 | [0.04015, 0.04820] | 10,709.1 | 0.51x |
| PyMatching v2 (C++) | 11 | 50,000 | 334 | 0.00668 | [0.00600, 0.00743] | 21,250.7 | 1.00x |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 50,000 | 2111 | 0.04222 | [0.04049, 0.04402] | 8,152.9 | 0.38x |
| PyMatching v2 (C++) | 11 | 100,000 | 647 | 0.00647 | [0.00599, 0.00699] | 21,243.6 | 1.00x |
| PyMatching v2 (C++) | 13 | 1,000 | 3 | 0.00300 | [0.00102, 0.00878] | 11,222.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 1,000 | 39 | 0.03900 | [0.02866, 0.05287] | 3,219.0 | 0.29x |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 1,000 | 39 | 0.03900 | [0.02866, 0.05287] | 4,712.0 | 0.42x |
| QECTOR Union-Find (CPU) | 13 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 175.8 | 0.02x |
| PyMatching v2 (C++) | 13 | 5,000 | 18 | 0.00360 | [0.00228, 0.00568] | 11,971.0 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 5,000 | 201 | 0.04020 | [0.03510, 0.04601] | 9,106.1 | 0.76x |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 5,000 | 201 | 0.04020 | [0.03510, 0.04601] | 7,144.5 | 0.60x |
| PyMatching v2 (C++) | 13 | 10,000 | 37 | 0.00370 | [0.00269, 0.00510] | 12,135.2 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 10,000 | 410 | 0.04100 | [0.03729, 0.04507] | 7,611.7 | 0.63x |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 10,000 | 410 | 0.04100 | [0.03729, 0.04507] | 4,590.3 | 0.38x |
| PyMatching v2 (C++) | 13 | 50,000 | 221 | 0.00442 | [0.00388, 0.00504] | 11,935.5 | 1.00x |
| PyMatching v2 (C++) | 13 | 100,000 | 445 | 0.00445 | [0.00406, 0.00488] | 11,931.3 | 1.00x |
| PyMatching v2 (C++) | 15 | 1,000 | 2 | 0.00200 | [0.00055, 0.00726] | 5,967.5 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 1,000 | 41 | 0.04100 | [0.03036, 0.05515] | 1,929.0 | 0.32x |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 1,000 | 41 | 0.04100 | [0.03036, 0.05515] | 2,774.5 | 0.47x |
| PyMatching v2 (C++) | 15 | 5,000 | 12 | 0.00240 | [0.00137, 0.00419] | 6,813.9 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 5,000 | 170 | 0.03400 | [0.02932, 0.03939] | 5,347.4 | 0.79x |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 5,000 | 170 | 0.03400 | [0.02932, 0.03939] | 2,662.0 | 0.39x |
| PyMatching v2 (C++) | 15 | 10,000 | 44 | 0.00440 | [0.00328, 0.00590] | 7,060.6 | 1.00x |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 10,000 | 376 | 0.03760 | [0.03405, 0.04151] | 4,486.3 | 0.64x |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 10,000 | 376 | 0.03760 | [0.03405, 0.04151] | 2,797.9 | 0.40x |
| PyMatching v2 (C++) | 15 | 50,000 | 171 | 0.00342 | [0.00295, 0.00397] | 7,240.2 | 1.00x |
| PyMatching v2 (C++) | 15 | 100,000 | 314 | 0.00314 | [0.00281, 0.00351] | 7,296.0 | 1.00x |

## Not measured (73 cells)

These cells were **not run** and carry no numbers. They are listed so the gaps in the grid are explicit rather than quietly filled in.

| Decoder | d | Shots | Reason | Probe rate (dec/s) | Projected decode |
|:---|:---:|---:|:---|---:|---:|
| ldpc BP-OSD | 3 | 100,000 | over per-cell decode budget | 2,482.2 | 40s |
| ldpc BP-OSD | 5 | 5,000 | over per-cell decode budget | 106.4 | 47s |
| ldpc BP-OSD | 5 | 10,000 | over per-cell decode budget | 106.4 | 94s |
| ldpc BP-OSD | 5 | 50,000 | over per-cell decode budget | 106.4 | 470s |
| ldpc BP-OSD | 5 | 100,000 | over per-cell decode budget | 106.4 | 940s |
| QECTOR Sparse Blossom (CPU) | 7 | 50,000 | over per-cell decode budget | 1,629.3 | 31s |
| QECTOR Sparse Blossom (CPU) | 7 | 100,000 | over per-cell decode budget | 1,629.3 | 61s |
| ldpc BP-OSD | 7 | 1,000 | over per-cell decode budget | 28.0 | 36s |
| ldpc BP-OSD | 7 | 5,000 | over per-cell decode budget | 28.0 | 179s |
| ldpc BP-OSD | 7 | 10,000 | over per-cell decode budget | 28.0 | 357s |
| ldpc BP-OSD | 7 | 50,000 | over per-cell decode budget | 28.0 | 1,785s |
| ldpc BP-OSD | 7 | 100,000 | over per-cell decode budget | 28.0 | 3,570s |
| QECTOR Sparse Blossom (CPU) | 9 | 10,000 | over per-cell decode budget | 223.2 | 45s |
| QECTOR Sparse Blossom (CPU) | 9 | 50,000 | over per-cell decode budget | 223.2 | 224s |
| QECTOR Sparse Blossom (CPU) | 9 | 100,000 | over per-cell decode budget | 223.2 | 448s |
| QECTOR CUDA Batch (GPU, unweighted) | 9 | 100,000 | over per-cell decode budget | 3,146.1 | 32s |
| ldpc BP-OSD | 9 | 1,000 | over per-cell decode budget | 10.8 | 93s |
| ldpc BP-OSD | 9 | 5,000 | over per-cell decode budget | 10.8 | 463s |
| ldpc BP-OSD | 9 | 10,000 | over per-cell decode budget | 10.8 | 925s |
| ldpc BP-OSD | 9 | 50,000 | over per-cell decode budget | 10.8 | 4,627s |
| ldpc BP-OSD | 9 | 100,000 | over per-cell decode budget | 10.8 | 9,255s |
| QECTOR Sparse Blossom (CPU) | 11 | 5,000 | over per-cell decode budget | 77.6 | 64s |
| QECTOR Sparse Blossom (CPU) | 11 | 10,000 | over per-cell decode budget | 77.6 | 129s |
| QECTOR Sparse Blossom (CPU) | 11 | 50,000 | over per-cell decode budget | 77.6 | 644s |
| QECTOR Sparse Blossom (CPU) | 11 | 100,000 | over per-cell decode budget | 77.6 | 1,288s |
| QECTOR Union-Find (CPU) | 11 | 10,000 | over per-cell decode budget | 251.5 | 40s |
| QECTOR Union-Find (CPU) | 11 | 50,000 | over per-cell decode budget | 251.5 | 199s |
| QECTOR Union-Find (CPU) | 11 | 100,000 | over per-cell decode budget | 251.5 | 398s |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 50,000 | over per-cell decode budget | 1,480.4 | 34s |
| QECTOR CUDA Batch (GPU, unweighted) | 11 | 100,000 | over per-cell decode budget | 1,480.4 | 68s |
| QECTOR OpenCL Batch (GPU, unweighted) | 11 | 100,000 | over per-cell decode budget | 2,690.8 | 37s |
| ldpc BP-OSD | 11 | 1,000 | over per-cell decode budget | 5.0 | 202s |
| ldpc BP-OSD | 11 | 5,000 | over per-cell decode budget | 5.0 | 1,009s |
| ldpc BP-OSD | 11 | 10,000 | over per-cell decode budget | 5.0 | 2,018s |
| ldpc BP-OSD | 11 | 50,000 | over per-cell decode budget | 5.0 | 10,092s |
| ldpc BP-OSD | 11 | 100,000 | over per-cell decode budget | 5.0 | 20,184s |
| QECTOR Sparse Blossom (CPU) | 13 | 1,000 | over per-cell decode budget | 30.2 | 33s |
| QECTOR Sparse Blossom (CPU) | 13 | 5,000 | over per-cell decode budget | 30.2 | 166s |
| QECTOR Sparse Blossom (CPU) | 13 | 10,000 | over per-cell decode budget | 30.2 | 331s |
| QECTOR Sparse Blossom (CPU) | 13 | 50,000 | over per-cell decode budget | 30.2 | 1,656s |
| QECTOR Sparse Blossom (CPU) | 13 | 100,000 | over per-cell decode budget | 30.2 | 3,311s |
| QECTOR Union-Find (CPU) | 13 | 5,000 | over per-cell decode budget | 90.3 | 55s |
| QECTOR Union-Find (CPU) | 13 | 10,000 | over per-cell decode budget | 90.3 | 111s |
| QECTOR Union-Find (CPU) | 13 | 50,000 | over per-cell decode budget | 90.3 | 554s |
| QECTOR Union-Find (CPU) | 13 | 100,000 | over per-cell decode budget | 90.3 | 1,107s |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 50,000 | over per-cell decode budget | 858.5 | 58s |
| QECTOR CUDA Batch (GPU, unweighted) | 13 | 100,000 | over per-cell decode budget | 858.5 | 116s |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 50,000 | over per-cell decode budget | 1,495.3 | 33s |
| QECTOR OpenCL Batch (GPU, unweighted) | 13 | 100,000 | over per-cell decode budget | 1,495.3 | 67s |
| ldpc BP-OSD | 13 | 1,000 | over per-cell decode budget | 2.9 | 341s |
| ldpc BP-OSD | 13 | 5,000 | over per-cell decode budget | 2.9 | 1,707s |
| ldpc BP-OSD | 13 | 10,000 | over per-cell decode budget | 2.9 | 3,413s |
| ldpc BP-OSD | 13 | 50,000 | over per-cell decode budget | 2.9 | 17,065s |
| ldpc BP-OSD | 13 | 100,000 | over per-cell decode budget | 2.9 | 34,130s |
| QECTOR Sparse Blossom (CPU) | 15 | 1,000 | over per-cell decode budget | 12.1 | 82s |
| QECTOR Sparse Blossom (CPU) | 15 | 5,000 | over per-cell decode budget | 12.1 | 412s |
| QECTOR Sparse Blossom (CPU) | 15 | 10,000 | over per-cell decode budget | 12.1 | 824s |
| QECTOR Sparse Blossom (CPU) | 15 | 50,000 | over per-cell decode budget | 12.1 | 4,120s |
| QECTOR Sparse Blossom (CPU) | 15 | 100,000 | over per-cell decode budget | 12.1 | 8,240s |
| QECTOR Union-Find (CPU) | 15 | 1,000 | over per-cell decode budget | 33.0 | 30s |
| QECTOR Union-Find (CPU) | 15 | 5,000 | over per-cell decode budget | 33.0 | 152s |
| QECTOR Union-Find (CPU) | 15 | 10,000 | over per-cell decode budget | 33.0 | 303s |
| QECTOR Union-Find (CPU) | 15 | 50,000 | over per-cell decode budget | 33.0 | 1,517s |
| QECTOR Union-Find (CPU) | 15 | 100,000 | over per-cell decode budget | 33.0 | 3,034s |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 50,000 | over per-cell decode budget | 500.6 | 100s |
| QECTOR CUDA Batch (GPU, unweighted) | 15 | 100,000 | over per-cell decode budget | 500.6 | 200s |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 50,000 | over per-cell decode budget | 826.8 | 60s |
| QECTOR OpenCL Batch (GPU, unweighted) | 15 | 100,000 | over per-cell decode budget | 826.8 | 121s |
| ldpc BP-OSD | 15 | 1,000 | over per-cell decode budget | 1.8 | 548s |
| ldpc BP-OSD | 15 | 5,000 | over per-cell decode budget | 1.8 | 2,738s |
| ldpc BP-OSD | 15 | 10,000 | over per-cell decode budget | 1.8 | 5,475s |
| ldpc BP-OSD | 15 | 50,000 | over per-cell decode budget | 1.8 | 27,377s |
| ldpc BP-OSD | 15 | 100,000 | over per-cell decode budget | 1.8 | 54,753s |

## How to read this

- Throughput figures are only meaningful on an otherwise-idle machine.
- LER figures are subject to binomial error; at low p and low shot counts the confidence interval can exceed the difference between decoders. Check ci95_lo/ci95_hi per row.
- Accuracy and speed are independent axes: a lower LER at the same (d, p) is more accurate, a higher throughput is faster. This table deliberately does not collapse them into one score.
- Throughput counts decode time only — `LerResult.seconds` wraps the single `decode_batch` call. Circuit construction and sampling are excluded for every decoder equally.
- A cell with 0 errors is not evidence of a zero error rate; read its `ci95_hi` as an upper bound. The LER chart plots those as open downward markers.
