# QECTOR v0.7.0 — circuit-level decoder comparison

Every row below is one `ler.estimate_ler_circuit_level` measurement: the same Stim rotated-surface-code circuit, the same decomposed DEM, the same detector/observable samples and the same `decode_batch` resolver for every decoder, scored against the circuit's own logical observables. `ler.assert_comparable` gated these rows before writing.

- **Noise model**: circuit-level, p = 0.005 (gate, reset and measurement noise over d rounds of syndrome extraction)
- **Per-cell decode budget**: 25s. Cells projected to exceed it appear under *Not measured* — nothing is extrapolated.
- **Git commit**: `b0f5456470` (tree dirty: True)
- **Platform**: Windows-10-10.0.26100-SP0 · Python 3.11.9
- **Versions**: qector 0.7.0, pymatching 2.4.0, ldpc ?, stim 1.16.0


## Measured results (77 cells)

| Decoder | d | Shots | Errors | LER | 95% CI | Throughput (dec/s) | vs PyMatching |
|:---|:---:|---:|---:|---:|:---:|---:|---:|
| PyMatching v2 (C++) | 3 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 1,949,317.9 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 3 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 154,268.6 | 0.08x |
| QECTOR Union-Find (CPU) | 3 | 1,000 | 16 | 0.01600 | [0.00987, 0.02583] | 222,321.0 | 0.11x |
| ldpc BP-OSD | 3 | 1,000 | 12 | 0.01200 | [0.00688, 0.02086] | 2,266.7 | 0.001x |
| PyMatching v2 (C++) | 3 | 5,000 | 84 | 0.01680 | [0.01359, 0.02075] | 1,865,810.9 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 3 | 5,000 | 84 | 0.01680 | [0.01359, 0.02075] | 236,337.3 | 0.13x |
| QECTOR Union-Find (CPU) | 3 | 5,000 | 105 | 0.02100 | [0.01738, 0.02536] | 481,023.6 | 0.26x |
| ldpc BP-OSD | 3 | 5,000 | 87 | 0.01740 | [0.01413, 0.02141] | 2,285.6 | 0.001x |
| PyMatching v2 (C++) | 3 | 10,000 | 202 | 0.02020 | [0.01762, 0.02315] | 1,848,941.5 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 3 | 10,000 | 202 | 0.02020 | [0.01762, 0.02315] | 248,615.8 | 0.13x |
| QECTOR Union-Find (CPU) | 3 | 10,000 | 233 | 0.02330 | [0.02052, 0.02644] | 571,931.9 | 0.31x |
| ldpc BP-OSD | 3 | 10,000 | 199 | 0.01990 | [0.01734, 0.02283] | 2,329.7 | 0.001x |
| PyMatching v2 (C++) | 3 | 50,000 | 960 | 0.01920 | [0.01803, 0.02044] | 2,234,836.6 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 3 | 50,000 | 960 | 0.01920 | [0.01803, 0.02044] | 250,396.0 | 0.11x |
| QECTOR Union-Find (CPU) | 3 | 50,000 | 1128 | 0.02256 | [0.02129, 0.02390] | 812,805.9 | 0.36x |
| ldpc BP-OSD | 3 | 50,000 | 969 | 0.01938 | [0.01821, 0.02063] | 2,259.1 | 0.001x |
| PyMatching v2 (C++) | 3 | 100,000 | 1891 | 0.01891 | [0.01808, 0.01977] | 2,437,651.0 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 3 | 100,000 | 1891 | 0.01891 | [0.01808, 0.01977] | 245,628.9 | 0.10x |
| QECTOR Union-Find (CPU) | 3 | 100,000 | 2210 | 0.02210 | [0.02121, 0.02303] | 825,725.6 | 0.34x |
| PyMatching v2 (C++) | 5 | 1,000 | 19 | 0.01900 | [0.01220, 0.02948] | 266,410.9 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 5 | 1,000 | 19 | 0.01900 | [0.01220, 0.02948] | 8,690.2 | 0.03x |
| QECTOR Union-Find (CPU) | 5 | 1,000 | 26 | 0.02600 | [0.01780, 0.03782] | 79,887.5 | 0.30x |
| ldpc BP-OSD | 5 | 1,000 | 21 | 0.02100 | [0.01378, 0.03189] | 87.8 | 0x |
| PyMatching v2 (C++) | 5 | 5,000 | 73 | 0.01460 | [0.01163, 0.01832] | 286,109.9 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 5 | 5,000 | 73 | 0.01460 | [0.01163, 0.01832] | 9,018.3 | 0.03x |
| QECTOR Union-Find (CPU) | 5 | 5,000 | 148 | 0.02960 | [0.02525, 0.03467] | 91,362.6 | 0.32x |
| PyMatching v2 (C++) | 5 | 10,000 | 166 | 0.01660 | [0.01428, 0.01930] | 234,845.4 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 5 | 10,000 | 166 | 0.01660 | [0.01428, 0.01930] | 9,204.4 | 0.04x |
| QECTOR Union-Find (CPU) | 5 | 10,000 | 268 | 0.02680 | [0.02381, 0.03015] | 97,208.1 | 0.41x |
| PyMatching v2 (C++) | 5 | 50,000 | 809 | 0.01618 | [0.01511, 0.01732] | 225,455.0 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 5 | 50,000 | 810 | 0.01620 | [0.01513, 0.01734] | 8,990.2 | 0.04x |
| QECTOR Union-Find (CPU) | 5 | 50,000 | 1356 | 0.02712 | [0.02573, 0.02858] | 113,116.3 | 0.50x |
| PyMatching v2 (C++) | 5 | 100,000 | 1596 | 0.01596 | [0.01520, 0.01676] | 249,064.6 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 5 | 100,000 | 1596 | 0.01596 | [0.01520, 0.01676] | 9,121.1 | 0.04x |
| QECTOR Union-Find (CPU) | 5 | 100,000 | 2645 | 0.02645 | [0.02547, 0.02746] | 112,796.6 | 0.45x |
| PyMatching v2 (C++) | 7 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 54,523.2 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 7 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 1,631.4 | 0.03x |
| QECTOR Union-Find (CPU) | 7 | 1,000 | 18 | 0.01800 | [0.01142, 0.02827] | 9,141.5 | 0.17x |
| PyMatching v2 (C++) | 7 | 5,000 | 74 | 0.01480 | [0.01181, 0.01854] | 62,572.1 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 7 | 5,000 | 75 | 0.01500 | [0.01198, 0.01876] | 1,565.3 | 0.03x |
| QECTOR Union-Find (CPU) | 7 | 5,000 | 107 | 0.02140 | [0.01774, 0.02579] | 20,730.9 | 0.33x |
| PyMatching v2 (C++) | 7 | 10,000 | 130 | 0.01300 | [0.01096, 0.01541] | 89,104.5 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 7 | 10,000 | 133 | 0.01330 | [0.01123, 0.01574] | 1,650.7 | 0.02x |
| QECTOR Union-Find (CPU) | 7 | 10,000 | 209 | 0.02090 | [0.01827, 0.02389] | 22,427.5 | 0.25x |
| PyMatching v2 (C++) | 7 | 50,000 | 619 | 0.01238 | [0.01145, 0.01339] | 69,244.3 | 1.00x |
| QECTOR Union-Find (CPU) | 7 | 50,000 | 1055 | 0.02110 | [0.01988, 0.02240] | 21,552.3 | 0.31x |
| PyMatching v2 (C++) | 7 | 100,000 | 1220 | 0.01220 | [0.01154, 0.01290] | 75,767.0 | 1.00x |
| QECTOR Union-Find (CPU) | 7 | 100,000 | 2042 | 0.02042 | [0.01956, 0.02132] | 22,642.8 | 0.30x |
| PyMatching v2 (C++) | 9 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 16,121.6 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 9 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 206.8 | 0.01x |
| QECTOR Union-Find (CPU) | 9 | 1,000 | 14 | 0.01400 | [0.00836, 0.02336] | 1,404.2 | 0.09x |
| PyMatching v2 (C++) | 9 | 5,000 | 36 | 0.00720 | [0.00521, 0.00995] | 28,259.0 | 1.00x |
| QECTOR Union-Find (CPU) | 9 | 5,000 | 85 | 0.01700 | [0.01377, 0.02097] | 4,691.5 | 0.17x |
| PyMatching v2 (C++) | 9 | 10,000 | 78 | 0.00780 | [0.00625, 0.00972] | 27,629.9 | 1.00x |
| QECTOR Union-Find (CPU) | 9 | 10,000 | 156 | 0.01560 | [0.01335, 0.01822] | 3,812.1 | 0.14x |
| PyMatching v2 (C++) | 9 | 50,000 | 450 | 0.00900 | [0.00821, 0.00987] | 29,152.0 | 1.00x |
| QECTOR Union-Find (CPU) | 9 | 50,000 | 827 | 0.01654 | [0.01546, 0.01770] | 4,599.5 | 0.16x |
| PyMatching v2 (C++) | 9 | 100,000 | 878 | 0.00878 | [0.00822, 0.00938] | 29,244.1 | 1.00x |
| QECTOR Union-Find (CPU) | 9 | 100,000 | 1732 | 0.01732 | [0.01653, 0.01815] | 4,606.0 | 0.16x |
| PyMatching v2 (C++) | 11 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 11,490.2 | 1.00x |
| QECTOR Sparse Blossom (CPU) | 11 | 1,000 | 7 | 0.00700 | [0.00339, 0.01438] | 84.2 | 0.007x |
| QECTOR Union-Find (CPU) | 11 | 1,000 | 13 | 0.01300 | [0.00761, 0.02211] | 367.3 | 0.03x |
| PyMatching v2 (C++) | 11 | 5,000 | 24 | 0.00480 | [0.00323, 0.00713] | 12,516.9 | 1.00x |
| PyMatching v2 (C++) | 11 | 10,000 | 77 | 0.00770 | [0.00617, 0.00961] | 13,693.2 | 1.00x |
| PyMatching v2 (C++) | 11 | 50,000 | 334 | 0.00668 | [0.00600, 0.00743] | 14,040.2 | 1.00x |
| PyMatching v2 (C++) | 11 | 100,000 | 647 | 0.00647 | [0.00599, 0.00699] | 13,815.7 | 1.00x |
| PyMatching v2 (C++) | 13 | 1,000 | 3 | 0.00300 | [0.00102, 0.00878] | 7,384.5 | 1.00x |
| QECTOR Union-Find (CPU) | 13 | 1,000 | 11 | 0.01100 | [0.00615, 0.01959] | 132.6 | 0.02x |
| PyMatching v2 (C++) | 13 | 5,000 | 18 | 0.00360 | [0.00228, 0.00568] | 7,576.5 | 1.00x |
| PyMatching v2 (C++) | 13 | 10,000 | 37 | 0.00370 | [0.00269, 0.00510] | 6,932.6 | 1.00x |
| PyMatching v2 (C++) | 13 | 50,000 | 221 | 0.00442 | [0.00388, 0.00504] | 7,243.0 | 1.00x |
| PyMatching v2 (C++) | 13 | 100,000 | 445 | 0.00445 | [0.00406, 0.00488] | 7,218.3 | 1.00x |
| PyMatching v2 (C++) | 15 | 1,000 | 2 | 0.00200 | [0.00055, 0.00726] | 3,350.7 | 1.00x |
| PyMatching v2 (C++) | 15 | 5,000 | 12 | 0.00240 | [0.00137, 0.00419] | 3,887.9 | 1.00x |
| PyMatching v2 (C++) | 15 | 10,000 | 44 | 0.00440 | [0.00328, 0.00590] | 4,107.9 | 1.00x |
| PyMatching v2 (C++) | 15 | 50,000 | 171 | 0.00342 | [0.00295, 0.00397] | 4,074.8 | 1.00x |
| PyMatching v2 (C++) | 15 | 100,000 | 314 | 0.00314 | [0.00281, 0.00351] | 6,480.5 | 1.00x |

## Not measured (63 cells)

These cells were **not run** and carry no numbers. They are listed so the gaps in the grid are explicit rather than quietly filled in.

| Decoder | d | Shots | Reason | Probe rate (dec/s) | Projected decode |
|:---|:---:|---:|:---|---:|---:|
| ldpc BP-OSD | 3 | 100,000 | over per-cell decode budget | 2,445.9 | 41s |
| ldpc BP-OSD | 5 | 5,000 | over per-cell decode budget | 82.1 | 61s |
| ldpc BP-OSD | 5 | 10,000 | over per-cell decode budget | 82.1 | 122s |
| ldpc BP-OSD | 5 | 50,000 | over per-cell decode budget | 82.1 | 609s |
| ldpc BP-OSD | 5 | 100,000 | over per-cell decode budget | 82.1 | 1,217s |
| QECTOR Sparse Blossom (CPU) | 7 | 50,000 | over per-cell decode budget | 1,405.9 | 36s |
| QECTOR Sparse Blossom (CPU) | 7 | 100,000 | over per-cell decode budget | 1,405.9 | 71s |
| ldpc BP-OSD | 7 | 1,000 | over per-cell decode budget | 19.0 | 53s |
| ldpc BP-OSD | 7 | 5,000 | over per-cell decode budget | 19.0 | 263s |
| ldpc BP-OSD | 7 | 10,000 | over per-cell decode budget | 19.0 | 526s |
| ldpc BP-OSD | 7 | 50,000 | over per-cell decode budget | 19.0 | 2,629s |
| ldpc BP-OSD | 7 | 100,000 | over per-cell decode budget | 19.0 | 5,258s |
| QECTOR Sparse Blossom (CPU) | 9 | 5,000 | over per-cell decode budget | 190.4 | 26s |
| QECTOR Sparse Blossom (CPU) | 9 | 10,000 | over per-cell decode budget | 190.4 | 53s |
| QECTOR Sparse Blossom (CPU) | 9 | 50,000 | over per-cell decode budget | 190.4 | 263s |
| QECTOR Sparse Blossom (CPU) | 9 | 100,000 | over per-cell decode budget | 190.4 | 525s |
| ldpc BP-OSD | 9 | 1,000 | over per-cell decode budget | 8.4 | 119s |
| ldpc BP-OSD | 9 | 5,000 | over per-cell decode budget | 8.4 | 597s |
| ldpc BP-OSD | 9 | 10,000 | over per-cell decode budget | 8.4 | 1,193s |
| ldpc BP-OSD | 9 | 50,000 | over per-cell decode budget | 8.4 | 5,967s |
| ldpc BP-OSD | 9 | 100,000 | over per-cell decode budget | 8.4 | 11,935s |
| QECTOR Sparse Blossom (CPU) | 11 | 5,000 | over per-cell decode budget | 82.0 | 61s |
| QECTOR Sparse Blossom (CPU) | 11 | 10,000 | over per-cell decode budget | 82.0 | 122s |
| QECTOR Sparse Blossom (CPU) | 11 | 50,000 | over per-cell decode budget | 82.0 | 609s |
| QECTOR Sparse Blossom (CPU) | 11 | 100,000 | over per-cell decode budget | 82.0 | 1,219s |
| QECTOR Union-Find (CPU) | 11 | 5,000 | over per-cell decode budget | 172.0 | 29s |
| QECTOR Union-Find (CPU) | 11 | 10,000 | over per-cell decode budget | 172.0 | 58s |
| QECTOR Union-Find (CPU) | 11 | 50,000 | over per-cell decode budget | 172.0 | 291s |
| QECTOR Union-Find (CPU) | 11 | 100,000 | over per-cell decode budget | 172.0 | 581s |
| ldpc BP-OSD | 11 | 1,000 | over per-cell decode budget | 3.7 | 271s |
| ldpc BP-OSD | 11 | 5,000 | over per-cell decode budget | 3.7 | 1,353s |
| ldpc BP-OSD | 11 | 10,000 | over per-cell decode budget | 3.7 | 2,706s |
| ldpc BP-OSD | 11 | 50,000 | over per-cell decode budget | 3.7 | 13,530s |
| ldpc BP-OSD | 11 | 100,000 | over per-cell decode budget | 3.7 | 27,059s |
| QECTOR Sparse Blossom (CPU) | 13 | 1,000 | over per-cell decode budget | 25.8 | 39s |
| QECTOR Sparse Blossom (CPU) | 13 | 5,000 | over per-cell decode budget | 25.8 | 194s |
| QECTOR Sparse Blossom (CPU) | 13 | 10,000 | over per-cell decode budget | 25.8 | 387s |
| QECTOR Sparse Blossom (CPU) | 13 | 50,000 | over per-cell decode budget | 25.8 | 1,937s |
| QECTOR Sparse Blossom (CPU) | 13 | 100,000 | over per-cell decode budget | 25.8 | 3,875s |
| QECTOR Union-Find (CPU) | 13 | 5,000 | over per-cell decode budget | 67.0 | 75s |
| QECTOR Union-Find (CPU) | 13 | 10,000 | over per-cell decode budget | 67.0 | 149s |
| QECTOR Union-Find (CPU) | 13 | 50,000 | over per-cell decode budget | 67.0 | 746s |
| QECTOR Union-Find (CPU) | 13 | 100,000 | over per-cell decode budget | 67.0 | 1,492s |
| ldpc BP-OSD | 13 | 1,000 | over per-cell decode budget | 2.2 | 452s |
| ldpc BP-OSD | 13 | 5,000 | over per-cell decode budget | 2.2 | 2,258s |
| ldpc BP-OSD | 13 | 10,000 | over per-cell decode budget | 2.2 | 4,516s |
| ldpc BP-OSD | 13 | 50,000 | over per-cell decode budget | 2.2 | 22,578s |
| ldpc BP-OSD | 13 | 100,000 | over per-cell decode budget | 2.2 | 45,156s |
| QECTOR Sparse Blossom (CPU) | 15 | 1,000 | over per-cell decode budget | 10.5 | 95s |
| QECTOR Sparse Blossom (CPU) | 15 | 5,000 | over per-cell decode budget | 10.5 | 474s |
| QECTOR Sparse Blossom (CPU) | 15 | 10,000 | over per-cell decode budget | 10.5 | 949s |
| QECTOR Sparse Blossom (CPU) | 15 | 50,000 | over per-cell decode budget | 10.5 | 4,745s |
| QECTOR Sparse Blossom (CPU) | 15 | 100,000 | over per-cell decode budget | 10.5 | 9,489s |
| QECTOR Union-Find (CPU) | 15 | 1,000 | over per-cell decode budget | 23.5 | 43s |
| QECTOR Union-Find (CPU) | 15 | 5,000 | over per-cell decode budget | 23.5 | 213s |
| QECTOR Union-Find (CPU) | 15 | 10,000 | over per-cell decode budget | 23.5 | 426s |
| QECTOR Union-Find (CPU) | 15 | 50,000 | over per-cell decode budget | 23.5 | 2,132s |
| QECTOR Union-Find (CPU) | 15 | 100,000 | over per-cell decode budget | 23.5 | 4,264s |
| ldpc BP-OSD | 15 | 1,000 | over per-cell decode budget | 1.4 | 729s |
| ldpc BP-OSD | 15 | 5,000 | over per-cell decode budget | 1.4 | 3,647s |
| ldpc BP-OSD | 15 | 10,000 | over per-cell decode budget | 1.4 | 7,294s |
| ldpc BP-OSD | 15 | 50,000 | over per-cell decode budget | 1.4 | 36,468s |
| ldpc BP-OSD | 15 | 100,000 | over per-cell decode budget | 1.4 | 72,937s |

## How to read this

- Throughput figures are only meaningful on an otherwise-idle machine.
- LER figures are subject to binomial error; at low p and low shot counts the confidence interval can exceed the difference between decoders. Check ci95_lo/ci95_hi per row.
- Accuracy and speed are independent axes: a lower LER at the same (d, p) is more accurate, a higher throughput is faster. This table deliberately does not collapse them into one score.
- Throughput counts decode time only — `LerResult.seconds` wraps the single `decode_batch` call. Circuit construction and sampling are excluded for every decoder equally.
- A cell with 0 errors is not evidence of a zero error rate; read its `ci95_hi` as an upper bound. The LER chart plots those as open downward markers.
