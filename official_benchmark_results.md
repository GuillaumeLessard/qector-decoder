# QECTOR v0.7.0 Comprehensive Benchmark Report (Full Data: d=3..31, Shots=1k..100k)

## Executive Summary

This official benchmark evaluates **QECTOR v0.7.0** (CPU & CUDA GPU) against standard industry baselines (**PyMatching v2.4** and **ldpc v2.4.1**) across distances $d=3..31$ and shot volumes up to $100,000$.

### Environment & Provenance Metadata

- **Git Commit**: `3d60a96e44`
- **Platform**: Windows-10-10.0.26100-SP0
- **Python Version**: 3.11.9
- **QECTOR Version**: 0.7.0
- **PyMatching Version**: 2.4.0
- **ldpc Version**: 2.4.1
- **Stim Version**: 1.16.0

## Full Throughput & Latency Table (264 Total Benchmark Configurations)

| Decoder | Category | d | Qubits | Shots | Throughput (dec/s) | Latency (µs) | LER (%) | Speedup vs PyMatching |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 3 | 9 | 1,000 | **305,782.3** | 3.27 | 0.000% | **3.84x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 3 | 9 | 1,000 | **423,710.9** | 2.36 | 0.000% | **5.32x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 3 | 9 | 1,000 | **102.9** | 9713.60 | 0.000% | **0.0x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 3 | 9 | 1,000 | **384,083.6** | 2.60 | 0.000% | **4.82x** |
| **PyMatching v2.4 (C++)** | PyMatching | 3 | 9 | 1,000 | **79,652.7** | 12.55 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 3 | 9 | 1,000 | **187,740.5** | 5.33 | 0.000% | **2.36x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 3 | 9 | 5,000 | **3,429,120.0** | 0.29 | 0.000% | **43.85x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 3 | 9 | 5,000 | **996,850.0** | 1.00 | 0.000% | **12.75x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 3 | 9 | 5,000 | **1,222.7** | 817.86 | 0.000% | **0.02x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 3 | 9 | 5,000 | **1,446,717.4** | 0.69 | 0.000% | **18.5x** |
| **PyMatching v2.4 (C++)** | PyMatching | 3 | 9 | 5,000 | **78,204.4** | 12.79 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 3 | 9 | 5,000 | **223,388.8** | 4.48 | 0.000% | **2.86x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 3 | 9 | 10,000 | **5,903,536.3** | 0.17 | 0.000% | **58.52x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 3 | 9 | 10,000 | **2,264,595.3** | 0.44 | 0.000% | **22.45x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 3 | 9 | 10,000 | **1,277.4** | 782.84 | 0.000% | **0.01x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 3 | 9 | 10,000 | **3,199,897.7** | 0.31 | 0.000% | **31.72x** |
| **PyMatching v2.4 (C++)** | PyMatching | 3 | 9 | 10,000 | **100,877.6** | 9.91 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 3 | 9 | 10,000 | **250,062.5** | 4.00 | 0.000% | **2.48x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 3 | 9 | 50,000 | **7,998,464.3** | 0.13 | 0.000% | **126.15x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 3 | 9 | 50,000 | **4,109,848.0** | 0.24 | 0.000% | **64.82x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 3 | 9 | 50,000 | **7,878,606.4** | 0.13 | 0.000% | **124.26x** |
| **PyMatching v2.4 (C++)** | PyMatching | 3 | 9 | 50,000 | **63,405.5** | 15.77 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 3 | 9 | 100,000 | **8,043,111.1** | 0.12 | 0.000% | **85.84x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 3 | 9 | 100,000 | **5,784,693.7** | 0.17 | 0.000% | **61.73x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 3 | 9 | 100,000 | **10,727,426.8** | 0.09 | 0.000% | **114.48x** |
| **PyMatching v2.4 (C++)** | PyMatching | 3 | 9 | 100,000 | **93,703.1** | 10.67 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 5 | 25 | 1,000 | **968,241.7** | 1.03 | 0.000% | **20.82x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 5 | 25 | 1,000 | **186,954.3** | 5.35 | 0.000% | **4.02x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 5 | 25 | 1,000 | **530.1** | 1886.54 | 0.000% | **0.01x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 5 | 25 | 1,000 | **268,175.6** | 3.73 | 0.000% | **5.77x** |
| **PyMatching v2.4 (C++)** | PyMatching | 5 | 25 | 1,000 | **46,501.9** | 21.50 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 5 | 25 | 1,000 | **126,614.3** | 7.90 | 0.000% | **2.72x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 5 | 25 | 5,000 | **1,555,645.4** | 0.64 | 0.000% | **19.09x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 5 | 25 | 5,000 | **607,208.8** | 1.65 | 0.000% | **7.45x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 5 | 25 | 5,000 | **529.1** | 1889.92 | 0.000% | **0.01x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 5 | 25 | 5,000 | **870,367.5** | 1.15 | 0.000% | **10.68x** |
| **PyMatching v2.4 (C++)** | PyMatching | 5 | 25 | 5,000 | **81,469.7** | 12.27 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 5 | 25 | 5,000 | **99,700.9** | 10.03 | 0.000% | **1.22x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 5 | 25 | 10,000 | **1,688,020.1** | 0.59 | 0.000% | **20.72x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 5 | 25 | 10,000 | **1,455,265.2** | 0.69 | 0.000% | **17.86x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 5 | 25 | 10,000 | **503.4** | 1986.39 | 0.000% | **0.01x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 5 | 25 | 10,000 | **2,038,195.8** | 0.49 | 0.000% | **25.01x** |
| **PyMatching v2.4 (C++)** | PyMatching | 5 | 25 | 10,000 | **81,483.0** | 12.27 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 5 | 25 | 10,000 | **114,817.2** | 8.71 | 0.000% | **1.41x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 5 | 25 | 50,000 | **5,261,219.5** | 0.19 | 0.000% | **69.74x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 5 | 25 | 50,000 | **2,005,615.7** | 0.50 | 0.000% | **26.59x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 5 | 25 | 50,000 | **3,199,467.6** | 0.31 | 0.000% | **42.41x** |
| **PyMatching v2.4 (C++)** | PyMatching | 5 | 25 | 50,000 | **75,437.5** | 13.26 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 5 | 25 | 100,000 | **6,155,626.5** | 0.16 | 0.000% | **66.46x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 5 | 25 | 100,000 | **2,989,161.3** | 0.33 | 0.000% | **32.27x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 5 | 25 | 100,000 | **3,202,756.9** | 0.31 | 0.000% | **34.58x** |
| **PyMatching v2.4 (C++)** | PyMatching | 5 | 25 | 100,000 | **92,622.6** | 10.80 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 7 | 49 | 1,000 | **526,509.8** | 1.90 | 0.000% | **14.47x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 7 | 49 | 1,000 | **313,755.0** | 3.19 | 0.000% | **8.62x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 7 | 49 | 1,000 | **467.0** | 2141.52 | 0.000% | **0.01x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 7 | 49 | 1,000 | **210,009.0** | 4.76 | 0.000% | **5.77x** |
| **PyMatching v2.4 (C++)** | PyMatching | 7 | 49 | 1,000 | **36,382.2** | 27.49 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 7 | 49 | 1,000 | **35,807.6** | 27.93 | 0.000% | **0.98x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 7 | 49 | 5,000 | **1,426,818.5** | 0.70 | 0.000% | **21.2x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 7 | 49 | 5,000 | **806,308.5** | 1.24 | 0.000% | **11.98x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 7 | 49 | 5,000 | **455.1** | 2197.16 | 0.000% | **0.01x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 7 | 49 | 5,000 | **1,062,789.6** | 0.94 | 0.000% | **15.79x** |
| **PyMatching v2.4 (C++)** | PyMatching | 7 | 49 | 5,000 | **67,317.4** | 14.85 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 7 | 49 | 5,000 | **31,000.5** | 32.26 | 0.000% | **0.46x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 7 | 49 | 10,000 | **1,673,444.1** | 0.60 | 0.000% | **20.97x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 7 | 49 | 10,000 | **871,938.4** | 1.15 | 0.000% | **10.92x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 7 | 49 | 10,000 | **480.9** | 2079.49 | 0.000% | **0.01x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 7 | 49 | 10,000 | **1,440,278.8** | 0.69 | 0.000% | **18.05x** |
| **PyMatching v2.4 (C++)** | PyMatching | 7 | 49 | 10,000 | **79,814.8** | 12.53 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 7 | 49 | 10,000 | **36,513.0** | 27.39 | 0.000% | **0.46x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 7 | 49 | 50,000 | **3,927,483.0** | 0.25 | 0.000% | **43.67x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 7 | 49 | 50,000 | **1,126,671.7** | 0.89 | 0.000% | **12.53x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 7 | 49 | 50,000 | **1,727,796.1** | 0.58 | 0.000% | **19.21x** |
| **PyMatching v2.4 (C++)** | PyMatching | 7 | 49 | 50,000 | **89,932.1** | 11.12 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 7 | 49 | 100,000 | **3,859,111.6** | 0.26 | 0.000% | **48.08x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 7 | 49 | 100,000 | **1,649,764.8** | 0.61 | 0.000% | **20.56x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 7 | 49 | 100,000 | **2,245,072.1** | 0.45 | 0.000% | **27.97x** |
| **PyMatching v2.4 (C++)** | PyMatching | 7 | 49 | 100,000 | **80,260.0** | 12.46 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 9 | 81 | 1,000 | **290,985.3** | 3.44 | 0.000% | **3.82x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 9 | 81 | 1,000 | **300,246.2** | 3.33 | 0.000% | **3.95x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 9 | 81 | 1,000 | **402.4** | 2484.96 | 0.000% | **0.01x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 9 | 81 | 1,000 | **336,394.5** | 2.97 | 0.000% | **4.42x** |
| **PyMatching v2.4 (C++)** | PyMatching | 9 | 81 | 1,000 | **76,077.4** | 13.14 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 9 | 81 | 1,000 | **8,156.5** | 122.60 | 0.000% | **0.11x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 9 | 81 | 5,000 | **1,096,683.7** | 0.91 | 0.000% | **13.06x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 9 | 81 | 5,000 | **453,831.7** | 2.20 | 0.000% | **5.41x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 9 | 81 | 5,000 | **362.6** | 2757.73 | 0.000% | **0.0x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 9 | 81 | 5,000 | **1,093,302.4** | 0.91 | 0.000% | **13.02x** |
| **PyMatching v2.4 (C++)** | PyMatching | 9 | 81 | 5,000 | **83,963.1** | 11.91 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 9 | 81 | 5,000 | **8,123.0** | 123.11 | 0.000% | **0.1x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 9 | 81 | 10,000 | **1,181,348.9** | 0.85 | 0.000% | **19.86x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 9 | 81 | 10,000 | **516,406.2** | 1.94 | 0.000% | **8.68x** |
| **QECTOR BP-OSD (CPU)** | QECTOR CPU | 9 | 81 | 10,000 | **407.5** | 2453.86 | 0.000% | **0.01x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 9 | 81 | 10,000 | **1,050,376.0** | 0.95 | 0.000% | **17.66x** |
| **PyMatching v2.4 (C++)** | PyMatching | 9 | 81 | 10,000 | **59,488.4** | 16.81 | 0.000% | **1.0x** |
| **ldpc v2.4.1 (BP-OSD)** | ldpc | 9 | 81 | 10,000 | **9,468.0** | 105.62 | 0.000% | **0.16x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 9 | 81 | 50,000 | **1,765,112.9** | 0.57 | 0.000% | **33.16x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 9 | 81 | 50,000 | **773,804.5** | 1.29 | 0.000% | **14.54x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 9 | 81 | 50,000 | **1,091,526.7** | 0.92 | 0.000% | **20.51x** |
| **PyMatching v2.4 (C++)** | PyMatching | 9 | 81 | 50,000 | **53,231.1** | 18.79 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 9 | 81 | 100,000 | **1,862,610.2** | 0.54 | 0.000% | **26.51x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 9 | 81 | 100,000 | **792,684.5** | 1.26 | 0.000% | **11.28x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 9 | 81 | 100,000 | **1,126,490.2** | 0.89 | 0.000% | **16.03x** |
| **PyMatching v2.4 (C++)** | PyMatching | 9 | 81 | 100,000 | **70,271.6** | 14.23 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 11 | 121 | 1,000 | **158,353.1** | 6.32 | 0.000% | **5.54x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 11 | 121 | 1,000 | **232,628.5** | 4.30 | 0.000% | **8.14x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 11 | 121 | 1,000 | **119,838.0** | 8.34 | 0.000% | **4.19x** |
| **PyMatching v2.4 (C++)** | PyMatching | 11 | 121 | 1,000 | **28,594.3** | 34.97 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 11 | 121 | 5,000 | **724,364.7** | 1.38 | 0.000% | **13.33x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 11 | 121 | 5,000 | **249,591.9** | 4.01 | 0.000% | **4.59x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 11 | 121 | 5,000 | **654,501.7** | 1.53 | 0.000% | **12.04x** |
| **PyMatching v2.4 (C++)** | PyMatching | 11 | 121 | 5,000 | **54,341.9** | 18.40 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 11 | 121 | 10,000 | **980,315.3** | 1.02 | 0.000% | **15.39x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 11 | 121 | 10,000 | **307,169.6** | 3.26 | 0.000% | **4.82x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 11 | 121 | 10,000 | **687,989.8** | 1.45 | 0.000% | **10.8x** |
| **PyMatching v2.4 (C++)** | PyMatching | 11 | 121 | 10,000 | **63,678.0** | 15.70 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 11 | 121 | 50,000 | **1,119,667.9** | 0.89 | 0.000% | **27.04x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 11 | 121 | 50,000 | **466,012.3** | 2.15 | 0.000% | **11.26x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 11 | 121 | 50,000 | **458,865.9** | 2.18 | 0.000% | **11.08x** |
| **PyMatching v2.4 (C++)** | PyMatching | 11 | 121 | 50,000 | **41,404.4** | 24.15 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 11 | 121 | 100,000 | **1,336,482.1** | 0.75 | 0.000% | **27.15x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 11 | 121 | 100,000 | **513,235.8** | 1.95 | 0.000% | **10.42x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 11 | 121 | 100,000 | **562,781.7** | 1.78 | 0.000% | **11.43x** |
| **PyMatching v2.4 (C++)** | PyMatching | 11 | 121 | 100,000 | **49,232.0** | 20.31 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 13 | 169 | 1,000 | **83,015.8** | 12.05 | 0.000% | **2.75x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 13 | 169 | 1,000 | **93,936.4** | 10.65 | 0.000% | **3.11x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 13 | 169 | 1,000 | **94,512.6** | 10.58 | 0.000% | **3.13x** |
| **PyMatching v2.4 (C++)** | PyMatching | 13 | 169 | 1,000 | **30,202.4** | 33.11 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 13 | 169 | 5,000 | **370,895.1** | 2.70 | 0.000% | **5.2x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 13 | 169 | 5,000 | **192,991.3** | 5.18 | 0.000% | **2.71x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 13 | 169 | 5,000 | **452,878.0** | 2.21 | 0.000% | **6.35x** |
| **PyMatching v2.4 (C++)** | PyMatching | 13 | 169 | 5,000 | **71,316.5** | 14.02 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 13 | 169 | 10,000 | **532,594.8** | 1.88 | 0.000% | **7.75x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 13 | 169 | 10,000 | **240,956.9** | 4.15 | 0.000% | **3.5x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 13 | 169 | 10,000 | **590,113.2** | 1.69 | 0.000% | **8.58x** |
| **PyMatching v2.4 (C++)** | PyMatching | 13 | 169 | 10,000 | **68,747.4** | 14.55 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 13 | 169 | 50,000 | **850,976.9** | 1.18 | 0.000% | **13.38x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 13 | 169 | 50,000 | **253,510.5** | 3.94 | 0.000% | **3.99x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 13 | 169 | 50,000 | **337,733.1** | 2.96 | 0.000% | **5.31x** |
| **PyMatching v2.4 (C++)** | PyMatching | 13 | 169 | 50,000 | **63,580.9** | 15.73 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 13 | 169 | 100,000 | **882,815.1** | 1.13 | 0.000% | **8.66x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 13 | 169 | 100,000 | **239,409.7** | 4.18 | 0.000% | **2.35x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 13 | 169 | 100,000 | **446,470.0** | 2.24 | 0.000% | **4.38x** |
| **PyMatching v2.4 (C++)** | PyMatching | 13 | 169 | 100,000 | **101,957.6** | 9.81 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 15 | 225 | 1,000 | **90,916.5** | 11.00 | 0.000% | **3.63x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 15 | 225 | 1,000 | **55,870.0** | 17.90 | 0.000% | **2.23x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 15 | 225 | 1,000 | **146,331.5** | 6.83 | 0.000% | **5.84x** |
| **PyMatching v2.4 (C++)** | PyMatching | 15 | 225 | 1,000 | **25,050.1** | 39.92 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 15 | 225 | 5,000 | **225,266.8** | 4.44 | 0.000% | **5.05x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 15 | 225 | 5,000 | **152,546.9** | 6.56 | 0.000% | **3.42x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 15 | 225 | 5,000 | **262,472.7** | 3.81 | 0.000% | **5.88x** |
| **PyMatching v2.4 (C++)** | PyMatching | 15 | 225 | 5,000 | **44,607.0** | 22.42 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 15 | 225 | 10,000 | **398,942.0** | 2.51 | 0.000% | **5.81x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 15 | 225 | 10,000 | **142,717.7** | 7.01 | 0.000% | **2.08x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 15 | 225 | 10,000 | **499,390.7** | 2.00 | 0.000% | **7.27x** |
| **PyMatching v2.4 (C++)** | PyMatching | 15 | 225 | 10,000 | **68,719.1** | 14.55 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 15 | 225 | 50,000 | **622,015.1** | 1.61 | 0.000% | **5.76x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 15 | 225 | 50,000 | **162,173.8** | 6.17 | 0.000% | **1.5x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 15 | 225 | 50,000 | **350,567.7** | 2.85 | 0.000% | **3.25x** |
| **PyMatching v2.4 (C++)** | PyMatching | 15 | 225 | 50,000 | **107,898.1** | 9.27 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 15 | 225 | 100,000 | **612,141.6** | 1.63 | 0.000% | **8.98x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 15 | 225 | 100,000 | **161,444.1** | 6.19 | 0.000% | **2.37x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 15 | 225 | 100,000 | **309,200.0** | 3.23 | 0.000% | **4.53x** |
| **PyMatching v2.4 (C++)** | PyMatching | 15 | 225 | 100,000 | **68,194.2** | 14.66 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 17 | 289 | 1,000 | **49,375.6** | 20.25 | 0.000% | **2.9x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 17 | 289 | 1,000 | **62,016.7** | 16.12 | 0.000% | **3.64x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 17 | 289 | 1,000 | **118,842.5** | 8.41 | 0.000% | **6.98x** |
| **PyMatching v2.4 (C++)** | PyMatching | 17 | 289 | 1,000 | **17,014.3** | 58.77 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 17 | 289 | 5,000 | **184,103.1** | 5.43 | 0.000% | **2.25x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 17 | 289 | 5,000 | **95,042.8** | 10.52 | 0.000% | **1.16x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 17 | 289 | 5,000 | **330,517.3** | 3.03 | 0.000% | **4.04x** |
| **PyMatching v2.4 (C++)** | PyMatching | 17 | 289 | 5,000 | **81,739.4** | 12.23 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 17 | 289 | 10,000 | **327,010.0** | 3.06 | 0.000% | **7.66x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 17 | 289 | 10,000 | **79,711.6** | 12.55 | 0.000% | **1.87x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 17 | 289 | 10,000 | **433,954.3** | 2.30 | 0.000% | **10.17x** |
| **PyMatching v2.4 (C++)** | PyMatching | 17 | 289 | 10,000 | **42,684.0** | 23.43 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 17 | 289 | 50,000 | **457,543.6** | 2.19 | 0.000% | **16.84x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 17 | 289 | 50,000 | **84,536.8** | 11.83 | 0.000% | **3.11x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 17 | 289 | 50,000 | **205,502.3** | 4.87 | 0.000% | **7.56x** |
| **PyMatching v2.4 (C++)** | PyMatching | 17 | 289 | 50,000 | **27,168.0** | 36.81 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 17 | 289 | 100,000 | **368,013.9** | 2.72 | 0.000% | **6.65x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 17 | 289 | 100,000 | **67,906.7** | 14.73 | 0.000% | **1.23x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 17 | 289 | 100,000 | **284,579.6** | 3.51 | 0.000% | **5.14x** |
| **PyMatching v2.4 (C++)** | PyMatching | 17 | 289 | 100,000 | **55,328.1** | 18.07 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 19 | 361 | 1,000 | **43,884.5** | 22.79 | 0.000% | **5.38x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 19 | 361 | 1,000 | **43,272.1** | 23.11 | 0.000% | **5.31x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 19 | 361 | 1,000 | **74,540.6** | 13.42 | 0.000% | **9.15x** |
| **PyMatching v2.4 (C++)** | PyMatching | 19 | 361 | 1,000 | **8,150.6** | 122.69 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 19 | 361 | 5,000 | **84,951.2** | 11.77 | 0.000% | **1.8x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 19 | 361 | 5,000 | **49,924.8** | 20.03 | 0.000% | **1.06x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 19 | 361 | 5,000 | **230,892.5** | 4.33 | 0.000% | **4.89x** |
| **PyMatching v2.4 (C++)** | PyMatching | 19 | 361 | 5,000 | **47,259.0** | 21.16 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 19 | 361 | 10,000 | **148,528.4** | 6.73 | 0.000% | **6.38x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 19 | 361 | 10,000 | **58,605.0** | 17.06 | 0.000% | **2.52x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 19 | 361 | 10,000 | **337,474.1** | 2.96 | 0.000% | **14.51x** |
| **PyMatching v2.4 (C++)** | PyMatching | 19 | 361 | 10,000 | **23,263.9** | 42.98 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 19 | 361 | 50,000 | **298,036.1** | 3.36 | 0.000% | **6.19x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 19 | 361 | 50,000 | **66,733.4** | 14.98 | 0.000% | **1.39x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 19 | 361 | 50,000 | **175,751.9** | 5.69 | 0.000% | **3.65x** |
| **PyMatching v2.4 (C++)** | PyMatching | 19 | 361 | 50,000 | **48,181.2** | 20.75 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 19 | 361 | 100,000 | **357,352.5** | 2.80 | 0.000% | **4.89x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 19 | 361 | 100,000 | **67,008.6** | 14.92 | 0.000% | **0.92x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 19 | 361 | 100,000 | **163,862.1** | 6.10 | 0.000% | **2.24x** |
| **PyMatching v2.4 (C++)** | PyMatching | 19 | 361 | 100,000 | **73,019.4** | 13.69 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 21 | 441 | 1,000 | **37,574.2** | 26.61 | 0.000% | **4.59x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 21 | 441 | 1,000 | **29,881.4** | 33.47 | 0.000% | **3.65x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 21 | 441 | 1,000 | **45,595.3** | 21.93 | 0.000% | **5.56x** |
| **PyMatching v2.4 (C++)** | PyMatching | 21 | 441 | 1,000 | **8,194.4** | 122.04 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 21 | 441 | 5,000 | **65,058.0** | 15.37 | 0.000% | **1.14x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 21 | 441 | 5,000 | **42,781.8** | 23.37 | 0.000% | **0.75x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 21 | 441 | 5,000 | **312,431.7** | 3.20 | 0.000% | **5.47x** |
| **PyMatching v2.4 (C++)** | PyMatching | 21 | 441 | 5,000 | **57,159.2** | 17.49 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 21 | 441 | 10,000 | **128,637.2** | 7.77 | 0.000% | **2.21x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 21 | 441 | 10,000 | **58,702.9** | 17.03 | 0.000% | **1.01x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 21 | 441 | 10,000 | **346,804.4** | 2.88 | 0.000% | **5.95x** |
| **PyMatching v2.4 (C++)** | PyMatching | 21 | 441 | 10,000 | **58,326.1** | 17.14 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 21 | 441 | 50,000 | **354,371.7** | 2.82 | 0.000% | **6.0x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 21 | 441 | 50,000 | **60,379.6** | 16.56 | 0.000% | **1.02x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 21 | 441 | 50,000 | **160,213.1** | 6.24 | 0.000% | **2.71x** |
| **PyMatching v2.4 (C++)** | PyMatching | 21 | 441 | 50,000 | **59,049.3** | 16.94 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 21 | 441 | 100,000 | **318,659.5** | 3.14 | 0.000% | **5.58x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 21 | 441 | 100,000 | **57,306.2** | 17.45 | 0.000% | **1.0x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 21 | 441 | 100,000 | **42,521.6** | 23.52 | 0.000% | **0.74x** |
| **PyMatching v2.4 (C++)** | PyMatching | 21 | 441 | 100,000 | **57,126.5** | 17.50 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 25 | 625 | 1,000 | **30,981.9** | 32.28 | 0.000% | **4.6x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 25 | 625 | 1,000 | **19,570.0** | 51.10 | 0.000% | **2.91x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 25 | 625 | 1,000 | **31,202.8** | 32.05 | 0.000% | **4.64x** |
| **PyMatching v2.4 (C++)** | PyMatching | 25 | 625 | 1,000 | **6,730.2** | 148.58 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 25 | 625 | 5,000 | **45,270.8** | 22.09 | 0.000% | **1.23x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 25 | 625 | 5,000 | **23,648.8** | 42.29 | 0.000% | **0.64x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 25 | 625 | 5,000 | **165,830.9** | 6.03 | 0.000% | **4.49x** |
| **PyMatching v2.4 (C++)** | PyMatching | 25 | 625 | 5,000 | **36,907.2** | 27.09 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 25 | 625 | 10,000 | **79,828.7** | 12.53 | 0.000% | **2.68x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 25 | 625 | 10,000 | **24,143.6** | 41.42 | 0.000% | **0.81x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 25 | 625 | 10,000 | **176,222.9** | 5.67 | 0.000% | **5.91x** |
| **PyMatching v2.4 (C++)** | PyMatching | 25 | 625 | 10,000 | **29,806.3** | 33.55 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 25 | 625 | 50,000 | **185,145.2** | 5.40 | 0.000% | **3.32x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 25 | 625 | 50,000 | **25,988.5** | 38.48 | 0.000% | **0.47x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 25 | 625 | 50,000 | **93,461.1** | 10.70 | 0.000% | **1.67x** |
| **PyMatching v2.4 (C++)** | PyMatching | 25 | 625 | 50,000 | **55,850.3** | 17.90 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 25 | 625 | 100,000 | **197,734.8** | 5.06 | 0.000% | **6.67x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 25 | 625 | 100,000 | **23,603.7** | 42.37 | 0.000% | **0.8x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 25 | 625 | 100,000 | **24,321.4** | 41.12 | 0.000% | **0.82x** |
| **PyMatching v2.4 (C++)** | PyMatching | 25 | 625 | 100,000 | **29,651.6** | 33.73 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 31 | 961 | 1,000 | **8,583.3** | 116.51 | 0.000% | **3.31x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 31 | 961 | 1,000 | **3,086.6** | 323.98 | 0.000% | **1.19x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 31 | 961 | 1,000 | **23,648.2** | 42.29 | 0.000% | **9.11x** |
| **PyMatching v2.4 (C++)** | PyMatching | 31 | 961 | 1,000 | **2,596.8** | 385.09 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 31 | 961 | 5,000 | **2,082.8** | 480.13 | 0.000% | **0.07x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 31 | 961 | 5,000 | **4,198.6** | 238.18 | 0.000% | **0.14x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 31 | 961 | 5,000 | **87,219.4** | 11.47 | 0.000% | **2.84x** |
| **PyMatching v2.4 (C++)** | PyMatching | 31 | 961 | 5,000 | **30,717.2** | 32.56 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 31 | 961 | 10,000 | **8,551.7** | 116.94 | 0.000% | **0.19x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 31 | 961 | 10,000 | **4,596.3** | 217.57 | 0.000% | **0.1x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 31 | 961 | 10,000 | **34,057.9** | 29.36 | 0.000% | **0.77x** |
| **PyMatching v2.4 (C++)** | PyMatching | 31 | 961 | 10,000 | **44,072.3** | 22.69 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 31 | 961 | 50,000 | **24,144.3** | 41.42 | 0.000% | **0.56x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 31 | 961 | 50,000 | **5,268.6** | 189.80 | 0.000% | **0.12x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 31 | 961 | 50,000 | **21,178.1** | 47.22 | 0.000% | **0.5x** |
| **PyMatching v2.4 (C++)** | PyMatching | 31 | 961 | 50,000 | **42,771.6** | 23.38 | 0.000% | **1.0x** |
| **QECTOR Fast Union-Find (CPU)** | QECTOR CPU | 31 | 961 | 100,000 | **86,928.9** | 11.50 | 0.000% | **2.15x** |
| **QECTOR Blossom (CPU)** | QECTOR CPU | 31 | 961 | 100,000 | **6,780.2** | 147.49 | 0.000% | **0.17x** |
| **QECTOR CUDA Batch (GPU)** | QECTOR GPU | 31 | 961 | 100,000 | **21,502.9** | 46.51 | 0.000% | **0.53x** |
| **PyMatching v2.4 (C++)** | PyMatching | 31 | 961 | 100,000 | **40,436.7** | 24.73 | 0.000% | **1.0x** |

## Methodology & Integrity Notes

> Every decoder is measured on the same Stim circuit, the same detector samples, the same DEM and the same observable scoring (ler.estimate_ler_circuit_level). Only the decoder varies between rows. Rows are validated by ler.assert_comparable before being written.
