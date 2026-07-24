# Belief-matching head-to-head — QECTOR vs PyMatching (circuit-level)

- `surface_code:rotated_memory_x`, rounds=d, circuit noise p=0.005
- shots/point: 4000; Wilson 95% intervals
- AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD; Stim 1.16.0; PyMatching 2.4.0

| d | shots | PyMatching LER | QECTOR-MWPM LER | QECTOR-belief LER | LER reduction |
|---|-------|----------------|-----------------|-------------------|---------------|
| 3 | 4000 | 0.0127 [0.0097,0.0167] | 0.0127 | 0.0112 [0.0084,0.0150] | 11.8% |
| 5 | 4000 | 0.0090 [0.0065,0.0124] | 0.0090 | 0.0075 [0.0053,0.0107] | 16.7% |
| 7 | 4000 | 0.0070 [0.0048,0.0101] | 0.0070 | 0.0032 [0.0019,0.0056] | 53.6% |
