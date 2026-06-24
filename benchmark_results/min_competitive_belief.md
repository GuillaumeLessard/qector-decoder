# Belief-matching head-to-head — QECTOR vs PyMatching (circuit-level)

- `surface_code:rotated_memory_x`, rounds=d, circuit noise p=0.005
- shots/point: 20000; Wilson 95% intervals
- AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD; Stim 1.16.0; PyMatching 2.4.0

| d | shots | PyMatching LER | QECTOR-MWPM LER | QECTOR-belief LER | LER reduction  ref-belief LER |
|---|-------|----------------|-----------------|-------------------|-------------------------------|
| 3 | 20000 | 0.0112 [0.0099,0.0128] | 0.0112 | 0.0109 [0.0096,0.0125] | 2.7%  0.0098 |
| 5 | 20000 | 0.0084 [0.0073,0.0098] | 0.0084 | 0.0056 [0.0047,0.0067] | 33.7%  0.0054 |
| 7 | 20000 | 0.0053 [0.0043,0.0064] | 0.0054 | 0.0039 [0.0031,0.0049] | 25.7%  0.0035 |
