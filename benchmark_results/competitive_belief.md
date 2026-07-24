# Belief-matching head-to-head — QECTOR vs PyMatching (circuit-level)

- `surface_code:rotated_memory_x`, rounds=d, circuit noise p=0.005
- shots/point: 3000; Wilson 95% intervals
- AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD; Stim 1.16.0; PyMatching 2.4.0

| d | shots | PyMatching LER | QECTOR-MWPM LER | QECTOR-belief LER | LER reduction |
|---|-------|----------------|-----------------|-------------------|---------------|
| 3 | 3000 | 0.0117 [0.0084,0.0162] | 0.0117 | 0.0123 [0.0090,0.0170] | -5.7% |
| 5 | 3000 | 0.0077 [0.0051,0.0115] | 0.0077 | 0.0050 [0.0030,0.0082] | 34.8% |
| 7 | 3000 | 0.0060 [0.0038,0.0095] | 0.0060 | 0.0030 [0.0016,0.0057] | 50.0% |
