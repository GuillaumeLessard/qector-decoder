# Stim circuit-level head-to-head — QECTOR vs PyMatching

- Task: `surface_code:rotated_memory_x`, rounds = distance, circuit-level depolarizing+measurement+reset noise p = 0.005
- Shots per point: 10000
- CPU: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD; Python 3.11.0; NumPy 2.2.6; Stim 1.16.0; PyMatching 2.4.0

LER with Wilson 95% interval; latency is per-shot decode time (hot path).

| d | rounds | raw mech | collapsed edges | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | QB µs/shot | PM µs/shot | UF µs/shot |
|---|--------|----------|-----------------|--------------------|----------------|---------------|-----------|-----------|-----------|
| 3 | 3 | 568 | 78 | 0.0122 [0.0102,0.0145] | 0.0122 [0.0102,0.0145] | 0.0145 | 0.6 | 0.5 | 0.4 |
| 5 | 5 | 3718 | 502 | 0.0084 [0.0068,0.0104] | 0.0082 [0.0066,0.0102] | 0.0430 | 7.1 | 3.5 | 2.1 |
| 7 | 7 | 11982 | 1558 | 0.0041 [0.0030,0.0056] | 0.0041 [0.0030,0.0056] | 0.0230 | 42.3 | 12.0 | 6.6 |
| 9 | 9 | 26823 | 3534 | 0.0028 [0.0019,0.0040] | 0.0028 [0.0019,0.0040] | 0.0185 | 115.4 | 26.8 | 15.4 |
| 11 | 11 | 50484 | 6718 | 0.0015 [0.0009,0.0025] | 0.0015 [0.0009,0.0025] | 0.0162 | 309.4 | 39.7 | 29.5 |
