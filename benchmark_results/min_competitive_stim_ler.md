# Stim circuit-level head-to-head — QECTOR vs PyMatching

- Task: `surface_code:rotated_memory_x`, rounds = distance, circuit-level depolarizing+measurement+reset noise p = 0.005
- Shots per point: 40000
- CPU: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD; Python 3.11.0; NumPy 2.2.6; Stim 1.16.0; PyMatching 2.4.0

LER with Wilson 95% interval; latency is per-shot decode time (hot path).

| d | rounds | raw mech | collapsed edges | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | QB µs/shot | PM µs/shot | UF µs/shot |
|---|--------|----------|-----------------|--------------------|----------------|---------------|-----------|-----------|-----------|
| 3 | 3 | 568 | 78 | 0.0124 [0.0114,0.0135] | 0.0124 [0.0114,0.0135] | 0.0139 | 0.5 | 0.4 | 0.4 |
| 5 | 5 | 3718 | 502 | 0.0090 [0.0081,0.0099] | 0.0089 [0.0080,0.0099] | 0.0408 | 7.3 | 2.8 | 2.1 |
| 7 | 7 | 11982 | 1558 | 0.0049 [0.0043,0.0057] | 0.0049 [0.0043,0.0056] | 0.0208 | 43.1 | 9.1 | 6.8 |
| 9 | 9 | 26823 | 3534 | 0.0032 [0.0027,0.0038] | 0.0033 [0.0028,0.0039] | 0.0209 | 110.5 | 22.5 | 16.0 |
| 11 | 11 | 50484 | 6718 | 0.0017 [0.0013,0.0022] | 0.0015 [0.0012,0.0020] | 0.0161 | 243.8 | 43.7 | 30.6 |
