# Stim circuit-level head-to-head — QECTOR vs PyMatching

- Task: `surface_code:rotated_memory_x`, rounds = distance, circuit-level depolarizing+measurement+reset noise p = 0.005
- Shots per point: 40000
- CPU: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD; Python 3.11.0; NumPy 2.2.6; Stim 1.16.0; PyMatching 2.4.0

LER with Wilson 95% interval; latency is per-shot decode time (hot path).

| d | rounds | raw mech | collapsed edges | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | QB µs/shot | PM µs/shot | UF µs/shot |
|---|--------|----------|-----------------|--------------------|----------------|---------------|-----------|-----------|-----------|
| 3 | 3 | 568 | 78 | 0.0123 [0.0112,0.0134] | 0.0123 [0.0112,0.0134] | 0.0139 | 0.6 | 0.4 | 0.5 |
| 5 | 5 | 3718 | 502 | 0.0089 [0.0080,0.0099] | 0.0089 [0.0080,0.0099] | 0.0405 | 7.4 | 2.7 | 2.1 |
| 7 | 7 | 11982 | 1558 | 0.0048 [0.0042,0.0056] | 0.0047 [0.0041,0.0054] | 0.0209 | 41.9 | 8.7 | 6.5 |
| 9 | 9 | 26823 | 3534 | 0.0027 [0.0023,0.0033] | 0.0027 [0.0023,0.0033] | 0.0208 | 120.1 | 21.4 | 16.2 |
| 11 | 11 | 50484 | 6718 | 0.0021 [0.0017,0.0026] | 0.0021 [0.0017,0.0026] | 0.0168 | 338.8 | 41.9 | 30.7 |
