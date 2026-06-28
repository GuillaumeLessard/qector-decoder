# Stim circuit-level head-to-head — QECTOR vs PyMatching

- Task: `surface_code:rotated_memory_x`, rounds = distance, circuit-level depolarizing+measurement+reset noise p = 0.005
- Shots per point: 2000
- CPU: AMD64 Family 22 Model 48 Stepping 1, AuthenticAMD; Python 3.11.0; NumPy 2.2.6; Stim 1.16.0; PyMatching 2.4.0

LER with Wilson 95% interval; latency is per-shot decode time (hot path).

| d | rounds | raw mech | collapsed edges | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | QB µs/shot | PM µs/shot | UF µs/shot |
|---|--------|----------|-----------------|--------------------|----------------|---------------|-----------|-----------|-----------|
| 3 | 3 | 568 | 78 | 0.0110 [0.0073,0.0166] | 0.0110 [0.0073,0.0166] | 0.0120 | 5.6 | 2.0 | 7.5 |
| 5 | 5 | 3718 | 502 | 0.0075 [0.0046,0.0123] | 0.0075 [0.0046,0.0123] | 0.0430 | 78.7 | 12.7 | 13.6 |
| 7 | 7 | 11982 | 1558 | 0.0065 [0.0038,0.0111] | 0.0060 [0.0034,0.0105] | 0.0210 | 499.1 | 41.9 | 53.9 |
