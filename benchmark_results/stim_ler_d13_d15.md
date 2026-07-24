# Stim circuit-level head-to-head — QECTOR vs PyMatching

- Task: `surface_code:rotated_memory_x`, rounds = distance, circuit-level depolarizing+measurement+reset noise p = 0.005
- Shots per point: 20000
- CPU: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD; Python 3.11.0; NumPy 2.2.6; Stim 1.16.0; PyMatching 2.4.0

LER with Wilson 95% interval; latency is per-shot decode time (hot path).

| d | rounds | raw mech | collapsed edges | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | QB µs/shot | PM µs/shot | UF µs/shot |
|---|--------|----------|-----------------|--------------------|----------------|---------------|-----------|-----------|-----------|
| 13 | 13 | 85005 | 11398 | 0.0008 [0.0005,0.0012] | 0.0008 [0.0005,0.0012] | 0.0129 | 820.5 | 81.1 | 52.6 |
| 15 | 15 | 132426 | 17862 | 0.0005 [0.0003,0.0009] | 0.0005 [0.0003,0.0009] | 0.0103 | 1965.1 | 203.2 | 89.8 |
