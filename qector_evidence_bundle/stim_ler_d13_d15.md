# Stim circuit-level head-to-head — QECTOR vs PyMatching

- Task: `surface_code:rotated_memory_x`, rounds = distance, circuit-level depolarizing+measurement+reset noise p = 0.005
- Shots per point: 2000
- CPU: AMD64 Family 22 Model 48 Stepping 1, AuthenticAMD; Python 3.11.0; NumPy 2.2.6; Stim 1.16.0; PyMatching 2.4.0

LER with Wilson 95% interval; latency is per-shot decode time (hot path).

| d | rounds | raw mech | collapsed edges | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | QB µs/shot | PM µs/shot | UF µs/shot |
|---|--------|----------|-----------------|--------------------|----------------|---------------|-----------|-----------|-----------|
| 9 | 9 | 26823 | 3534 | 0.0040 [0.0020,0.0079] | 0.0040 [0.0020,0.0079] | 0.0230 | 1371.9 | 110.0 | 117.1 |
| 11 | 11 | 50484 | 6718 | 0.0005 [0.0001,0.0028] | 0.0005 [0.0001,0.0028] | 0.0145 | 3964.0 | 246.0 | 256.2 |
