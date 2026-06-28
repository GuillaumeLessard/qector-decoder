# Stim circuit-level head-to-head — QECTOR vs PyMatching

- Task: `surface_code:rotated_memory_x`, rounds = distance, circuit-level depolarizing+measurement+reset noise p = 0.005
- Shots per point: 2000
- CPU: AMD64 Family 22 Model 48 Stepping 1, AuthenticAMD; Python 3.11.0; NumPy 2.2.6; Stim 1.16.0; PyMatching 2.4.0

LER with Wilson 95% interval; latency is per-shot decode time (hot path).

| d | rounds | raw mech | collapsed edges | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | QB µs/shot | PM µs/shot | UF µs/shot |
|---|--------|----------|-----------------|--------------------|----------------|---------------|-----------|-----------|-----------|
| 3 | 3 | 556 | 78 | 0.0130 [0.0089,0.0190] | 0.0130 [0.0089,0.0190] | 0.0145 | 7.6 | 2.0 | 3.0 |
| 5 | 5 | 3706 | 502 | 0.0080 [0.0049,0.0130] | 0.0080 [0.0049,0.0130] | 0.0380 | 71.4 | 12.4 | 13.2 |
| 7 | 7 | 11974 | 1558 | 0.0045 [0.0024,0.0085] | 0.0045 [0.0024,0.0085] | 0.0170 | 456.5 | 40.2 | 47.2 |
