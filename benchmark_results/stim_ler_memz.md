# Stim circuit-level head-to-head — QECTOR vs PyMatching

- Task: `surface_code:rotated_memory_x`, rounds = distance, circuit-level depolarizing+measurement+reset noise p = 0.005
- Shots per point: 40000
- CPU: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD; Python 3.11.0; NumPy 2.2.6; Stim 1.16.0; PyMatching 2.4.0

LER with Wilson 95% interval; latency is per-shot decode time (hot path).

| d | rounds | raw mech | collapsed edges | QECTOR-Blossom LER | PyMatching LER | QECTOR-UF LER | QB µs/shot | PM µs/shot | UF µs/shot |
|---|--------|----------|-----------------|--------------------|----------------|---------------|-----------|-----------|-----------|
| 3 | 3 | 556 | 78 | 0.0115 [0.0105,0.0126] | 0.0115 [0.0105,0.0126] | 0.0137 | 0.5 | 0.4 | 0.4 |
| 5 | 5 | 3706 | 502 | 0.0067 [0.0059,0.0075] | 0.0067 [0.0059,0.0075] | 0.0348 | 7.3 | 3.2 | 2.2 |
| 7 | 7 | 11974 | 1558 | 0.0038 [0.0032,0.0044] | 0.0038 [0.0032,0.0045] | 0.0169 | 41.3 | 11.0 | 6.9 |
| 9 | 9 | 26815 | 3534 | 0.0020 [0.0016,0.0025] | 0.0021 [0.0017,0.0026] | 0.0132 | 116.9 | 29.3 | 15.6 |
| 11 | 11 | 50476 | 6718 | 0.0011 [0.0008,0.0015] | 0.0011 [0.0008,0.0015] | 0.0099 | 344.1 | 58.1 | 30.2 |
