# Memory & Scaling Analysis

Latency is not the whole story — large-distance QEC workloads usually hit memory
scaling before raw compute. This document describes how to measure scaling with the
shipped tools; like the latency numbers, the actual figures are generated on your
hardware rather than baked in.

## What scales

For a matching code of distance `d`:

| Quantity | Where it comes from |
|----------|---------------------|
| Checks (detectors) | rows of `H` — `code.n_checks` |
| Qubits (edges) | columns of `H` — `code.n_qubits` |
| Edges in the matching graph | `len(check_to_edges(code.check_to_qubits))` |
| Peak Python allocation per decode | `benchmark_decoder(...)["peak_python_alloc_kib"]` |

`rotated_surface_code(d)` has `d²` qubits; `toric_code(L)` has `2L²`;
`unrotated_surface_code(d)` has `2d(d−1)`. These are exact and printed by the
benchmark CSV (`n_qubits`, `n_checks` columns).

## Measuring the empirical scaling slope

Run the driver across a distance sweep and fit latency / memory vs problem size:

```bash
python scripts/run_competitive_benchmark.py \
    --code rotated_surface --distances 3 5 7 9 11 13 \
    --decoders blossom sparse_blossom union_find \
    --trials 3000 --warmup 300 --out benchmark_results/scaling
```

Then fit a power law to the CSV (`lat_median_us` vs `n_qubits`):

```python
import csv, numpy as np
rows = list(csv.DictReader(open("benchmark_results/scaling.csv")))
for dec in {r["decoder"] for r in rows}:
    pts = [(int(r["n_qubits"]), float(r["lat_median_us"])) for r in rows if r["decoder"] == dec]
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    slope = np.polyfit(x, y, 1)[0]
    print(f"{dec}: latency ~ n_qubits^{slope:.2f}")
```

Do the same with `peak_python_alloc_kib` to get the memory slope.

## Peak VRAM (GPU)

Native GPU memory is not visible to `tracemalloc`. Measure VRAM with the vendor tools
while a large-batch GPU benchmark runs:

```bash
# NVIDIA / CUDA
nvidia-smi --query-gpu=memory.used --format=csv -l 1
# while running, e.g.:
python -c "import numpy as np; from qector_decoder_v3 import codes, CUDABatchDecoder; \
c=codes.rotated_surface_code(11); d=CUDABatchDecoder(c.check_to_qubits,c.n_qubits); \
s=(np.random.default_rng(0).random((1<<16,c.n_checks))<0.08).astype('uint8'); d.batch_decode(s)"
```

For OpenCL, use `clinfo` for device limits and the platform profiler for live usage.

## Backend crossover (where GPU starts winning)

`AutoDecoder.calibrate()` measures the CPU↔GPU crossover on the current machine and
records the per-size timings used to choose it:

```python
from qector_decoder_v3 import codes
from qector_decoder_v3.backend import AutoDecoder

code = codes.rotated_surface_code(9)
dec = AutoDecoder(code.check_to_qubits, code.n_qubits)
info = dec.calibrate(sizes=(64, 256, 1024, 4096, 16384, 65536))
print(info["crossover"], "= smallest batch where GPU beats Rayon CPU")
print(info["timings"])      # full CPU vs GPU timing table
```

If the GPU never wins (small codes, modest batches), calibration disables GPU and the
diagnostics say so — no silent slowdowns.

## Allocation behaviour

`FastUnionFindDecoder` uses pre-allocated reusable buffers (a near-zero-allocation hot
path). Compare `peak_python_alloc_kib` across decoders in the benchmark CSV to see the
allocation cost of each path; the Python wrapper adds only the output NumPy array.
