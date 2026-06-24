# GPU Batch Benchmark Notes

QECTOR v3 includes two GPU batch paths:

- `CUDABatchDecoder` for NVIDIA systems through NVRTC + CUDA Driver API.
- `OpenCLBatchDecoder` for NVIDIA, AMD, and Intel systems through OpenCL.

Both GPU paths are intended for **offline batch simulation / Monte-Carlo sweeps**, not
real-time hardware control. Small batches can be slower than CPU because transfer and
kernel-launch overhead dominate. Use `AutoDecoder.calibrate()` or local benchmark
scripts before quoting a speedup.

## Valid public claims

The current repository supports these claims:

- CUDA and OpenCL batch outputs are tested for bit-identity against CPU paths on the
  supported configurations in the test suite.
- GPU paths have CPU fallback / degraded-mode behavior for runtime failures.
- Checked-in reference artifacts can be reviewed under `benchmark_results/`, including
  memory profiling for `cpu_batch`, `blossom`, `fast_union_find`, and `cuda_batch`.
- Throughput is hardware-specific and must be regenerated on the buyer/reviewer machine
  before sales or scientific claims are made.

## Reference memory artifact

The checked-in `benchmark_results/native_memory.json` reference artifact records:

- Windows 10 x64
- Python 3.11.0
- Rust 1.96.0
- QECTOR 0.5.0
- CUDA available: true
- OpenCL available: true
- batch size: 16384
- distances: 5, 9, 13

For d=13, batch 16384, the artifact records approximately:

| Decoder | RSS base MiB | RSS peak MiB | Native delta MiB |
|---|---:|---:|---:|
| `cpu_batch` | 120.98 | 130.39 | 9.41 |
| `blossom` | 123.64 | 129.52 | 5.88 |
| `fast_union_find` | 121.98 | 122.00 | 0.02 |
| `cuda_batch` | 211.57 | 214.24 | 2.67 |

This is a memory/fidelity artifact, not a universal speed claim.

## Run locally

```bash
python scripts/native_memory_profile.py --distances 5 9 13 --batch 16384 --out benchmark_results/native_memory
python scripts/auto_backend_calibrate.py --out benchmark_results/auto_backend_calibration
```

For commercial evaluation, include the generated JSON/CSV files, environment block,
CUDA/OpenCL runtime versions, driver version, GPU model, batch sizes, seeds, command
line, wall time, memory profile, and output hashes.

## Forbidden until regenerated and documented

Do not claim any of the following unless the repo contains a reproducible artifact for
that exact machine/workload:

- universal GPU superiority over CPU
- universal speedup over PyMatching
- RTX 4090 or any other specific GPU speedup not present in the artifact
- SaaS readiness
- OEM readiness
- real-time hardware decoding within coherence-time constraints
