# QECTOR Decoder v3 — Environmental & Hardware Fallbacks

This document formally records environmental boundaries, hardware fallbacks, and feature availability logic in QECTOR Decoder v3.

---

## 1. OpenCL Support & Distribution Policy

- **Wheel Distribution**: PyPI pre-built wheels (`qector-decoder-v3`) ship with **CUDA feature enabled** (`--no-default-features --features cuda`) for NVIDIA GPUs. OpenCL is **not** linked in pre-built wheels because build runners lack OpenCL import libraries at link time.
- **Runtime Behavior**: Calling `OpenCLBatchDecoder` on a standard PyPI wheel raises a clear `RuntimeError`:
  ```
  RuntimeError: OpenCLBatchDecoder not available in CUDA-only wheel. Build from source: maturin develop --features opencl
  ```
- **Source Build**: Developers with OpenCL devices can enable the backend via:
  ```bash
  maturin develop --release --features opencl
  ```

---

## 2. CUDA Tiers & License Enforcement

- **Hardware Probe**: `gpu_backend.gpu_available()` and `CUDABatchDecoder.is_available()` probe live NVIDIA GPUs via CuPy / CUDA runtime.
- **License Gating**: Instantiating `CUDABatchDecoder` requires **Pro or Enterprise tier**. In Community tier (no key configured), `CUDABatchDecoder` raises `PermissionError`:
  ```
  PermissionError: GPU decoding requires at least Pro tier (current: Community). Upgrade at https://qector.store/pricing
  ```
- **Unweighted GPU Mode Warning**: When `CUDABatchDecoder` is constructed without `edge_weights` on a weighted DEM path, a `UserWarning` is emitted warning that unweighted cluster growth costs higher LER on circuit noise.

---

## 3. CUDA BP-OSD Batch & Single-Shot Semantics

- **Batch Focus**: `CUDABpOsdDecoder` is optimized for high-throughput batched BP-OSD on non-graphlike / qLDPC codes across large shot arrays.
- **Single-Shot**: `CUDABpOsdDecoder.decode(syndrome)` accepts a 1-D syndrome of length `n_checks` and returns a 1-D correction, implemented natively as a one-row batch (added in the v1.0 cycle — earlier builds exposed only `batch_decode`). For latency-critical single shots, prefer the CPU `BpOsdDecoder` / `MatrixBPOSDDecoder`; the CUDA kernel's fixed launch overhead only amortizes across batches.

---

## 4. Known Limitation: Two CUDA Contexts (Rust driver-API + CuPy runtime)

The native `CUDABatchDecoder`/`CUDABpOsdDecoder` create their own CUDA **driver-API** context (`cuCtxCreate_v2`), while CuPy uses the runtime API's **primary** context. Two contexts on one device means device memory allocated in one is not valid in the other, and under a *single process that exercises both GPU paths at once under load* (e.g. running the entire pytest suite with the GPU visible and an Enterprise key), an intermittent native access-violation in `cupy.asarray` has been observed (observed 2026-08-04, GTX 1660 Ti, cupy 14.1.1).

- **Not reproducible** in isolated/single-path runs: the GPU-native tests pass standalone (94/94 under Enterprise) and the full suite passes with the device hidden (`CUDA_VISIBLE_DEVICES=-1`).
- **Workaround**: run GPU-native and CuPy-batched-BP workloads in separate processes, or hide the device for the monolithic suite run.
- **Proper fix (not yet done — touches the working CUDA path)**: share the primary context via `cuDevicePrimaryCtxRetain`/`cuDevicePrimaryCtxRelease` instead of `cuCtxCreate_v2`/`cuCtxDestroy` in `src/cuda_batch.rs` / `src/cuda_bp_osd.rs`. This is a candidate v1.x robustness upgrade; it is deliberately **not** done unilaterally because the current dual-stream/pinned path is field-proven and the fix changes the context lifecycle.
