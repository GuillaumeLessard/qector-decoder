# QECTOR Decoder v3 — Installation Guide (v1.0.0)

> Audience: users installing `qector-decoder-v3` 1.0.0 from PyPI.

---

## 1. PyPI wheels (recommended)

QECTOR publishes **deterministic binary wheels only** (no sdist).

```bash
pip install qector-decoder-v3        # CPython 3.9–3.13, any platform (binary wheels)
pip install "qector-decoder-v3[stim]" # + Stim / Sinter / PyMatching / LDPC
pip install "qector-decoder-v3[bench]"# + psutil, matplotlib, scipy for the harness
pip install "qector-decoder-v3[all]"  # full validation env (stim + bench + pytest)
```

15 binary wheels per release: `cp39..cp313 × manylinux_2_17_x86_64 / win_amd64 / macosx_11_0_arm64`.
`pip install` on any other platform will fail (no sdist fallback on purpose: see
`docs/RELEASING.md`).

Verify:
```bash
python -c "import qector_decoder_v3 as q; print(q.__version__)"
qector-doctor            # prints native-core status + license tier
```

## 2. Optional backends

| Backend | Requirement | Detection | What it accelerates |
|---|---|---|---|
| **CUDA Rust (native)** | CUDA Toolkit + `libloading` (bundled) | `q.cuda_is_available()` | `CUDABatchDecoder`, `CUDABpOsdDecoder` |
| **OpenCL (auto, off by default)** | OpenCL runtime | `q.opencl_is_available()` | `OpenCLBatchDecoder` |
| **Belief-matching** | `pymatching` + `stim` | `q.dem.from_stim` path | reference accuracy |

CUDA auto-routing is OFF unless `QECTOR_ENABLE_CUDA_AUTO=1`. OpenCL auto-routing
OFF unless `QECTOR_ENABLE_OPENCL_AUTO=1`.

### GPU weights — required for production

```python
from qector_decoder_v3.dem import from_stim

model = from_stim(circuit.detector_error_model())
c2q = model.check_to_qubits()
weights = model.weights().tolist()

d = q.CUDABatchDecoder(c2q, model.num_errors, weights)  # weighted path
print(d.precision)  # "f32" (default) or "f64" (when precision="f64")
```

Without weights: the GPU falls back to unweighted cluster growth (bit-identical
to CPU UnionFind) and warns (suppressed by `QECTOR_SILENT=1` — which the benchmark
harness sets to keep logs free of banner noise). Pass DEM weights for production use.

## 3. Source build

```bash
git clone https://github.com/GuillaumeLessard/qector-decoder.git
cd qector-decoder
maturin develop --release --no-default-features --features cuda   # CUDA
maturin develop --release --no-default-features --features opencl # OpenCL
maturin develop --release --no-default-features --features full   # CUDA+OpenCL+grpc
```

* Rust sources ship as 12-base64 secrets `RUST_SRC_B64_1..12` (`rust_core.sha256`).
  CI restores with `scripts/pack_rust_core.py unpack --from-env`.
* After ANY `src/*.rs` change, run `python scripts/pack_rust_core.py pack --out .secrets`
  and push the 12 secrets — plus commit `rust_core.sha256` in the same commit
  (see `docs/RELEASING.md`).

## 4. Platform notes

* **Linux**: `manylinux_2_17_x86_64` wheels need glibc ≥ 2.17 (CentOS 7+).
* **macOS**: arm64 wheel requires macOS 11.0+. Intel macOS is not a published target.
* **Windows**: win_amd64 wheel requires VC++ Redist 2022.
* **Python 3.9**: `pyproject.toml` pins `tomli>=2.0` for 3.9/3.10 (`tomllib` only stdlib since 3.11).

## 5. License tier

| Tier | max_distance | GPU | What unlocks |
|---|---|---|---|
| Community | 7 | No | included in published wheel |
| Pro | 19 | No | `QECTOR_LICENSE_KEY` / `QECTOR_LICENSE_FILE` |
| Enterprise | 63 | Yes | same + `~/.qector/license.key` |

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDABatchDecoder constructed WITHOUT edge_weights` | Unweighted path | Pass DEM weights; warning is `QECTOR_SILENT`-aware (suppressed when `QECTOR_SILENT=1`) |
| `Community tier d X > 7` | Distance cap | Use a smaller code, or a valid tier key |
| `src/ missing` in CI | secrets not refreshed | Run `pack --out .secrets` + `gh secret set` + commit `rust_core.sha256` |
| `test_grpc_distance_cap_rejects_oversized_code` fails on dev machine | Process latched `~/.qector/license.key` | Test skips on Enterprise, asserts only on unlicensed runners (CI) |
