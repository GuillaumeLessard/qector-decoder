# QECTOR Decoder v3

**Python and Rust quantum error correction decoder package — v0.5.6**

QECTOR Decoder v3 helps researchers and developers build and benchmark quantum error correction decoder workflows from Python, with native Rust performance paths, GPU acceleration, and artifact-backed reproducible evidence.

Common search terms: quantum error correction, QEC decoder, quantum decoder Python, Rust quantum error correction, PyMatching-compatible decoder, Stim workflow, Sinter benchmark, BP-OSD, LDPC, qLDPC, surface code decoder, MWPM decoder, union find decoder, belief matching, PyO3, maturin, QECTOR.

Website: https://www.qector.store  
PyPI: https://pypi.org/project/qector-decoder-v3/  
DOI: https://doi.org/10.5281/zenodo.20825980  
Repository: https://github.com/GuillaumeLessard/qector-decoder  
Commercial licensing: https://www.qector.store

---

## Installation

```bash
pip install qector-decoder-v3
```

Verify:

```bash
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; print('QECTOR OK')"
```

Optional extras:

```bash
pip install "qector-decoder-v3[stim]"
pip install "qector-decoder-v3[bench]"
pip install "qector-decoder-v3[all]"
```

---

## Windows note

Use the `pip` bound to your active standard CPython environment. Do not force `py -m pip` unless you have verified which interpreter the Windows launcher selected.

On some systems `py` can select a free-threaded interpreter (`python3.13t.exe`). QECTOR v0.5.x publishes standard CPython wheels only, not `cp313t` free-threaded wheels.

```powershell
pip install qector-decoder-v3
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; print('QECTOR OK')"
```

Check launcher targets with `py -0p`.

---

## Supported wheels

| Platform | Wheel |
|---|---|
| Linux x86\_64 | Published |
| Windows x64 | Published |
| macOS arm64 / Apple Silicon | Published |
| macOS Intel x86\_64 | Not published in v0.5.x |
| CPython free-threaded `cp313t` | Not published in v0.5.x |

Supported Python: standard CPython 3.9 – 3.13.

---

## Quick start

```python
import numpy as np
from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder

check_to_qubits = [[0, 1], [1, 2], [2, 3], [3, 4]]
n_qubits = 5
syndrome = np.array([0, 1, 0, 0], dtype=np.uint8)

uf = UnionFindDecoder(check_to_qubits, n_qubits)
print(uf.decode(syndrome))

mwpm = BlossomDecoder(check_to_qubits, n_qubits)
print(mwpm.decode(syndrome))
```

Batch decoding:

```python
import numpy as np
from qector_decoder_v3 import BatchDecoder

checks = [[0, 1], [1, 2], [2, 3], [3, 4]]
syndromes = np.random.randint(0, 2, size=(4096, 4), dtype=np.uint8)

cpu = BatchDecoder(checks, n_qubits=5)
corrections = cpu.parallel_batch_decode(syndromes)
corrections_single = cpu.decode(syndromes[0])   # single-shot (v0.5.3+)
print(corrections.shape)
```

---

## API surface

### Core decoders

```python
from qector_decoder_v3 import (
    UnionFindDecoder,
    FastUnionFindDecoder,
    BlossomDecoder,
    SparseBlossomDecoder,
    BatchDecoder,
    CUDABatchDecoder,
    LookupTableDecoder,
    PredecodedDecoder,
)
```

### Stim / DEM integration

`stim_compat` exposes two parallel entry points with different input scopes:

```python
import stim
from qector_decoder_v3.stim_compat import (
    from_stim_detector_error_model,   # accepts DetectorErrorModel or str
    stim_circuit_to_check_matrix,     # superset: also accepts stim.Circuit
    to_stim_decoder,
    stim_decoder_from_dem,
)

# from a DetectorErrorModel
dem = stim.Circuit.generated(
    "surface_code:rotated_memory_x", distance=5
).detector_error_model(decompose_errors=True)
c2q, nq = from_stim_detector_error_model(dem)

# from a full stim.Circuit — stim_circuit_to_check_matrix converts it internally
circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5)
c2q, nq = stim_circuit_to_check_matrix(circuit)
```

> **v0.5.6 clarification:** `stim_circuit_to_check_matrix` and
> `from_stim_detector_error_model` are **parallel implementations**, not a Python
> alias. Both return identical results for `DetectorErrorModel` input.
> `stim_circuit_to_check_matrix` additionally accepts a `stim.Circuit` object by
> calling `.detector_error_model(decompose_errors=True)` internally before delegating.
> Use whichever matches your input type.

### Sinter integration

```python
import sinter
from qector_decoder_v3.sinter_compat import (
    QectorSinterDecoder,
    QectorDecoderWrapper,       # backward-compat alias for QectorSinterDecoder
    qector_sinter_decoders,
)

samples = sinter.collect(
    num_workers=4,
    tasks=tasks,
    decoders=["qector_belief", "qector_blossom", "qector_unionfind"],
    custom_decoders=qector_sinter_decoders(),
)

# standalone single-syndrome decode — no Sinter required (v0.5.3+)
dec = QectorSinterDecoder("blossom")
obs = dec.decode(syndrome, dem=dem)
```

### BeliefMatching raw H constructor (v0.5.3+)

```python
import numpy as np
from qector_decoder_v3.belief_matching import BeliefMatching

# from a Stim DEM (recommended for circuit-level noise)
bm = BeliefMatching.from_detector_error_model(dem)

# from a raw check matrix H with uniform prior
H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
bm = BeliefMatching(H, p=0.05)
obs = bm.decode(syndrome)
```

### CUDA / GPU batch decoding

```python
from qector_decoder_v3 import CUDABatchDecoder

if CUDABatchDecoder.is_available():
    dec = CUDABatchDecoder(check_to_qubits, n_qubits)
    corrections = dec.batch_decode(syndromes)
else:
    print("No CUDA GPU detected — use BatchDecoder for CPU batch decoding")
```

---

## Independent validation (v0.5.3)

Validated by independent automated test suite — 87/87 checks PASS.  
Platform: Windows 10, AMD Ryzen 16-core, NVIDIA GTX 1660 Ti (CUDA 7.5), Python 3.11, NumPy 2.2.6, PyMatching 2.4.0, stim 1.16.0.  
Full artifact: `benchmark_results/results_v053_retest.json`

**v0.5.3 API fixes (all 3 verified):**

| Fix | Before | After |
|---|---|---|
| `BatchDecoder.decode(syndrome)` | absent | present — 1-row batch wrapper |
| `BeliefMatching(H, p=...)` | TypeError on raw numpy H | accepts raw H with uniform prior |
| `QectorSinterDecoder.decode(syndrome, dem)` | absent | present — DEM cached on first call |

**Benchmark claims:**

| Claim | Result |
|---|---|
| 30 decoder × code combinations, 100% syndrome-valid corrections | ✅ |
| `pymatching_compat` bit-identical to PyMatching 2.4.0 | ✅ |
| Blossom LER within 0.00% of PyMatching on rep code d=3–9 | ✅ |
| Blossom LER within 1.78% of PyMatching on rotated surface code d=3–7 | ✅ |
| CUDA batch 100% CPU-agreeing across all tested batch sizes (GTX 1660 Ti) | ✅ |
| CUDA batch 6.9–7.7× faster than CPU batch at 100k shots | ✅ |
| Workbench single-decode rep d=5 Blossom: 285,713 dec/s · p50 3.50 µs · p99 9.50 µs | ✅ |
| AutoDecoder backends: cpu, cuda (GTX 1660 Ti), opencl=False | ✅ |
| Workbench JSON/CSV/PDF export pipeline end-to-end | ✅ |
| LookupTableDecoder table\_size for rep d=5: 64 entries | ✅ |
| d=101 stress decode completes without error | ✅ |
| Invalid input rejected with clear ValueError / TypeError | ✅ |

### Single-shot latency reference (µs/decode, 2000 samples, independently validated)

| Decoder | rep d=5 | rep d=9 | surf d=3 | surf d=5 |
|---|---|---|---|---|
| UnionFindDecoder | 9.3 | 10.0 | 12.2 | 10.1 |
| FastUnionFindDecoder | 9.5 | 10.2 | 11.4 | 12.1 |
| BlossomDecoder | 10.6 | 10.6 | 14.8 | 16.8 |
| SparseBlossomDecoder | 11.8 | 10.6 | 11.5 | 29.2 |
| BatchDecoder (CPU) | 11.2 | 9.7 | 9.5 | 10.7 |
| LookupTableDecoder | 8.7 | 10.7 | 9.5 | — |

LookupTableDecoder is the fastest single-shot decoder on rep d=5 (precomputed table, 64 entries, O(1) lookup).

---

## Changelog summary

| Version | Date | Key change |
|---|---|---|
| **0.5.6** | 2026-06-28 | `stim_compat` doc fix: `stim_circuit_to_check_matrix` documented as parallel impl, not alias |
| 0.5.5 | 2026-06-28 | `PredecodedDecoder.batch_decode()` wheel sync; PYTHONPATH guard; 775 tests pass |
| 0.5.4 | 2026-06-27 | `NeuralPredecoder.train()` clear error on numpy>=2.0; 125/125 validation |
| 0.5.3 | 2026-06-25 | `BatchDecoder.decode()`, `BeliefMatching(H, p)`, `QectorSinterDecoder.decode()` |
| 0.5.2 | 2026-06-25 | Adaptive Blossom k; QECTOR Workbench; `QectorDecoderWrapper` alias; evidence bundle |

---

## Known limitations

- **Union-Find is ~3× less accurate than MWPM** — expected speed/accuracy trade-off.
- **Single-round code-capacity noise does not produce surface-code distance scaling.** Use circuit-level Stim DEM with `qector_sinter_decoders()` for threshold curves.
- **SparseBlossom batch may return different (but valid) corrections than single-shot on degenerate syndromes.** Benign matching degeneracy.
- **`CUDABatchDecoder` raises `RuntimeError` cleanly when no CUDA GPU is present.** Always call `CUDABatchDecoder.is_available()` first.
- **`NeuralPredecoder.train()` requires numpy<2.0** until the Rust binding is updated to the modern `Bound<'py, PyArray2<u8>>` API.

---

## License

Source-available. Commercial use requires written licensing: https://www.qector.store
