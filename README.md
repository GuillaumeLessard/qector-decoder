# QECTOR Decoder v3

[![PyPI version](https://img.shields.io/pypi/v/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![License](https://img.shields.io/badge/License-Custom-yellow.svg)](https://github.com/GuillaumeLessard/qector-decoder/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/GuillaumeLessard/qector-decoder.svg)](https://github.com/GuillaumeLessard/qector-decoder/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/GuillaumeLessard/qector-decoder)](https://github.com/GuillaumeLessard/qector-decoder/commits/main)

Source-available Rust/Python QEC decoder package — v0.5.7

PyMatching-compatible MWPM validation, belief-matching accuracy mode, BP-OSD for LDPC/qLDPC, CPU/GPU batch decoding, Stim/Sinter integration, and artifact-backed benchmark evidence.

Website: https://qector.store  
PyPI: https://pypi.org/project/qector-decoder-v3/  
DOI: https://doi.org/10.5281/zenodo.20825980  
Commercial licensing: https://qector.store

---

## Install

```bash
pip install qector-decoder-v3
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; print('QECTOR OK')"
```

Optional extras:

```bash
pip install "qector-decoder-v3[stim]"
pip install "qector-decoder-v3[bench]"
pip install "qector-decoder-v3[all]"
```

---

## Supported wheel targets

| Platform | Wheel |
|---|---|
| Linux x86\_64 | Published |
| Windows x64 | Published |
| macOS arm64 / Apple Silicon | Published |
| macOS Intel x86\_64 | Not published in v0.5.x |
| CPython free-threaded cp313t | Not published in v0.5.x |

Supported Python: CPython 3.9 – 3.13.

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

`stim_compat` exposes two entry points with different input scopes:

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

# from a full stim.Circuit (stim_circuit_to_check_matrix converts it internally)
circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5)
c2q, nq = stim_circuit_to_check_matrix(circuit)
```

> **Note (v0.5.6):** `stim_circuit_to_check_matrix` and `from_stim_detector_error_model`
> are parallel implementations, not a Python alias. Both produce identical results for
> `DetectorErrorModel` input. `stim_circuit_to_check_matrix` additionally accepts a
> `stim.Circuit` object by calling `.detector_error_model(decompose_errors=True)`
> before delegating. Use whichever matches your input type.

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

### BeliefMatching

```python
import numpy as np
from qector_decoder_v3.belief_matching import BeliefMatching

# from a Stim DEM (recommended for circuit-level noise)
bm = BeliefMatching.from_detector_error_model(dem)

# from a raw check matrix H with uniform prior (v0.5.3+)
H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
bm = BeliefMatching(H, p=0.05)
obs = bm.decode(syndrome)
```

### CUDA / GPU

```python
from qector_decoder_v3 import CUDABatchDecoder

if CUDABatchDecoder.is_available():
    dec = CUDABatchDecoder(check_to_qubits, n_qubits)
    corrections = dec.batch_decode(syndromes)
else:
    print("No CUDA GPU — use BatchDecoder for CPU batch decoding")
```

---

## Independent validation (v0.5.3)

Validated by independent automated test suite — 87/87 checks PASS on v0.5.3.  
Platform: Windows 10, AMD Ryzen 16-core, NVIDIA GTX 1660 Ti (CUDA 7.5), Python 3.11, NumPy 2.2.6, PyMatching 2.4.0, stim 1.16.0.  
Full artifact: `benchmark_results/results_v053_retest.json`

| Claim | Result |
|---|---|
| 30 decoder × code combinations, 100% syndrome-valid corrections | ✅ |
| `pymatching_compat` bit-identical to PyMatching 2.4.0 | ✅ |
| Blossom LER within 0.00% of PyMatching on rep code d=3–9 | ✅ |
| Blossom LER within 1.78% of PyMatching on rotated surface code d=3–7 | ✅ |
| CUDA batch 100% CPU-agreeing at all batch sizes, GTX 1660 Ti | ✅ |
| CUDA batch 6.9–7.7× faster than CPU batch at 100k shots | ✅ |
| Workbench single-decode rep d=5 Blossom: 285,713 dec/s · p50 3.50 µs · p99 9.50 µs | ✅ |
| AutoDecoder backends: cpu, cuda (GTX 1660 Ti), opencl=False | ✅ |
| Workbench JSON/CSV/PDF export pipeline end-to-end | ✅ |
| LookupTableDecoder table\_size for rep d=5: 64 entries | ✅ |

### Single-shot latency reference (µs/decode, 2000 samples, independently validated)

| Decoder | rep d=5 | rep d=9 | surf d=3 | surf d=5 |
|---|---|---|---|---|
| UnionFindDecoder | 9.3 | 10.0 | 12.2 | 10.1 |
| FastUnionFindDecoder | 9.5 | 10.2 | 11.4 | 12.1 |
| BlossomDecoder | 10.6 | 10.6 | 14.8 | 16.8 |
| SparseBlossomDecoder | 11.8 | 10.6 | 11.5 | 29.2 |
| BatchDecoder (CPU) | 11.2 | 9.7 | 9.5 | 10.7 |
| LookupTableDecoder | 8.7 | 10.7 | 9.5 | — |

---

## Changelog summary

| Version | Date | Key change |
|---|---|---|
| 0.5.7 | 2026-06-29 | Advanced strategic QEC decoders: Fusion/Sparse Blossom, EBP, Restart Belief, KAT/QCT, Astra GNN, GPU pipelines |
| 0.5.6 | 2026-06-28 | `stim_compat` doc fix: `stim_circuit_to_check_matrix` is parallel impl, not alias |
| 0.5.5 | 2026-06-28 | `PredecodedDecoder.batch_decode()` wheel fix; PYTHONPATH guard; 775 tests pass |
| 0.5.4 | 2026-06-27 | `NeuralPredecoder.train()` clear error on numpy>=2.0; 125/125 validation |
| 0.5.3 | 2026-06-25 | `BatchDecoder.decode()`, `BeliefMatching(H, p)`, `QectorSinterDecoder.decode()` |
| 0.5.2 | 2026-06-25 | Adaptive Blossom k; QECTOR Workbench; `QectorDecoderWrapper` alias; evidence bundle |

Full changelog: [CHANGELOG.md](CHANGELOG.md)

---

## Honest limitations

- Union-Find is roughly 3× less accurate than MWPM on measured workloads.
- Single-round code-capacity noise does not produce surface-code threshold curves. Use circuit-level Stim DEM with `qector_sinter_decoders()` for threshold experiments.
- GPU speedups are hardware and batch-size dependent.
- `NeuralPredecoder.train()` requires numpy<2.0 until the Rust binding is updated.
- PyMatching and Stim remain important reference tools.

---

## Topics

quantum-error-correction qec quantum-computing decoder mwpm union-find bp-osd ldpc qldpc surface-code stim pymatching rust python cuda batch-decoding gpu-acceleration error-correction high-performance reproducible-research benchmarking pyo3 validation belief-propagation high-throughput

---

## License

Source-available. Commercial use requires written licensing: https://qector.store
