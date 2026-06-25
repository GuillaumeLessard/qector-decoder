# QECTOR Decoder v3

**Python and Rust quantum error correction decoder package.**

QECTOR Decoder v3 helps researchers and developers test quantum error correction decoder workflows from Python, with native Rust performance paths and reproducible benchmark artifacts.

Common search terms: quantum error correction, QEC decoder, quantum decoder Python, Rust quantum error correction, PyMatching-compatible decoder, Stim workflow, Sinter benchmark, BP-OSD, LDPC, qLDPC, surface code decoder, MWPM decoder, union find decoder, belief matching, PyO3, maturin, QECTOR.

Website: https://www.qector.store  
PyPI: https://pypi.org/project/qector-decoder-v3/  
DOI: https://doi.org/10.5281/zenodo.20825980  
Repository: https://github.com/GuillaumeLessard/qector-decoder  
Commercial licensing: https://www.qector.store

---

## Installation

Recommended command:

```bash
pip install qector-decoder-v3
```

Verify the install:

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

Use the `pip` bound to your active standard Python environment. Do not force `py -m pip` unless you have checked which interpreter the Windows launcher selected.

On some systems, `py` can select a free-threaded interpreter such as `python3.13t.exe`. QECTOR v0.5.x publishes standard CPython wheels, not `cp313t` free-threaded wheels.

Working Windows command:

```powershell
pip install qector-decoder-v3
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; print('QECTOR OK')"
```

Check launcher targets with:

```powershell
py -0p
```

---

## Supported public wheels

| Platform | Wheel status |
|---|---|
| Linux x86_64 | Published |
| Windows x64 | Published |
| macOS arm64 / Apple Silicon | Published |
| macOS Intel x86_64 | Not published in v0.5.x public CI |
| CPython free-threaded builds such as `cp313t` | Not published in v0.5.x |

Supported Python classifiers are standard CPython 3.9 to 3.13.

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
# single-shot also works (v0.5.3+):
corrections_single = cpu.decode(syndromes[0])
print(corrections.shape)
```

---

## API surface

### Stim / DEM integration

```python
import stim
from qector_decoder_v3.stim_compat import (
    from_stim_detector_error_model,
    stim_circuit_to_check_matrix,  # identical alias
    to_stim_decoder,
    stim_decoder_from_dem,
)

dem = stim.Circuit.generated(
    "surface_code:rotated_memory_x", distance=5
).detector_error_model(decompose_errors=True)

c2q, nq = from_stim_detector_error_model(dem)
# or equivalently: stim_circuit_to_check_matrix(dem)

decoder = stim_decoder_from_dem(dem)
```

### Sinter integration

```python
import sinter
from qector_decoder_v3.sinter_compat import (
    QectorSinterDecoder,
    QectorDecoderWrapper,   # backward-compat alias for QectorSinterDecoder
    qector_sinter_decoders,
)

samples = sinter.collect(
    num_workers=4,
    tasks=tasks,
    decoders=["qector_belief", "qector_blossom", "qector_unionfind"],
    custom_decoders=qector_sinter_decoders(),
)

# standalone single-syndrome decode (v0.5.3+, no Sinter required):
dec = QectorSinterDecoder("blossom")
obs = dec.decode(syndrome, dem=dem)
```

### BeliefMatching raw H constructor (v0.5.3+)

```python
import numpy as np
from qector_decoder_v3.belief_matching import BeliefMatching

# From a Stim DEM (recommended for circuit-level noise):
bm = BeliefMatching.from_detector_error_model(dem)

# From a raw check matrix H (uniform prior p=0.1):
H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
bm = BeliefMatching(H, p=0.05)
obs = bm.decode(syndrome)
```

### CUDA / GPU

```python
from qector_decoder_v3 import CUDABatchDecoder

# Always check availability before constructing
if CUDABatchDecoder.is_available():
    dec = CUDABatchDecoder(check_to_qubits, n_qubits)
    corrections = dec.batch_decode(syndromes)
else:
    print("No CUDA GPU detected — use BatchDecoder for CPU batch decoding")
```

---

## Independent validation (v0.5.3)

Validated by independent automated test suite (87/87 checks PASS on v0.5.3).
Platform: Windows 10, AMD Ryzen 16-core, NVIDIA GTX 1660 Ti (CUDA 7.5), Python 3.11, PyMatching 2.4.0.
Full artifact: `benchmark_results/results_v053_retest.json`.

**v0.5.3 API fixes (all 3 verified):**

| Fix | Before | After |
|---|---|---|
| `BatchDecoder.decode(syndrome)` | absent (only `parallel_batch_decode`) | present, 1-row batch wrapper |
| `BeliefMatching(H, p=0.1)` | TypeError (expected `_Matrices`) | accepts raw numpy H, uniform prior |
| `QectorSinterDecoder.decode(syndrome, dem)` | absent | present, DEM cached on first call |

| Claim | Result |
|---|---|
| 30 decoder x code combinations, 100% syndrome-valid corrections | Confirmed |
| `pymatching_compat` bit-identical to PyMatching 2.4.0 | Confirmed |
| Blossom LER within 0.00% of PyMatching on repetition code d=3-9 | Confirmed |
| Blossom LER within 1.78% of PyMatching on rotated surface code d=3-7 | Confirmed |
| CUDA batch 100% CPU-agreeing at all tested batch sizes (GTX 1660 Ti) | Confirmed |
| CUDA batch 6.9-7.7x faster than CPU batch at 100k shots | Confirmed |
| Workbench single-decode rep d=5 Blossom: 285,713 dec/s, p50 3.50 us, p99 9.50 us | Confirmed |
| AutoDecoder backends: cpu, cuda GTX 1660 Ti, opencl=False | Confirmed |
| Workbench JSON/CSV/PDF export pipeline end-to-end | Confirmed |
| LookupTableDecoder table_size for rep d=5: 64 entries | Confirmed |
| d=101 stress decode completes without error | Confirmed |
| Invalid input rejected with clear ValueError / TypeError | Confirmed |

### Single-shot latency reference (us/decode, 2000 samples, independently validated)

| Decoder | rep d=5 | rep d=9 | surf d=3 | surf d=5 |
|---|---|---|---|---|
| UnionFindDecoder | 9.3 | 10.0 | 12.2 | 10.1 |
| FastUnionFindDecoder | 9.5 | 10.2 | 11.4 | 12.1 |
| BlossomDecoder | 10.6 | 10.6 | 14.8 | 16.8 |
| SparseBlossomDecoder | 11.8 | 10.6 | 11.5 | 29.2 |
| CPUBatchDecoder | 11.2 | 9.7 | 9.5 | 10.7 |
| LookupTableDecoder | 8.7 | 10.7 | 9.5 | — |

LookupTableDecoder is the fastest single-shot decoder on rep d=5 (precomputed table, 64 entries, O(1) lookup).

### Known limitations

- **Union-Find is ~3x less accurate than MWPM** — expected speed/accuracy trade-off.
- **Single-round code-capacity noise does not produce surface-code distance scaling.** Use circuit-level Stim DEM with `qector_sinter_decoders()` for threshold curves.
- **SparseBlossom batch may return different (but valid) corrections than single-shot on degenerate syndromes.** Benign matching degeneracy.
- **CUDABatchDecoder raises RuntimeError cleanly when no CUDA GPU is present.** Use `CUDABatchDecoder.is_available()` to check first.

---

## License

Source-available. Commercial use requires written licensing through https://www.qector.store.
