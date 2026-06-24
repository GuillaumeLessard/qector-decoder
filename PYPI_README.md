# QECTOR Decoder v3

**Source-available Rust/Python quantum error correction decoding platform.**

QECTOR Decoder v3 provides a Python package backed by a native Rust extension for quantum error correction research and validation workflows. It includes PyMatching-compatible MWPM validation, Union-Find decoding, belief-matching experiments, BP-OSD/qLDPC workflows, batch decoding, and optional GPU backend checks where the release build and target machine support them.

Website: https://www.qector.store  
Repository: https://github.com/GuillaumeLessard/qector-decoder  
PyPI: https://pypi.org/project/qector-decoder-v3/  
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

Optional research and validation extras:

```bash
# Stim, Sinter, PyMatching, LDPC and belief-matching ecosystem
pip install "qector-decoder-v3[stim]"

# Benchmark and plotting harness
pip install "qector-decoder-v3[bench]"

# Full validation environment
pip install "qector-decoder-v3[all]"
```

---

## Supported public wheels

QECTOR v0.5.x public releases target standard CPython wheels for:

| Platform | Wheel status |
|---|---|
| Linux x86_64 | Published |
| Windows x64 | Published |
| macOS arm64 / Apple Silicon | Published |
| macOS Intel x86_64 | Not published in v0.5.x public CI |
| CPython free-threaded builds such as `cp313t` | Not published in v0.5.x |

Supported Python classifiers are standard CPython 3.9 to 3.13.

Windows note: do not force `py -m pip` unless you have checked which interpreter the Windows launcher selected. If `py` selects a free-threaded interpreter such as `python3.13t.exe`, pip may fall back to a source build and fail because the public repository does not ship the proprietary Rust core. Use the standard active Python environment instead:

```powershell
pip install qector-decoder-v3
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; print('QECTOR OK')"
```

Check your Python launcher targets with:

```powershell
py -0p
```

---

## Quick start

```python
import numpy as np
from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder

check_to_qubits = [[0, 1], [1, 2], [2, 3], [3, 4]]
n_qubits = 5
syndrome = np.array([0, 1, 0, 0], dtype=np.uint8)

fast = UnionFindDecoder(check_to_qubits, n_qubits)
print(fast.decode(syndrome))

mwpm = BlossomDecoder(check_to_qubits, n_qubits)
print(mwpm.decode(syndrome))
```

Batch decoding:

```python
import numpy as np
from qector_decoder_v3 import BatchDecoder, CUDABatchDecoder

checks = [[0, 1], [1, 2], [2, 3], [3, 4]]
syndromes = np.random.randint(0, 2, size=(4096, 4), dtype=np.uint8)

cpu = BatchDecoder(checks, n_qubits=5)
corrections = cpu.parallel_batch_decode(syndromes)

if CUDABatchDecoder.is_available():
    gpu = CUDABatchDecoder(checks, n_qubits=5)
    corrections = gpu.batch_decode(syndromes)
```

Stim workflow:

```python
import stim
from qector_decoder_v3 import BlossomDecoder
from qector_decoder_v3.stim_compat import stim_circuit_to_check_matrix

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=5,
    rounds=5,
    after_clifford_depolarization=0.005,
)

checks, n_qubits = stim_circuit_to_check_matrix(circuit)
decoder = BlossomDecoder(checks, n_qubits)
```

---

## Included decoder families

| Module | Primary use | Status |
|---|---|---|
| `UnionFindDecoder` | Fast approximate decoding | Stable public API |
| `FastUnionFindDecoder` | Optimized Union-Find path | Stable public API |
| `BlossomDecoder` | Exact MWPM / PyMatching-parity validation | Stable public API |
| `SparseBlossomDecoder` | Faster near-optimal matching | Experimental |
| `BeliefMatching` | Correlated-noise accuracy experiments | Research/accuracy mode |
| `BpOsdDecoder` | LDPC and qLDPC workflows | Experimental |
| `BatchDecoder` / `CPUBatchDecoder` | CPU Monte Carlo sweeps | Stable public API |
| `CUDABatchDecoder` | CUDA batch decoding | Runtime/build dependent |
| `OpenCLBatchDecoder` | OpenCL batch decoding | Runtime/build dependent |
| `stim_compat` | Stim circuit and DEM conversion | Stable utility |
| `sinter_compat` | Sinter custom decoder integration | Stable utility |
| `rest_api` | Local decoding service | Local/partner review only |

---

## Evidence-backed positioning

QECTOR Decoder v3 is positioned as a source-available QEC R&D platform, not as a blanket replacement for every mature decoder in every workload.

The repository includes public benchmark artifacts and reproduction scripts for:

- PyMatching-parity logical-error-rate checks on selected surface-code workloads
- belief-matching accuracy experiments on selected workloads
- GPU bit-identity checks against CPU output on a tested NVIDIA machine
- native memory profiling for selected decoder paths

Important boundaries:

- PyMatching remains faster for standard MWPM latency in the checked-in comparison artifacts.
- Belief-matching is an accuracy/research mode and is much slower in the provided experiments.
- GPU availability and performance depend on wheel build features, drivers, hardware, and runtime checks.
- OpenCL support must be confirmed on the target machine or built under the appropriate licensed/custom configuration.
- REST/API surfaces are for local experiments or controlled review unless separately hardened.

Full methodology, reproducibility notes, and benchmark artifacts are in the GitHub repository:

https://github.com/GuillaumeLessard/qector-decoder

---

## GPU availability check

```python
from qector_decoder_v3 import CUDABatchDecoder, OpenCLBatchDecoder

print("CUDA:", CUDABatchDecoder.is_available())
print("OpenCL:", OpenCLBatchDecoder.is_available())
```

Do this before making any hardware-specific performance claim.

---

## Licensing

QECTOR Decoder v3 is source-available.

Personal, academic, educational and non-commercial research use is allowed under the repository license. Company use, funded institutional work, SaaS, hosted API deployment, OEM integration, redistribution, paid consulting, or commercial benchmarking requires a commercial license.

Website and commercial licensing:

https://www.qector.store

Contact:

admin@qector.store

---

## Citation

```bibtex
@software{lessard2026qector,
  author  = {Guillaume Lessard},
  title   = {{QECTOR Decoder v3}: Rust/Python Quantum Error Correction Decoding Platform},
  year    = {2026},
  version = {0.5.0},
  url     = {https://www.qector.store},
  note    = {Source-available. Commercial license required for commercial use.}
}
```
