# QECTOR Decoder v3

[Skip to content](#qector-decoder-v3)

## Repository navigation

- [Code](https://github.com/GuillaumeLessard/qector-decoder)
- [Issues](https://github.com/GuillaumeLessard/qector-decoder/issues)
- [Pull requests](https://github.com/GuillaumeLessard/qector-decoder/pulls)
- [Actions](https://github.com/GuillaumeLessard/qector-decoder/actions)
- [Security](https://github.com/GuillaumeLessard/qector-decoder/security)
- [Insights](https://github.com/GuillaumeLessard/qector-decoder/insights)
- [Settings](https://github.com/GuillaumeLessard/qector-decoder/settings)
- [Tests](https://github.com/GuillaumeLessard/qector-decoder/tree/main/tests)

[![PyPI version](https://img.shields.io/pypi/v/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qector-decoder-v3.svg)](https://pypi.org/project/qector-decoder-v3/)
[![License](https://img.shields.io/badge/License-Custom-yellow.svg)](https://github.com/GuillaumeLessard/qector-decoder/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/GuillaumeLessard/qector-decoder.svg)](https://github.com/GuillaumeLessard/qector-decoder/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/GuillaumeLessard/qector-decoder)](https://github.com/GuillaumeLessard/qector-decoder/commits/main)

Source-available Rust/Python QEC decoder...

PyMatching-compatible MWPM validation, belief-matching accuracy mode, BP-OSD for LDPC/qLDPC, CPU/GPU batch decoding, and artifact-backed benchmark evidence.

Website: https://qector.store
PyPI: https://pypi.org/project/qector-decoder-v3/
DOI: https://doi.org/10.5281/zenodo.20825980
Commercial licensing: https://qector.store

## Topics

quantum-error-correction, qec, quantum-computing, decoder, mwpm, union-find, bp-osd, ldpc, qldpc, surface-code, stim, pymatching, rust, python, cuda, batch-decoding, gpu-acceleration, error-correction, high-performance, reproducible-research, benchmarking, scientific-computing, research-tool, pyo3, performance, validation, belief-propagation, high-throughput

## Install

```bash
pip install qector-decoder-v3
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; print('QECTOR OK')"
```

## Supported public wheel targets

| Platform | Wheel status |
|---|---|
| Linux x86_64 | Published |
| Windows x64 | Published |
| macOS arm64 / Apple Silicon | Published |
| macOS Intel x86_64 | Not published in v0.5.x public CI |
| CPython free-threaded builds such as cp313t | Not published in v0.5.x |

Supported Python classifiers are standard CPython 3.9 to 3.13.

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

## Independent validation (v0.5.3)

Validated by independent automated test suite (87/87 checks PASS on v0.5.3).
Platform: Windows 10, AMD Ryzen 16-core, NVIDIA GTX 1660 Ti (CUDA 7.5), Python 3.11, PyMatching 2.4.0, stim 1.16.0.
Full machine-readable artifact: `benchmark_results/results_v053_retest.json`.

| Claim | Result |
|---|---|
| All 30 decoder × code combinations produce 100% syndrome-valid corrections | ✅ Confirmed |
| `pymatching_compat` bit-identical to PyMatching 2.4.0 | ✅ Confirmed |
| Blossom LER within 0.00% of PyMatching on repetition code (d=3–9) | ✅ Confirmed |
| Blossom LER within 1.78% of PyMatching on rotated surface code (d=3–7) | ✅ Confirmed |
| CUDA batch output 100% CPU-agreeing across all batch sizes, GTX 1660 Ti | ✅ Confirmed |
| CUDA batch 6.9–7.7× faster than CPU batch at 100k shots | ✅ Confirmed |
| Workbench single-decode (rep d=5, Blossom): 285,713 dec/s · p50 3.50 µs · p99 9.50 µs | ✅ Confirmed |
| AutoDecoder backends detected: cpu, cuda (GTX 1660 Ti), opencl=False | ✅ Confirmed |
| Workbench JSON/CSV/PDF export pipeline end-to-end functional | ✅ Confirmed |
| LookupTableDecoder table_size for rep d=5: 64 entries | ✅ Confirmed |

## Honest limitations

- Union-Find is roughly 3x less accurate than MWPM on measured workloads.
- Single-round code-capacity noise does not produce surface-code threshold curves.
- GPU speedups are hardware and batch-size dependent.
- PyMatching and Stim remain important reference tools.

## License

Source-available. Commercial use requires written licensing through https://qector.store.
