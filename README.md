# QECTOR Decoder v3

Source-available Rust/Python quantum error correction decoding platform.

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

## Independent validation v0.5.3

Validated by independent automated test suite: 86/87 checks passing on v0.5.2 baseline; all 5 API failures closed in v0.5.3, post-fix 33/33 PASS + 1 SKIP.

Artifact: benchmark_results/validation_v051.json

## Honest limitations

- Union-Find is roughly 3x less accurate than MWPM on measured workloads.
- Single-round code-capacity noise does not produce surface-code threshold curves.
- GPU speedups are hardware and batch-size dependent.
- PyMatching and Stim remain important reference tools.

## License

Source-available. Commercial use requires written licensing through https://qector.store.
