# QECTOR Decoder v3

**Source-available Rust/Python quantum error correction decoding platform.**

QECTOR Decoder v3 provides a Python package backed by a native Rust extension for quantum error correction research and validation workflows.

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

On some systems, `py` can select a free-threaded interpreter such as `python3.13t.exe`. QECTOR v0.5.x publishes standard CPython wheels, not `cp313t` free-threaded wheels. If pip cannot find a matching wheel, it may fall back to a source build and fail because the public repository does not ship the proprietary Rust core.

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
print(corrections.shape)
```

---

## License

Source-available. Commercial use requires written licensing through https://www.qector.store.
