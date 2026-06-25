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
