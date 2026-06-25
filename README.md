# QECTOR Decoder v3

<div align="center">

[![CI](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/CI.yml/badge.svg)](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/CI.yml)
[![tests](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/tests.yml/badge.svg)](https://github.com/GuillaumeLessard/qector-decoder/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/badge/PyPI-qector--decoder--v3-blue.svg)](https://pypi.org/project/qector-decoder-v3/)
[![version](https://img.shields.io/badge/version-0.5.3-blue.svg)](https://pypi.org/project/qector-decoder-v3/)
[![python](https://img.shields.io/badge/python-3.9--3.13-blue.svg)](https://pypi.org/project/qector-decoder-v3/)
[![license](https://img.shields.io/badge/license-Source--Available-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20825980.svg)](https://doi.org/10.5281/zenodo.20825980)

**Source-available Rust/Python quantum error correction decoding platform.**

PyMatching-compatible MWPM validation · Belief-matching accuracy mode · BP-OSD for LDPC/qLDPC · CPU/GPU batch decoding · Artifact-backed benchmark evidence

[Website](https://www.qector.store) · [PyPI](https://pypi.org/project/qector-decoder-v3/) · [DOI](https://doi.org/10.5281/zenodo.20825980) · [Commercial licensing](https://www.qector.store)

</div>

---

## Install

Recommended PyPI command:

```bash
pip install qector-decoder-v3
```

Verify with the same Python environment:

```bash
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; print('QECTOR OK')"
```

Check the active Python and pip:

```bash
python --version
pip --version
```

## Windows note

Use the `pip` bound to your active standard Python environment. Do not force `py -m pip` unless you have checked which interpreter the Windows launcher selected.

On some Windows systems, `py` can select a free-threaded interpreter such as `python3.13t.exe`. QECTOR v0.5.x publishes standard CPython wheels, not `cp313t` free-threaded wheels. If pip cannot find a matching wheel, it may fall back to a source build and fail because the public repository does not ship the proprietary Rust core.

Inspect Python launchers with:

```powershell
py -0p
```

Working Windows command:

```powershell
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
| CPython free-threaded builds such as `cp313t` | Not published in v0.5.x |

Supported Python classifiers are standard CPython **3.9 to 3.13**.

## Optional extras

```bash
pip install "qector-decoder-v3[stim]"
pip install "qector-decoder-v3[bench]"
pip install "qector-decoder-v3[all]"
```

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

## Public links

| Resource | Link |
|---|---|
| Website | https://www.qector.store |
| PyPI | https://pypi.org/project/qector-decoder-v3/ |
| DOI | https://doi.org/10.5281/zenodo.20825980 |
| Repository | https://github.com/GuillaumeLessard/qector-decoder |
| Commercial licensing | https://www.qector.store |

## Independent validation (v0.5.3)

Validated by independent automated test suite (86/87 checks passing on v0.5.2 baseline; all 5 API failures closed in v0.5.3, post-fix 33/33 PASS + 1 SKIP).
Platform: Windows 10, AMD Ryzen 16-core, NVIDIA GTX 1660 Ti (CUDA 7.5), Python 3.11, PyMatching 2.4.0, stim 1.16.0.
Full machine-readable artifact: `benchmark_results/validation_v051.json`.

| Claim | Result |
|---|---|
| All 30 decoder × code combinations produce 100% syndrome-valid corrections | ✅ Confirmed |
| `pymatching_compat` bit-identical to PyMatching 2.4.0 | ✅ Confirmed |
| Blossom LER within 0.00% of PyMatching on repetition code (d=3–9) | ✅ Confirmed |
| Blossom LER within 1.78% of PyMatching on rotated surface code (d=3–7) | ✅ Confirmed |
| CUDA batch output 100% CPU-agreeing across all batch sizes, GTX 1660 Ti | ✅ Confirmed |
| CUDA batch 6.9–7.7× faster than CPU batch at 100k shots | ✅ Confirmed |
| Workbench single-decode (rep d=5, Blossom): 277,778 dec/s · p50 3.60 µs · p99 11.61 µs | ✅ Confirmed |
| AutoDecoder backends detected: cpu, cuda (GTX 1660 Ti), opencl=False | ✅ Confirmed |
| Workbench JSON/CSV/PDF export pipeline end-to-end functional | ✅ Confirmed |
| LookupTableDecoder table_size for rep d=5: 64 entries | ✅ Confirmed |
| d=101 stress decode completes without error | ✅ Confirmed |
| Invalid input rejected with clear `ValueError` / `TypeError` | ✅ Confirmed |

## Honest limitations (validated)

- **Union-Find is ~3× less accurate than MWPM.** Measured across d=3–9 repetition and surface codes. Expected speed/accuracy trade-off. Use Blossom or SparseBlossom when accuracy matters.
- **Single-round code-capacity noise does not produce surface-code threshold curves.** Bundled `rotated_surface_code` under code-capacity (single-round) noise shows d=3/5/7 LER curves overlapping within ~1%. PyMatching on the same H/L behaves identically — this is a property of the noise model, not a decoder defect. All corrections remain 100% valid. For genuine threshold curves, use **circuit-level noise via Stim DEM** with `BeliefMatching` or `qector_sinter_decoders()` (see `stim_compat` docstring for the full example).
- **SparseBlossom batch may return different (but valid) corrections than single-shot on degenerate cases.** Benign matching degeneracy, not an error.
- **GPU wins only at sufficient batch sizes** (typically ≥ 4096). Measure on your own hardware before quoting speedup.
- **CUDABatchDecoder raises a clean `RuntimeError` when no driver is present.** Check `CUDABatchDecoder.is_available()` before constructing.

## License

Source-available. See [LICENSE](LICENSE). Commercial use requires written licensing through https://www.qector.store.
