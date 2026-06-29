# Installation Guide — QECTOR Decoder v3 (v0.5.7)

The fastest path is a direct PyPI wheel install. Source builds are only needed
when you want to develop against the Rust core or target a platform without a
published wheel.

---

## PyPI wheel install (recommended)

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

## Windows note

Use the `pip` bound to your active standard CPython environment. Do not force
`py -m pip` unless you have verified which interpreter the Windows launcher selected.
On some systems `py` selects the free-threaded interpreter (`python3.13t.exe`).
QECTOR v0.5.x publishes standard CPython wheels only — not `cp313t`.

Check launcher targets:

```powershell
py -0p
```

Working PowerShell install:

```powershell
pip install qector-decoder-v3
python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; print('QECTOR OK')"
```

---

## Source build (Windows PowerShell)

Install Rust from https://rustup.rs first, then:

```powershell
git clone https://github.com/GuillaumeLessard/qector-decoder.git
cd qector-decoder

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip maturin

$env:PYO3_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features

.\.venv\Scripts\python.exe -c "from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')"
```

Expected output ending:

```
Installed qector-decoder-v3-0.5.7
QECTOR OK
```

pip may print ignored optional-extra messages for pytest, stim, pymatching, sinter,
and ldpc during install. That is expected — the base command installs the minimal
CPU-safe runtime only.

---

## Source build (Git Bash on Windows)

```bash
git clone https://github.com/GuillaumeLessard/qector-decoder.git
cd qector-decoder

python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip maturin

export PYO3_PYTHON="$(pwd -W)/.venv/Scripts/python.exe"
python -m maturin develop --release --no-default-features

python -c "from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')"
```

---

## Source build (Linux / macOS)

```bash
git clone https://github.com/GuillaumeLessard/qector-decoder.git
cd qector-decoder

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin

maturin develop --release --no-default-features

python -c "from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')"
```

---

## Optional test dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install `
    "pytest>=7" "hypothesis>=6" "fastapi>=0.110" "uvicorn>=0.29" "httpx>=0.27" `
    stim pymatching sinter ldpc beliefmatching psutil matplotlib tabulate scipy

.\.venv\Scripts\python.exe -m pytest python/tests -q --tb=short
```

---

## Optional feature builds

The default install uses `--no-default-features` for the safest public CPU path.

To build with CUDA and OpenCL support:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
```

CUDA requires the NVCC toolchain and CUDA driver >= 7.5.
OpenCL requires a valid ICD loader and GPU driver. See `docs/PLATFORM_ARTIFACT_ROADMAP.md`
for the known AMD OCL SDK Light false-negative on `is_available()`.

---

## Verified platforms

| Platform | Python | Wheel | Notes |
|---|---|---|---|
| Windows 10 x64 | 3.11, 3.13 | Published | GTX 1660 Ti CUDA 7.5 validated |
| Linux x86\_64 | 3.9 – 3.13 | Published | CI-built, not locally validated |
| macOS arm64 | 3.9 – 3.13 | Published | CI-built |
| macOS x86\_64 | 3.9 – 3.13 | Not published | Source build required |
| cp313t (free-threaded) | — | Not published | Standard CPython only in v0.5.x |
