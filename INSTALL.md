# Installation Guide — QECTOR Decoder v0.5.4

The current repository does **not** include `install.py`. Use the source build path below.

## Windows PowerShell

Install Rust first from rustup, then run:

```powershell
git clone https://github.com/GuillaumeLessard/qector-decoder.git
cd qector-decoder

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip maturin

$env:PYO3_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features

.\.venv\Scripts\python.exe -c "from qector_decoder_v3 import UnionFindDecoder; print('QECTOR OK')"
```

### Verified Windows result

This command has been verified from a fresh clone on a second Windows PC with Python 3.11. The expected successful ending is:

```text
Installed qector-decoder-v3-0.5.4
QECTOR OK
```

During the base install, pip may print ignored optional-extra messages for packages such as pytest, stim, pymatching, sinter, and ldpc. That is normal because the public command installs the minimal CPU-safe runtime build. Install optional test and benchmark dependencies only when needed.

## Git Bash on Windows

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

## Optional test dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install "pytest>=7" "hypothesis>=6" "fastapi>=0.110" "uvicorn>=0.29" "httpx>=0.27" stim pymatching sinter ldpc beliefmatching psutil matplotlib tabulate scipy
.\.venv\Scripts\python.exe -m pytest python/tests -q --tb=short
```

## Optional feature builds

The first install uses `--no-default-features` because it is the safest public CPU path.

```powershell
# CUDA only
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features --features cuda

# OpenCL only
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features --features opencl

# Full infrastructure build
.\.venv\Scripts\python.exe -m maturin develop --release --features full
```

## Common fixes

If `maturin` is missing, install it with the venv Python:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip maturin
```

If Rust is missing, install Rust and restart the shell.

If GPU feature builds fail, build the CPU-safe path first:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release --no-default-features
```