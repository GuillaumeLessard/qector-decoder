Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\Admin\Desktop\qector-build-rc-0.6.2"
Set-Location $RepoRoot

& ".\.venv\Scripts\python.exe" -m pip install dist/*.whl --force-reinstall --no-deps
& ".\.venv\Scripts\python.exe" -c "import qector_decoder_v3; print(qector_decoder_v3.__version__); print(qector_decoder_v3.CUDABatchDecoder.is_available()); print(qector_decoder_v3.OpenCLBatchDecoder.is_available())"
& ".\.venv\Scripts\python.exe" -m qector_decoder_v3.bench_quick
