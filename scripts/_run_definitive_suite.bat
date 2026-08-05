@echo off
REM Full-suite run with the GPU VISIBLE (Enterprise, dev.bat): after the
REM primary-context sharing fix, the Rust CUDA path and CuPy now share one
REM device context, so the full suite should complete without the access
REM violation that previously killed it when both GPU paths were active.
cd /d "%~dp0.."
call dev.bat .venv\Scripts\python.exe -m pytest python\tests -q --timeout=300 -p no:cacheprovider > test-results\pytest_gpuvisible.txt 2>&1

