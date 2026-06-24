@echo off
set PATH=C:\tmp_rust;%PATH%
set PYO3_PYTHON=D:\QECTOR\app\.venv\Scripts\python.exe
cd /D "D:\QECTOR\Qiskit\qector-decoder-v3"
cargo clean > build_log.txt 2>&1
cargo build --release >> build_log.txt 2>&1
echo BUILD_DONE >> build_log.txt
