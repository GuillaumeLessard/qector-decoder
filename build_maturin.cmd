@echo off
REM Build and install the extension module into the local .venv.
REM
REM This used to `cd` into a hard-coded absolute path on one machine
REM ("...\Desktop\qector-decoder-clean"), so it either failed outright or --
REM worse -- silently built a *different* checkout than the one it lives in.
REM %~dp0 is the directory of this script, which is always the right answer.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe not found in "%~dp0".
    echo Create the venv first: python -m venv .venv ^&^& .venv\Scripts\pip install maturin
    exit /b 1
)

.venv\Scripts\python.exe -m maturin develop --release > maturin_out.log 2>&1
set BUILD_RC=%ERRORLEVEL%
if not "%BUILD_RC%"=="0" (
    echo BUILD FAILED ^(exit %BUILD_RC%^); see maturin_out.log
    echo FAILED %BUILD_RC%> maturin_done.txt
    exit /b %BUILD_RC%
)
echo DONE> maturin_done.txt
