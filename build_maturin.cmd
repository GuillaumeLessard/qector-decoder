@echo off
cd /d "C:\Users\Clinque du Batiment\Desktop\qector-decoder-clean"
.venv\Scripts\python.exe -m maturin develop --release > maturin_out.log 2>&1
echo DONE> maturin_done.txt
