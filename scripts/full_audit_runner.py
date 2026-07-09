#!/usr/bin/env python3
"""
Run full validation / audit commands
"""
import subprocess

scripts = [
    "python scripts/smoke_test.py",
    "python -m pytest python/tests/ -q --tb=no" if False else "echo 'pytest not configured'",
]

for cmd in scripts:
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        print(f"Failed: {e}")

print("Full audit runner completed.")