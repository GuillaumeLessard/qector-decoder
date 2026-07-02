# QECTOR Decoder v3 (v0.5.7) - PyPI Release Bundle

Source-available package + twine-validated distribution artifacts.

## dist/ (the PyPI upload)
- qector_decoder_v3-0.5.7-cp311-cp311-win_amd64.whl  (CPython 3.11, Windows x64)
- qector_decoder_v3-0.5.7.tar.gz  (source distribution)
Both PASSED: python -m twine check

## Publish
    python -m twine upload dist/*
(or the OIDC Trusted Publisher CI workflow; no stored token needed)

## Notes
- Only the cp311/win_amd64 wheel is built locally; full multi-platform wheels are a CI step.
- src/ is a compile_error stub by design; the Rust core compiles into the wheel.
- v0.5.7. Simulation/software-validated; not a real-time or fault-tolerant hardware decoder.