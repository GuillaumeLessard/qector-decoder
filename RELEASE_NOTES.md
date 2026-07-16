# Changelog

All notable changes to QECTOR Decoder will be documented in this file.

## [0.6.6] - 2026-07-12
**Focus**: Critical stability fix and production hardening.

### Fixed
- **Critical Import Failure**: Resolved an `AttributeError` on `OpenCLBatchDecoder` during module initialization. In v0.6.5, an unguarded import was left in `__init__.py`, which failed on wheels built with `--no-default-features --features cuda`. This has been removed, allowing the properly guarded fallback to execute as intended.
- **Hypergraph Validation**: `UnionFindDecoder` and `FastUnionFindDecoder` now explicitly reject hypergraph codes where any qubit participates in >2 checks (`UfGraph::new` returns `Result<Self, String>`), eliminating silent syndrome-invalid corrections.
- **Input Validation**: Added comprehensive validation for empty, negative, duplicate, range, `u32::MAX`, and non-integer types, raising clean `ValueError`/`TypeError`.
- **Namespace Leakage**: Removed `os`, `sys`, `subprocess`, and `np` from the public `__init__.py` module scope.
- **Routing Safety**: `recommend_decoder` now safely avoids recommending the UF family on hypergraphs.

### Changed
- Packaging: `sdist` is now published alongside wheels. Wheel matrix expanded for Linux, Windows, and macOS arm64.
- Testing: Expanded test matrix and relaxed d=21 latency threshold for CI stability.

---

## [0.6.5] - 2026-07-12 `[YANKED]`
*This release was yanked from PyPI shortly after publication.*
- **Reason**: Critical import failure on all published wheels due to an unguarded `OpenCLBatchDecoder` reference in `__init__.py`. CI builds excluded OpenCL, causing an immediate `AttributeError` for all users. 
- **Resolution**: All users should upgrade directly to `v0.6.6`.

---

## [0.6.4] - 2026-07-10 `[YANKED]`
*This release was yanked from PyPI shortly after publication.*
- **Reason**: Internal CI/CD pipeline misconfiguration resulted in incomplete artifact publishing. 
- **Resolution**: Superseded by `v0.6.6`.

---

## [0.6.3] - 2026-07-10 `[YANKED]`
*This release was yanked from PyPI shortly after publication.*
- **Reason**: GitHub Actions secrets misconfiguration during the Rust build step.
- **Resolution**: Superseded by `v0.6.6`.

---

## [0.6.2] - 2026-07-07
**Focus**: Production hardening, correctness, and audit remediation.

### Highlights
- Comprehensive input validation and improved NumPy type coercion (`np.int*`, `np.bool_`).
- All docs, versioning, and metadata aligned to 0.6.2.
- *(Note: This was the last stable release prior to the v0.6.6 corrective rollout).*

---

## [0.6.0] - 2026-07-05
**Focus**: API drift correction and Python 3.9 compatibility.

### Fixed
- **API Drift**: Updated `README.md` and `PYPI_README.md` Stim detector-error-model quick-start examples to use `from_stim_detector_error_model` instead of the removed `stim_circuit_to_check_matrix`.
- **Python 3.9 Compatibility**: Replaced PEP 604 `X | None` union syntax with `typing.Optional`/`typing.Union` in `backend.py`, `qiskit_plugin.py`, `stim_compat.py`, and `__init__.py`.

### Changed
- Package metadata (`pyproject.toml`, `Cargo.toml`, `Cargo.lock`, runtime fallback version, `CITATION.cff`, `codemeta.json`) bumped to `0.6.0`.

---

## [0.5.9] - 2026-07-02
**Focus**: GPU acceleration, routing, and streaming workflows.

### Added
- CuPy-accelerated GPU backend (`gpu_backend.py`, `bp_cupy.py`).
- Automatic decoder backend routing (`routing.py`).
- Streaming/sliding-window decoding sessions (`streaming.py`).

### Removed
- Superseded `advanced.py` module and due-diligence bundle helper scripts.

---

## [0.5.0] - 2026-06-23
**Codename**: Lepton

### Fixed
- **Blossom exactness at large distance**: `BlossomDecoder` now uses an adaptive candidate cap `k = max(12, 4·√n_defects)`, restoring exact-MWPM logical-error-rate parity with PyMatching through d=15.

### Added
- **QECTOR Workbench**: Headless, fully-tested controller for benchmark jobs and JSON/CSV/PDF report generation.
- **Expanded validation suite**: 832 tests green, covering exact-MWPM parity, DEM-collapse equivalence, belief-matching cross-checks, and GPU CPU-bit-identity.
- **Full technical report**: Regenerated for 0.5.0, detailing accuracy parity and a ~0.8% threshold.

---

## 🛡️ License & Commercial Use Notice

QECTOR Decoder is released under the **QECTOR Source-Available License v1.0**. 
- **Free** for personal, academic, educational, and non-commercial research use.
- **Commercial, institutional, lab, or product-integration use requires a paid commercial license.**

For licensing inquiries, source-review access, or enterprise deployment, please contact:
- **Email**: [admin@qector.store](mailto:admin@qector.store)
- **Web**: [https://qector.store/pricing](https://qector.store/pricing)
- **Provenance**: Protected by timestamped archival (Zenodo DOI).

See `LICENSE` and `COMMERCIAL.md` for full terms.
