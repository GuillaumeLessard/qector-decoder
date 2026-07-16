# Installation Guide — QECTOR Decoder v0.6.6

> ⚠️ ****LICENSING** & **COMPLIANCE** **NOTICE**** > **QECTOR** Decoder is **source-available**, not open source. It is free for personal, academic, and non-commercial research use. > **Commercial, institutional, lab, or product-integration use requires a paid commercial license.** > View pricing and request evaluation access at: **[https://qector.store/pricing](https://qector.store/pricing)** > Official contact: **[admin@qector.store](mailto:admin@qector.store)**

---

## 📦 Recommended Installation (Pre-compiled Binary)

The public repository serves as the Python interface, documentation, and test suite. The high-performance proprietary Rust core is **not included** in this public repository.

For all standard use cases, install the pre-compiled, optimized binary wheel directly from PyPI. This guarantees bit-identical, syndrome-faithful performance without requiring any local C++ or Rust toolchains.

### 1. Create a Virtual Environment (Recommended)

```bash # Windows py -3.11 -m venv .venv .\.venv\Scripts\Activate.ps1

# Linux / macOS

python3 -m venv .venv source .venv/bin/activate

## 2. Install via PyPI

Upgrade `pip` first to ensure compatibility with modern binary wheels, then install **QECTOR**:

```bash python -m pip install --upgrade pip python -m pip install qector-decoder-v3 ```

*(Optional)* To install with compatibility layers and benchmarking tools:

```bash python -m pip install *qector-decoder-v3[stim,bench]* ```

## 3. Verify Installation

Run the following command to confirm the binary is correctly installed and functioning:

```bash python -c "from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder; import qector_decoder_v3; print(f'**QECTOR** OK - v{qector_decoder_v3.__version__}')" ```

**Expected Output:** ```text **QECTOR** OK - v0.6.6 ```

---

## 🏢 Commercial & Source Access

If your institution requires a custom build, specific feature flags (e.g., isolated **CUDA**/OpenCL configurations), or source-code review for security due diligence, this is **only available under a commercial license or Non-Disclosure Agreement (**NDA**)**.

To request access to the restricted technical archive or discuss enterprise deployment:
- **Web**: [https://qector.store/pricing](https://qector.store/pricing)
- **Email**: [admin@qector.store](mailto:[admin@qector.store](mailto:[admin@qector.store](mailto:admin@qector.store)))

---

## 🔧 Common Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **`ERROR: Could not find a version that satisfies the requirement`** | Ensure you are using Python 3.9, 3.10, 3.11, or 3.12. Upgrade pip: `python -m pip install --upgrade pip`. |
| **`ImportError: DLL load failed / .so not found`** | The pre-compiled wheel may be incompatible with highly unusual OS configurations. Ensure you are on a standard 64-bit Windows, Linux, or macOS environment. |
| **Missing optional dependencies (e.g., `stim`, `pymatching`)** | These are not bundled in the base wheel. Install them explicitly: `python -m pip install stim pymatching sinter`. |
| **GPU acceleration not detected** | The base PyPI wheel includes CPU and standard GPU fallbacks. For specialized enterprise GPU deployments, contact `[admin@qector.store](mailto:[admin@qector.store](mailto:admin@qector.store))` for custom build options. |

---

*© **2024**–**2026** Guillaume Lessard. All Rights Reserved. Protected by timestamped archival (Zenodo **DOI**) and commercial licensing terms.* ```
