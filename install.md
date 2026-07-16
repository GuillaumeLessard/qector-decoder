# Installation Guide — QECTOR Decoder v0.6.6

> ⚠️ **LICENSING & COMPLIANCE NOTICE**
>
> **QECTOR Decoder is source-available, not open source.** It is free for personal, academic, and non-commercial research use.
>
> **Commercial, institutional, lab, or product-integration use requires a paid commercial license.**
> * **View pricing & evaluation access:** [https://qector.store/pricing](https://qector.store/pricing)
> * **Official contact:** [admin@qector.store](mailto:admin@qector.store)

---

## 📦 Recommended Installation (Pre-compiled Binary)

The public repository serves as the Python interface, documentation, and test suite. The high-performance proprietary Rust core is **not included** in this public repository.

For all standard use cases, install the pre-compiled, optimized binary wheel directly from PyPI. This guarantees bit-identical, syndrome-faithful performance without requiring any local C++ or Rust toolchains.

### 1. Create a Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
