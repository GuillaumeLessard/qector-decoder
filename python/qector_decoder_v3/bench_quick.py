"""bench_quick - smoke diagnostic for qector-decoder-v3 v0.6.9"""

from __future__ import annotations

import platform
import sys
import time

# A5: every guard in this module caught `RuntimeError`, which is not a base of
# ImportError / PackageNotFoundError / OSError. A missing optional package or an
# absent GPU driver therefore propagated instead of degrading — in a *diagnostic*
# tool, whose whole job is to keep reporting when things are missing.


def _ver():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("qector-decoder-v3")
    except PackageNotFoundError:
        pass
    try:
        from . import __version__ as v

        return v
    except ImportError:
        return "unknown"


def _detect_avx2():
    try:
        import cpuinfo

        flags = cpuinfo.get_cpu_info().get("flags", [])
        return "avx2" in flags
    except (ImportError, AttributeError, OSError):
        pass
    try:
        import os

        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as fh:
                txt = fh.read().lower()
            return "avx2" in txt
    except OSError:
        pass
    return "unknown"


def main():
    print(f"QECTOR {_ver()} on {platform.platform()} Python {sys.version.split()[0]}")

    try:
        from . import CUDABatchDecoder

        print(
            f"CUDA available: {CUDABatchDecoder.is_available() if hasattr(CUDABatchDecoder, 'is_available') else False}"
        )
    except (ImportError, OSError, AttributeError, RuntimeError) as e:
        print(f"CUDA: False ({e})")

    try:
        from . import OpenCLBatchDecoder

        print(
            f"OpenCL available: {OpenCLBatchDecoder.is_available() if hasattr(OpenCLBatchDecoder, 'is_available') else False}"
        )
    except (ImportError, OSError, AttributeError, RuntimeError) as e:
        print(f"OpenCL: False ({e})")

    print(f"AVX2: {_detect_avx2()}")

    try:
        import numpy as np

        from . import BlossomDecoder, UnionFindDecoder

        checks = [[0, 1], [1, 2], [2, 3], [3, 4]]
        nq = 5
        syn = np.array([0, 1, 0, 0], dtype=np.uint8)

        uf = UnionFindDecoder(checks, nq)
        bl = BlossomDecoder(checks, nq)

        t0 = time.perf_counter()
        r1 = uf.decode(syn)
        t1 = time.perf_counter()
        print(f"UnionFind decode ok {r1} {(t1 - t0) * 1e6:.1f} us")

        t0 = time.perf_counter()
        r2 = bl.decode(syn)
        t1 = time.perf_counter()
        print(f"Blossom decode ok {r2} {(t1 - t0) * 1e6:.1f} us")
    except (ImportError, ValueError, TypeError, RuntimeError) as e:
        print(f"Quick decode failed: {e}")
        return 1

    print("bench_quick ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
