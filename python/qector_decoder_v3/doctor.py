"""qector-doctor - one-command environment diagnostic (todo8 item 0.5).

``qector-bench-quick`` answers "does a decode run here?". This answers the
question that actually costs people hours: *why does my install behave
differently from the one the docs describe?* Every check that can fail reports
**why** rather than only that it failed, and carries a remedy line.

The four failure modes this exists for, in the order they have burned this
project:

1. **The installed wheel is not the working tree.** ``python/tests/*`` and
   ``import qector_decoder_v3`` both load ``site-packages``, so a Python-source
   edit changes nothing until the wheel is rebuilt and reinstalled. A stale
   wheel once produced a phantom 26x performance regression. Check
   ``working-tree`` compares the imported package against a checkout when one is
   visible.
2. **A GPU is present but the licence tier gates it.** ``cuda_is_available()``
   reports the *hardware*; ``CUDABatchDecoder(...)`` additionally requires
   Enterprise. Check ``cuda`` constructs a two-node decoder to tell those apart,
   because "GPU detected" plus "GPU decoder refuses to build" is otherwise
   indistinguishable from a driver problem.
3. **A missing optional extra** silently changes which code path runs.
4. **A licence key that is configured but not accepted**, which surfaces much
   later as an unexplained distance cap.

Usage::

    qector-doctor              # human-readable report
    qector-doctor --json       # machine-readable, for support bundles
    qector-doctor --strict     # exit non-zero on WARN as well as FAIL

Exit status is 0 unless a check FAILs (``--strict``: unless a check FAILs or
WARNs). A machine with no GPU and a Community licence exits 0: neither is a
defect.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any

# Status ranks, worst last. Used for the summary and the exit code.
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}

# Optional backends, in the order they matter to a decode workflow. The remedy
# is the extra that installs each one (see [project.optional-dependencies]).
_OPTIONAL_BACKENDS = (
    ("stim", "circuit sampling and detector error models", 'pip install "qector-decoder-v3[stim]"'),
    ("sinter", "the sinter benchmarking harness", 'pip install "qector-decoder-v3[stim]"'),
    ("pymatching", "PyMatching cross-checks and the Matching shim", 'pip install "qector-decoder-v3[stim]"'),
    ("cupy", "the CuPy batched-BP GPU path", 'pip install "qector-decoder-v3[cupy-cuda12x]"'),
    ("torch", "the neural / GNN predecoders", "pip install torch"),
)


class Check:
    """One diagnostic line: a status, what was observed, and what to do."""

    __slots__ = ("detail", "name", "remedy", "status")

    def __init__(self, name: str, status: str, detail: str, remedy: str = "") -> None:
        self.name = name
        self.status = status
        self.detail = detail
        self.remedy = remedy

    def as_dict(self) -> dict[str, str]:
        return {"check": self.name, "status": self.status, "detail": self.detail, "remedy": self.remedy}


def _dist_version() -> str | None:
    """Version recorded in the installed distribution metadata, if any."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("qector-decoder-v3")
    except PackageNotFoundError:
        return None


def _module_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    # cupy ships under several distribution names (cupy, cupy-cuda12x, ...);
    # fall back to the module attribute rather than reporting "unknown".
    for dist in (name, f"{name}-cuda12x", f"{name}-cuda11x"):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:  # pragma: no cover - defensive; the caller already imported it
        return "unknown"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_package(qd: Any) -> list[Check]:
    """Where the package was imported from, and whether its versions agree."""
    out: list[Check] = []
    path = getattr(qd, "__file__", "<unknown>")
    runtime = getattr(qd, "__version__", "unknown")
    python_layer = getattr(qd, "__fallback_version__", None)
    dist = _dist_version()

    out.append(Check("package", PASS, f"qector_decoder_v3 {runtime} imported from {path}"))

    if dist is None:
        out.append(
            Check(
                "package-metadata",
                WARN,
                "no installed distribution metadata for 'qector-decoder-v3'",
                "The package is being imported from a source tree rather than an installed "
                "wheel. Install it (pip install <wheel>) before trusting version-sensitive results.",
            )
        )
    elif dist != runtime:
        out.append(
            Check(
                "package-metadata",
                FAIL,
                f"distribution metadata says {dist}, the loaded core reports {runtime}",
                "The compiled core and the wheel metadata come from different builds. "
                "Rebuild and reinstall: maturin build --release && "
                "pip install --force-reinstall --no-deps <wheel>",
            )
        )
    else:
        out.append(Check("package-metadata", PASS, f"distribution metadata and runtime agree ({dist})"))

    # __version__ is read from the compiled core; __fallback_version__ is the
    # pyproject/Cargo version baked into the Python layer. A difference means
    # the Python source was updated but the Rust core was not rebuilt.
    if python_layer is not None and python_layer != runtime:
        out.append(
            Check(
                "core-version-skew",
                WARN,
                f"Python layer declares {python_layer}, compiled core reports {runtime}",
                "The Rust core predates the Python source in this install. Rebuild it: "
                "maturin build --release && pip install --force-reinstall --no-deps <wheel>",
            )
        )
    else:
        out.append(Check("core-version-skew", PASS, f"Python layer and compiled core both at {runtime}"))
    return out


def check_native_core(qd: Any) -> Check:
    """The compiled PyO3 extension: present, and exporting the decoder families."""
    native = getattr(qd, "_native_module", None)
    if native is None:
        return Check(
            "native-core",
            FAIL,
            "the compiled extension module could not be imported",
            "Nothing in this package works without it. Reinstall a wheel for this "
            "interpreter (python -c 'import sys; print(sys.version)') or build from "
            "source with: maturin develop --release",
        )

    # A `_guard()` stub imports fine and only raises when constructed, which is
    # exactly how the 0.6.5 / 0.6.7 yanks reached users. Detect it by repr.
    stubbed = [
        name
        for name in ("UnionFindDecoder", "BlossomDecoder", "SparseBlossomDecoder", "BPOSDDecoder", "BatchDecoder")
        if repr(getattr(qd, name, None)).startswith("<unavailable ")
    ]
    if stubbed:
        return Check(
            "native-core",
            FAIL,
            f"core decoders resolved to unavailable stubs: {', '.join(stubbed)}",
            "The wheel was built without the symbols these names need; they raise "
            "RuntimeError on construction. Rebuild from source: maturin develop --release",
        )
    return Check("native-core", PASS, f"compiled extension loaded from {getattr(native, '__file__', '<builtin>')}")


def check_working_tree(qd: Any, repo: str | None) -> Check:
    """Compare the imported package against a source checkout, when one is visible.

    This is the check that exists because of a phantom performance regression:
    three sessions measured a working-tree edit that was never in the wheel they
    were importing.
    """
    import os

    root = repo or os.getcwd()
    source = os.path.join(root, "python", "qector_decoder_v3", "__init__.py")
    installed = getattr(qd, "__file__", None)

    if not os.path.isfile(source):
        return Check("working-tree", PASS, f"no source checkout at {root} - nothing to compare against")
    if installed is None:
        return Check("working-tree", WARN, "the imported package has no __file__ to compare", "")
    if os.path.realpath(source) == os.path.realpath(installed):
        return Check("working-tree", PASS, "importing the source checkout directly (editable/source install)")

    try:
        with open(source, "rb") as fh:
            src_bytes = fh.read()
        with open(installed, "rb") as fh:
            inst_bytes = fh.read()
    except OSError as exc:
        return Check("working-tree", WARN, f"could not compare source and install: {exc}", "")

    if src_bytes == inst_bytes:
        return Check("working-tree", PASS, f"installed package matches {source}")
    return Check(
        "working-tree",
        WARN,
        f"installed __init__.py differs from {source} ({len(inst_bytes)} vs {len(src_bytes)} bytes)",
        "You are testing the installed wheel, not your edits. Benchmarks and test "
        "results from this state describe the wheel. Rebuild and reinstall: "
        "maturin build --release && pip install --force-reinstall --no-deps <wheel>",
    )


def check_decode(qd: Any) -> Check:
    """Construct and run the two headline CPU decoders on a repetition chain.

    Presence is not the property users depend on; ``H @ correction == syndrome``
    is. A repetition chain puts every qubit in at most two checks, so it is legal
    input for every decoder family including Union-Find.
    """
    import numpy as np

    n = 8
    c2q = [[i, i + 1] for i in range(n - 1)]
    syn = np.zeros(n - 1, dtype=np.uint8)
    syn[2] = 1
    syn[5] = 1
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for ci, qs in enumerate(c2q):
        for q in qs:
            H[ci, q] ^= 1

    for name in ("UnionFindDecoder", "BlossomDecoder"):
        try:
            dec = getattr(qd, name)(c2q, n)
            corr = np.asarray(dec.decode(syn), dtype=np.uint8)
        except Exception as exc:
            return Check(
                "decode",
                FAIL,
                f"{name} could not decode a repetition chain: {type(exc).__name__}: {exc}",
                "This install cannot decode at all. Reinstall the wheel; if it persists, "
                "report it with this doctor output attached.",
            )
        if corr.shape != (n,) or not np.array_equal((H @ corr) & 1, syn):
            return Check(
                "decode",
                FAIL,
                f"{name} returned a correction that does not reproduce the syndrome: {corr.tolist()}",
                "Syndrome faithfulness (H @ correction == syndrome) is the core contract. "
                "Report this with the doctor output attached.",
            )
    return Check("decode", PASS, "UnionFindDecoder and BlossomDecoder decode faithfully (H @ c == s)")


def check_optional_backends() -> list[Check]:
    """Report which optional third-party backends can actually be imported."""
    import importlib.util

    out: list[Check] = []
    for name, purpose, remedy in _OPTIONAL_BACKENDS:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):  # a broken partial install
            found = False
        if found:
            out.append(Check(f"backend:{name}", PASS, f"{name} {_module_version(name)} available - {purpose}"))
        else:
            out.append(
                Check(
                    f"backend:{name}",
                    WARN,
                    f"{name} not importable - {purpose} is unavailable",
                    remedy,
                )
            )
    return out


def _classify_gpu_error(exc: BaseException) -> tuple[str, str]:
    """Map a GPU decoder construction failure to ``(detail, remedy)``.

    The licence gate and a driver fault both surface as ``RuntimeError`` across
    the PyO3 boundary (``enforce_gpu_access`` is mapped with
    ``PyRuntimeError::new_err``), so they can only be told apart by the message.
    Getting this wrong is the difference between "buy a licence" and "fix your
    driver".
    """
    msg = str(exc)
    low = msg.lower()
    if "enterprise" in low or "tier" in low:
        return (
            f"device present but gated by licence tier: {msg}",
            "The hardware is fine. GPU decoding requires the Enterprise tier - set "
            "QECTOR_LICENSE_KEY (or QECTOR_LICENSE_FILE) to an Enterprise key. "
            "See https://qector.store/pricing",
        )
    if "not available" in low or "without cuda" in low or "cuda-only wheel" in low:
        return (
            f"not compiled into this wheel: {msg}",
            "Build from source with the feature enabled: maturin develop --release "
            "--features cuda   (or --features opencl)",
        )
    return (
        f"construction failed: {type(exc).__name__}: {msg}",
        "The device was detected but the decoder could not be built - this is a "
        "driver/runtime fault rather than a licence or build problem. Check the "
        "vendor driver version.",
    )


def check_cuda(qd: Any) -> Check:
    """CUDA: is a device visible, and can a CUDA decoder actually be constructed?

    ``cuda_is_available()`` reports hardware only. Enterprise-tier enforcement
    happens in the constructor, so a Community install on a working GPU reports
    ``True`` here and still cannot decode on it.
    """
    try:
        available = bool(qd.cuda_is_available())
    except Exception as exc:  # pragma: no cover - probe must never crash the report
        return Check("cuda", WARN, f"CUDA probe raised {type(exc).__name__}: {exc}", "")

    if not available:
        return Check(
            "cuda",
            PASS,
            "no CUDA device usable (wheel built without CUDA, or no NVIDIA driver/device)",
            "Not a defect on a CPU-only machine. For GPU decoding you need an NVIDIA "
            "driver, a CUDA-enabled wheel, and an Enterprise licence.",
        )

    try:
        qd.CUDABatchDecoder([[0, 1]], 2)
    except Exception as exc:
        detail, remedy = _classify_gpu_error(exc)
        return Check("cuda", WARN, f"CUDA device detected, {detail}", remedy)
    return Check("cuda", PASS, "CUDA device detected and CUDABatchDecoder constructs")


def check_opencl(qd: Any) -> Check:
    """OpenCL: same split as CUDA, but the probe runs in a child process.

    ``opencl_is_available()`` spawns a subprocess that constructs and decodes a
    two-node batch, because some driver stacks abort the process during kernel
    setup. It therefore takes a few seconds and returns False on timeout.
    """
    try:
        available = bool(qd.opencl_is_available())
    except Exception as exc:  # pragma: no cover - probe must never crash the report
        return Check("opencl", WARN, f"OpenCL probe raised {type(exc).__name__}: {exc}", "")

    if not available:
        return Check(
            "opencl",
            PASS,
            "no OpenCL device usable (public wheels are CUDA-only by design)",
            "Not a defect. OpenCL needs a source build: maturin develop --release --features opencl",
        )

    try:
        qd.OpenCLBatchDecoder([[0, 1]], 2)
    except Exception as exc:
        detail, remedy = _classify_gpu_error(exc)
        return Check("opencl", WARN, f"OpenCL device detected, {detail}", remedy)
    return Check("opencl", PASS, "OpenCL device detected and OpenCLBatchDecoder constructs")


def check_licence(qd: Any) -> list[Check]:
    """Resolved tier, what it gates, and whether a configured key was accepted."""
    out: list[Check] = []
    try:
        info = qd.get_license_info()
    except Exception as exc:  # pragma: no cover - defensive
        return [Check("licence", WARN, f"could not read licence info: {type(exc).__name__}: {exc}", "")]

    tier = str(info.get("tier", "Community"))
    cap = info.get("max_distance", "?")
    gpu = bool(info.get("gpu_enabled", False))
    gnn = bool(info.get("gnn_enabled", False))
    gated = [
        f"code distance d<={cap}",
        f"GPU decoding {'enabled' if gpu else 'DISABLED (Enterprise only)'}",
        f"GNN/neural paths {'enabled' if gnn else 'DISABLED (Pro or Enterprise)'}",
    ]
    out.append(Check("licence-tier", PASS, f"{tier} tier - " + "; ".join(gated)))

    status = str(info.get("key_status", "unknown"))
    if status in ("valid",):
        out.append(Check("licence-key", PASS, "a licence key is configured and was accepted"))
    elif status in ("no_key", "unknown"):
        out.append(
            Check(
                "licence-key",
                PASS,
                "no licence key configured - running on the Community tier",
                "Community is a supported state, not an error. Set QECTOR_LICENSE_KEY or "
                "QECTOR_LICENSE_FILE (or ~/.qector/license.key) to raise the caps above.",
            )
        )
    else:
        out.append(
            Check(
                "licence-key",
                WARN,
                f"a licence key is configured but was not accepted (status: {status})",
                "The key is expired, revoked, or malformed, so the install has silently "
                "fallen back to Community caps. Re-issue it at https://qector.store/pricing",
            )
        )

    if bool(info.get("enforce_mode", False)):
        out.append(
            Check(
                "licence-enforce",
                WARN,
                "QECTOR_ENFORCE=1 is set - decoding hard-fails without a valid key",
                "Unset QECTOR_ENFORCE, or configure a valid key, before running workloads.",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_checks(repo: str | None = None, skip_gpu: bool = False) -> list[Check]:
    """Run every diagnostic and return the checks in report order."""
    import os

    os.environ.setdefault("QECTOR_SILENT", "1")

    try:
        import qector_decoder_v3 as qd
    except Exception as exc:
        return [
            Check(
                "package",
                FAIL,
                f"import qector_decoder_v3 failed: {type(exc).__name__}: {exc}",
                "Install a wheel built for this interpreter, or build from source: "
                "maturin develop --release",
            )
        ]

    checks: list[Check] = []
    checks += check_package(qd)
    checks.append(check_native_core(qd))
    checks.append(check_working_tree(qd, repo))
    checks.append(check_decode(qd))
    checks += check_optional_backends()
    if skip_gpu:
        checks.append(Check("cuda", PASS, "skipped (--skip-gpu)"))
        checks.append(Check("opencl", PASS, "skipped (--skip-gpu)"))
    else:
        checks.append(check_cuda(qd))
        checks.append(check_opencl(qd))
    checks += check_licence(qd)
    return checks


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qector-doctor",
        description="QECTOR Decoder v3 - environment diagnostic. Reports what is "
        "installed, what is gated, and what to do about it.",
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a report")
    p.add_argument("--strict", action="store_true", help="Exit non-zero on WARN as well as FAIL")
    p.add_argument(
        "--repo",
        default=None,
        help="Source checkout to compare the installed package against (default: cwd)",
    )
    p.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip the CUDA/OpenCL probes (the OpenCL probe spawns a child process and can take seconds)",
    )
    return p


def _print_report(checks: list[Check]) -> None:
    try:
        from qector_decoder_v3 import __version__ as ver
    except Exception:  # pragma: no cover - import already failed upstream
        ver = "unknown"
    print(f"QECTOR doctor - qector-decoder-v3 {ver}")
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print()
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{c.status}] {c.name.ljust(width)}  {c.detail}")
        if c.remedy and c.status != PASS:
            print(f"{' ' * (width + 9)}-> {c.remedy}")
    print()
    worst = max(_RANK[c.status] for c in checks)
    counts = {s: sum(1 for c in checks if c.status == s) for s in (PASS, WARN, FAIL)}
    verdict = {0: "healthy", 1: "usable, with caveats above", 2: "broken - see FAIL lines"}[worst]
    print(f"{counts[PASS]} PASS  {counts[WARN]} WARN  {counts[FAIL]} FAIL  -  {verdict}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    checks = run_checks(repo=args.repo, skip_gpu=args.skip_gpu)

    if args.json:
        print(json.dumps({"checks": [c.as_dict() for c in checks]}, indent=2))
    else:
        _print_report(checks)

    worst = max(_RANK[c.status] for c in checks)
    if worst >= _RANK[FAIL]:
        return 1
    if args.strict and worst >= _RANK[WARN]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
