"""GPU kernel sources are bit-identical to the CPU Union-Find, without a GPU (1.0.1).

`test_cuda_cpu_bit_identical.py` needs a CUDA device and a paid licence,
so on most CI runners it skips. This test runs the *kernel sources* on the CPU
instead:

* `src/cuda_kernels.cu::uf_decode_batch_cuda` is compiled as host C++ (g++/clang++)
  with `blockIdx`/`blockDim` shims and launched once per shot, unweighted and
  weighted, in two halves with per-half base pointers exactly as the dual-stream
  host code does;
* the OpenCL kernel string in `src/opencl_batch.rs` runs on any OpenCL CPU device
  pyopencl can see (e.g. PoCL).

Both must reproduce `UnionFindDecoder` bit-for-bit. The Rust sources are
proprietary and absent from the public repository, so the test skips there.
"""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import qector_decoder_v3 as qd
from qector_decoder_v3 import codes

ROOT = Path(__file__).resolve().parents[2]
CU = ROOT / "src" / "cuda_kernels.cu"
OCL = ROOT / "src" / "opencl_batch.rs"


def _edges(code):
    """Edge list in `UfGraph::new` order: per qubit ascending, boundary = n_checks."""
    m = code.n_checks
    q2c = [[] for _ in range(code.n_qubits)]
    for c, qs in enumerate(code.check_to_qubits):
        for x in qs:
            q2c[x].append(c)
    e = [(cs[0], cs[1] if len(cs) == 2 else m, x) for x, cs in enumerate(q2c) if len(cs) in (1, 2)]
    return [np.ascontiguousarray(col, dtype=np.uint32) for col in zip(*e)]


def _cases(seed):
    rng = np.random.default_rng(seed)
    for code in (codes.rotated_surface_code(7), codes.unrotated_surface_code(5), codes.toric_code(6),
                 codes.heavy_hex_code(5, "x"), codes.repetition_code(11)):
        H = code.parity_check_matrix()
        probs = rng.uniform(0.002, 0.08, code.n_qubits)
        err = (rng.random((300, code.n_qubits)) < probs).astype(np.uint8)
        yield code, probs, np.ascontiguousarray(((err @ H.T) & 1).astype(np.uint8))


@pytest.fixture(scope="module")
def cuda_host_lib(tmp_path_factory):
    if not CU.is_file():
        pytest.skip("proprietary kernel sources not present")
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no host C++ compiler")
    cu = CU.read_text()
    start = cu.index('extern "C" __global__ void uf_decode_batch_cuda(')
    end = cu.index('extern "C" __global__', start + 10)
    src = (
        "struct qx_dim3 { unsigned int x; };\n"
        "static qx_dim3 blockIdx{0}, blockDim{1}, threadIdx{0};\n#define __global__\n"
        + cu[start:end]
        + """
extern "C" void qx_run(unsigned int shot, const unsigned char* syn, const unsigned int* ea, const unsigned int* eb,
    const unsigned int* eq, unsigned char* corr, unsigned int* s32, unsigned char* s8, unsigned int nq, unsigned int nc,
    unsigned int nn, unsigned int ne, unsigned int nb, const float* el, float* sf) {
  blockIdx.x = shot; uf_decode_batch_cuda(syn, ea, eb, eq, corr, s32, s8, nq, nc, nn, ne, nb, el, sf);
}
"""
    )
    d = tmp_path_factory.mktemp("cuda_host")
    (d / "k.cpp").write_text(src)
    so = d / "k.so"
    subprocess.run([cxx, "-O2", "-shared", "-fPIC", "-o", str(so), str(d / "k.cpp")], check=True)
    lib = ctypes.CDLL(str(so))
    P, U = ctypes.c_void_p, ctypes.c_uint
    lib.qx_run.argtypes = [U, P, P, P, P, P, P, P, U, U, U, U, U, P, P]
    return lib


def _run_cuda_host(lib, code, S, edge_len):
    ea, eb, eq = _edges(code)
    n, m, E = code.n_qubits, code.n_checks, len(ea)
    N = m + 1
    B = len(S)
    s32_stride, s8_stride = 6 * N + 1 + 4 * E, 5 * N + E
    corr = np.zeros((B, n), np.uint8)
    s32 = np.zeros(B * s32_stride, np.uint32)
    s8 = np.zeros(B * s8_stride, np.uint8)
    sf = np.zeros(max(1, B * E), np.float32)

    def ptr(a, off=0):
        return ctypes.c_void_p(a.ctypes.data + off)

    half = B // 2
    for base, count in ((0, half), (half, B - half)):  # the two CUDA streams
        for shot in range(count):
            lib.qx_run(
                shot, ptr(S, base * m), ptr(ea), ptr(eb), ptr(eq), ptr(corr, base * n),
                ptr(s32, 4 * base * s32_stride), ptr(s8, base * s8_stride), n, m, N, E, count,
                ptr(edge_len) if edge_len is not None else None,
                ptr(sf, 4 * base * E) if edge_len is not None else None,
            )
    return corr


def _edge_len(code, w):
    """`UfGraph::set_qubit_weights`: cheapest edge = 2.0, clamp to [2, 4096], f32."""
    _, _, eq = _edges(code)
    wmin = w[eq][w[eq] > 0].min()
    return np.ascontiguousarray(np.clip(2.0 * np.maximum(w[eq], wmin) / wmin, 2.0, 4096.0).astype(np.float32))


def _cpu(code, S, w=None):
    dec = qd.UnionFindDecoder(code.check_to_qubits, n_qubits=code.n_qubits, edge_weights=w)
    return np.array([np.asarray(dec.decode(s), np.uint8) for s in S])


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_cuda_kernel_source_matches_cpu(cuda_host_lib, weighted):
    for code, probs, S in _cases(7 if weighted else 3):
        w = np.log((1 - probs) / probs) if weighted else None
        out = _run_cuda_host(cuda_host_lib, code, S, _edge_len(code, w) if weighted else None)
        ref = _cpu(code, S, w)
        same = (out == ref).all(axis=1)
        assert same.all(), f"{code.name}: {int((~same).sum())}/{len(S)} shots differ from the CPU"


def test_opencl_kernel_source_matches_cpu():
    if not OCL.is_file():
        pytest.skip("proprietary kernel sources not present")
    cl = pytest.importorskip("pyopencl")
    try:
        platforms = cl.get_platforms()
    except Exception:  # no ICD / driver
        pytest.skip("no OpenCL platform")
    devices = [dv for p in platforms for dv in p.get_devices()]
    if not devices:
        pytest.skip("no OpenCL device")
    kernel = re.search(r'const KERNEL_GLOBAL: &str = r#"(.*?)"#;', OCL.read_text(), re.S).group(1)
    ctx = cl.Context(devices[:1])
    queue = cl.CommandQueue(ctx)
    knl = cl.Kernel(cl.Program(ctx, kernel).build(), "uf_decode_batch")
    mf = cl.mem_flags
    for code, _, S in _cases(5):
        ea, eb, eq = _edges(code)
        B, N, E = len(S), code.n_checks + 1, len(ea)
        bufs = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a) for a in (S, ea, eb, eq)]
        out = np.zeros((B, code.n_qubits), np.uint8)
        obuf = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)
        s32 = cl.Buffer(ctx, mf.READ_WRITE, 4 * B * (6 * N + 1 + 4 * E))
        s8 = cl.Buffer(ctx, mf.READ_WRITE, B * (5 * N + E))
        knl(queue, (B,), None, *bufs, obuf, s32, s8, *(np.uint32(v) for v in (code.n_qubits, code.n_checks, N, E, B)))
        cl.enqueue_copy(queue, out, obuf)
        queue.finish()
        same = (out == _cpu(code, S)).all(axis=1)
        assert same.all(), f"{code.name}: {int((~same).sum())}/{len(S)} shots differ from the CPU"
