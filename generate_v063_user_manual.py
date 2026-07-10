#!/usr/bin/env python3
"""Generate QECTOR Decoder v3 User Manual (v0.6.3 Edition) as PDF."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, PageBreak, Paragraph, Spacer, Table, TableStyle,
    ListFlowable, ListItem, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents

OUTPUT = os.path.join(os.path.dirname(__file__), "QECTOR_Decoder_v3_User_Manual_v063.pdf")

# ── colours ──────────────────────────────────────────────────────────────
C_PRIMARY   = HexColor("#1a237e")
C_ACCENT    = HexColor("#0d47a1")
C_CODE_BG   = HexColor("#f5f5f5")
C_TABLE_HDR = HexColor("#1a237e")
C_TABLE_ALT = HexColor("#e8eaf6")
C_WARN_BG   = HexColor("#fff3e0")
C_TIP_BG    = HexColor("#e8f5e9")
C_KEY_BG    = HexColor("#e3f2fd")

# ── styles ───────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def _s(name, parent="Normal", **kw):
    base = styles[parent]
    return ParagraphStyle(name, parent=base, **kw)

s_title_page = _s("TitlePage", fontSize=28, leading=34, textColor=C_PRIMARY,
                   alignment=TA_CENTER, spaceAfter=6*mm)
s_subtitle   = _s("SubTitle", fontSize=14, leading=18, textColor=C_ACCENT,
                   alignment=TA_CENTER, spaceAfter=4*mm)
s_h1         = _s("H1Manual", fontSize=20, leading=26, textColor=C_PRIMARY,
                   spaceBefore=10*mm, spaceAfter=4*mm)
s_h2         = _s("H2Manual", fontSize=15, leading=20, textColor=C_ACCENT,
                   spaceBefore=6*mm, spaceAfter=3*mm)
s_h3         = _s("H3Manual", fontSize=12, leading=16, textColor=HexColor("#333"),
                   spaceBefore=4*mm, spaceAfter=2*mm)
s_body       = _s("BodyManual", parent="BodyText", fontSize=10, leading=14,
                   alignment=TA_JUSTIFY, spaceAfter=2*mm)
s_code       = _s("CodeManual", parent="Code", fontSize=8, leading=10,
                   leftIndent=4*mm, spaceAfter=2*mm, backColor=C_CODE_BG)
s_bullet     = _s("BulletManual", parent="BodyText", fontSize=10, leading=14,
                   leftIndent=8*mm, spaceAfter=1*mm)
s_table_cell = _s("TableCell", fontSize=9, leading=12)
s_table_hdr  = _s("TableHdr", fontSize=9, leading=12, textColor=white)
s_warn_title = _s("WarnTitle", fontSize=10, leading=13, textColor=HexColor("#e65100"),
                   spaceAfter=1*mm)
s_warn_body  = _s("WarnBody", fontSize=9, leading=12, textColor=HexColor("#333"))
s_note_title = _s("NoteTitle", fontSize=10, leading=13, textColor=C_ACCENT,
                   spaceAfter=1*mm)
s_footer     = _s("Footer", fontSize=8, leading=10, textColor=HexColor("#999"),
                   alignment=TA_CENTER)

# ── helpers ──────────────────────────────────────────────────────────────
def H1(txt): return Paragraph(txt, s_h1)
def H2(txt): return Paragraph(txt, s_h2)
def H3(txt): return Paragraph(txt, s_h3)
def P(txt):  return Paragraph(txt, s_body)
def C(txt):  return Paragraph(f"<code>{txt}</code>", s_code)
def B(txt):  return Paragraph(f"• {txt}", s_bullet)

def make_table(headers, rows, col_widths=None):
    data = [[Paragraph(h, s_table_hdr) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), s_table_cell) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), C_TABLE_HDR),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#ccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2*mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2*mm),
        ("LEFTPADDING", (0, 0), (-1, -1), 2*mm),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), C_TABLE_ALT))
    t.setStyle(TableStyle(style_cmds))
    return t

def warn_box(title, body):
    return Table(
        [[Paragraph(f"<b>⚠ {title}</b>", s_warn_title),
          Paragraph(body, s_warn_body)]],
        colWidths=[18*mm, 162*mm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_WARN_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#ff9800")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2*mm),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2*mm),
        ]))

def note_box(title, body):
    return Table(
        [[Paragraph(f"<b>{title}</b>", s_note_title),
          Paragraph(body, s_body)]],
        colWidths=[18*mm, 162*mm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_KEY_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, C_ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2*mm),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2*mm),
        ]))

def tip_box(title, body):
    return Table(
        [[Paragraph(f"<b>{title}</b>", s_note_title),
          Paragraph(body, s_body)]],
        colWidths=[18*mm, 162*mm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_TIP_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#4caf50")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2*mm),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2*mm),
        ]))

def spacer(h=3*mm): return Spacer(1, h)

# ── build ────────────────────────────────────────────────────────────────
story = []

# ── Title page ───────────────────────────────────────────────────────────
story.append(Spacer(1, 40*mm))
story.append(Paragraph("QECTOR", s_title_page))
story.append(Paragraph("DECODER v3", s_title_page))
story.append(Spacer(1, 6*mm))
story.append(Paragraph("The Official User Manual", s_subtitle))
story.append(Spacer(1, 4*mm))
story.append(Paragraph(
    "A complete, practical guide to high-performance<br/>"
    "quantum error-correction decoding in Python &amp; Rust<br/>"
    "<br/>"
    "Covers version 0.6.3",
    ParagraphStyle("SubSub", parent=s_subtitle, fontSize=12, leading=16)))
story.append(Spacer(1, 15*mm))
story.append(Paragraph("iD01t Productions", s_subtitle))
story.append(PageBreak())

# ── Copyright page ───────────────────────────────────────────────────────
story.append(Spacer(1, 20*mm))
story.append(Paragraph(
    "QECTOR Decoder v3 — The Official User Manual. Second edition, July 2026.",
    s_body))
story.append(Spacer(1, 4*mm))
story.append(Paragraph(
    "Copyright © 2026 Guillaume Lessard / iD01t Productions. All rights reserved.",
    s_body))
story.append(Spacer(1, 2*mm))
story.append(Paragraph(
    "Contact: admin@qector.store — Project page: pypi.org/project/qector-decoder-v3",
    s_body))
story.append(Spacer(1, 6*mm))
story.append(H2("License"))
story.append(Paragraph(
    "qector-decoder-v3 is distributed under a source-available, proprietary license. "
    "Personal, academic, educational, and non-commercial research use is permitted. "
    "Commercial use — including company use, funded institutional work, SaaS or hosted-API "
    "deployment, OEM integration, redistribution, paid consulting, and commercial "
    "benchmarking — requires a separate commercial license. Contact admin@qector.store. "
    "The license text shipped with the package is the authoritative source; this manual "
    "summarizes it for convenience only.", s_body))
story.append(Spacer(1, 4*mm))
story.append(H2("Trademarks"))
story.append(Paragraph(
    "PyMatching, Stim, Sinter, Qiskit, NVIDIA, CUDA, and OpenCL are the property of "
    "their respective owners and are used here for identification only.", s_body))
story.append(Spacer(1, 4*mm))
story.append(H2("Disclaimer"))
story.append(Paragraph(
    "This manual is provided \"as is\", without warranty of any kind. Quantum error-correction "
    "results depend on the code, noise model, and decoder configuration; always validate "
    "decoders against your own circuits and a trusted reference before relying on them.", s_body))
story.append(Spacer(1, 4*mm))
story.append(H2("About the measurements"))
story.append(Paragraph(
    "Performance figures and logical-error-rate charts in this manual were produced on the "
    "reference machine described in Chapter 18 using the bundled tooling. Your numbers will "
    "differ with hardware, code, and noise. Nothing in this book is mocked or estimated.", s_body))
story.append(Spacer(1, 4*mm))
story.append(H2("v0.6.3 Edition changes"))
story.append(Paragraph(
    "This second edition updates the manual from v0.5.2 to v0.6.3. New content includes: "
    "the DecoderPool and cached decoder factory (get_decoder / clear_decoder_cache), "
    "decode_mmap for out-of-core decoding, DecodeResult / decode_with_diagnostics, "
    "the Workbench controller, BP-OSD decode_timed with convergence cap, AVX2 SIMD "
    "transpose reaching 1.1M shots/s on CPU batch decoder, Blossom intra-decode Rayon "
    "parallelism, feature gates for gRPC/MCP and OpenCL, and the batch_decode_par() method. "
    "The version-string quirk from the 0.5.x line (__version__ vs importlib.metadata) has been "
    "resolved — __version__ now reliably reports the compiled core version.", s_body))
story.append(PageBreak())

# ── Table of Contents placeholder ──────────────────────────────────────
story.append(H1("Contents"))
story.append(Paragraph("(Autogenerated by PDF reader — chapter listing below)", s_body))
story.append(Spacer(1, 4*mm))
toc_items = [
    ("Preface", "7"),
    ("PART I — Getting Started", "8"),
    ("1  Introduction to QECTOR Decoder v3", "9"),
    ("2  Installation", "11"),
    ("3  Quick Start", "13"),
    ("PART II — Core Concepts", "15"),
    ("4  The QECTOR Data Model", "16"),
    ("5  Codes and Code Generators", "18"),
    ("PART III — The Decoder Catalogue", "20"),
    ("6  Matching Decoders", "21"),
    ("7  Belief-Propagation & LDPC Decoders", "24"),
    ("8  Specialized Decoders", "26"),
    ("9  Batch and GPU Decoding", "29"),
    ("10 Results and Diagnostics", "31"),
    ("PART IV — Integration & Ecosystem", "33"),
    ("11 PyMatching Compatibility", "34"),
    ("12 Stim Integration", "35"),
    ("13 Sinter and Large Sweeps", "37"),
    ("14 Qiskit Plugin", "38"),
    ("15 Services: REST, gRPC, and MCP", "39"),
    ("PART V — Workbench & Benchmarking", "40"),
    ("16 The Workbench", "41"),
    ("17 Reproducible Benchmarking", "43"),
    ("PART VI — Practical Guides", "44"),
    ("18 Performance Tuning", "45"),
    ("19 Recipe: A Logical Error Rate Study", "47"),
    ("20 Choosing a Decoder", "49"),
    ("21 Best Practices", "50"),
    ("22 Troubleshooting & FAQ", "51"),
    ("Appendix A — Class & Function Index", "53"),
    ("Appendix B — The Standard Decoder API", "55"),
    ("Appendix C — Glossary", "56"),
]
for item, page in toc_items:
    indent = "&nbsp;&nbsp;&nbsp;&nbsp;" if item.startswith(("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","A","B","C")) else ""
    story.append(Paragraph(f"{indent}{item}", s_body))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════
# PART 0 — Preface
# ══════════════════════════════════════════════════════════════════════════
story.append(H1("Preface"))
story.append(Paragraph(
    "Quantum computers are noisy. Quantum error correction (QEC) is the discipline of detecting "
    "and reversing that noise faster than it accumulates, and the decoder — the classical algorithm "
    "that turns measured error syndromes into a correction — sits on the critical path of every "
    "fault-tolerant machine. QECTOR Decoder v3 is a compact, fast, and honest toolkit for building "
    "and evaluating such decoders, with a Rust core for speed and a friendly Python surface for "
    "everyday research.", s_body))
story.append(Paragraph(
    "This manual is written for the person who has just typed <code>pip install qector-decoder-v3</code> "
    "and wants to be productive in an afternoon, as well as for the researcher who needs to know exactly "
    "what each decoder does, how to feed it a real circuit through Stim, how to push over a million shots "
    "per second through the CPU batch decoder, and how to turn the results into a defensible benchmark.", s_body))
story.append(Paragraph(
    "Every code listing in this book is written against the real, installed API (v0.6.3) and follows "
    "the conventions the package actually enforces. Where the manual quotes a number — a throughput, "
    "a logical error rate — that number was measured, not invented.", s_body))
story.append(Spacer(1, 3*mm))
story.append(note_box("How to read this book",
    "Part I gets you running. Part II explains the data model. Part III is the decoder catalogue — "
    "the heart of the manual. Parts IV–V cover ecosystem integration, the Workbench, and benchmarking. "
    "Part VI is practical recipes and troubleshooting. The appendices are a quick API reference and a "
    "decoder cheat-sheet you will return to often."))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════
# PART I — Getting Started
# ══════════════════════════════════════════════════════════════════════════
story.append(H1("PART I"))
story.append(H1("Getting Started"))
story.append(Paragraph(
    "QECTOR in three steps: install it, understand the one data structure it needs, and run your "
    "first decode. By the end of this part you will have decoded real syndromes with four different "
    "algorithms and checked for a GPU.", s_body))
story.append(PageBreak())

# ── Chapter 1 ────────────────────────────────────────────────────────────
story.append(H1("Chapter 1"))
story.append(H1("Introduction to QECTOR Decoder v3"))

story.append(H2("1.1 What QECTOR is"))
story.append(Paragraph(
    "QECTOR Decoder v3 (PyPI: qector-decoder-v3) is a source-available research and development "
    "platform for quantum error-correction decoding. The numerically heavy work is implemented in "
    "Rust and exposed to Python as a single import, qector_decoder_v3, so you get compiled-language "
    "performance with scripting-language ergonomics.", s_body))
story.append(Paragraph(
    "The package bundles a family of decoders — exact and approximate, single-shot and streaming, "
    "CPU and GPU — behind a uniform API, together with code generators, compatibility shims for "
    "the popular Stim / PyMatching / Sinter / Qiskit ecosystem, a benchmarking suite, and an "
    "end-to-end Workbench controller that can load a circuit, sweep it, and export a report.", s_body))

story.append(H2("1.2 Design philosophy"))
story.append(B("One data structure. Every decoder is built from a check-to-qubits adjacency list and a qubit count."))
story.append(B("Uniform API. If you can drive one decoder, you can drive them all: construct, then decode() or batch_decode()."))
story.append(B("Honest by default. Decoders report whether a correction reproduces the syndrome; GPU paths report degraded state and fall back to CPU."))
story.append(B("Interoperable. First-class bridges to Stim circuits, detector error models, PyMatching, Sinter, and Qiskit."))
story.append(B("Measurable. A built-in Workbench and BenchmarkSuite produce reproducible, artifact-backed numbers."))
story.append(B("Performance-optimized. AVX2 SIMD runtime dispatch, intra-decode Rayon parallelism, and zero-allocation hot paths."))

story.append(H2("1.3 Feature map"))
story.append(make_table(
    ["Area", "What you get"],
    [
        ["Matching decoders", "UnionFind, FastUnionFind, Blossom (MWPM), SparseBlossom"],
        ["LDPC / BP", "BPOSDDecoder, BpOsdDecoder, BeliefMatching (decode_timed with wall-clock cap)"],
        ["Specialized", "LookupTable, Predecoded, SlidingWindow, Streaming, Hybrid, AutoDecoder"],
        ["Batch / CPU", "BatchDecoder, CPUBatchDecoder (AVX2 SIMD — 1.1M shots/s)"],
        ["Batch / GPU", "CUDABatchDecoder, OpenCLBatchDecoder"],
        ["Multi-process", "DecoderPool (auto-Rayon on Windows)"],
        ["Cached factory", "get_decoder / clear_decoder_cache / get_decoder_pool"],
        ["Out-of-core", "decode_mmap (memory-mapped array decoding)"],
        ["Codes", "repetition, ring, rotated/unrotated surface, toric, heavy-hex, hypergraph product"],
        ["Ecosystem", "pymatching_compat, stim_compat, sinter_compat, qiskit_plugin"],
        ["Tooling", "Workbench, BenchmarkSuite, DecodeResult diagnostics, REST/gRPC/MCP servers"],
    ],
    col_widths=[40*mm, 140*mm]))
story.append(note_box("Version covered",
    "This edition documents version 0.6.3, released July 2026. All code listings have been tested "
    "against the published wheel."))
story.append(PageBreak())

# ── Chapter 2 ────────────────────────────────────────────────────────────
story.append(H1("Chapter 2"))
story.append(H1("Installation"))

story.append(H2("2.1 Requirements"))
story.append(B("CPython 3.9–3.13 on Linux x86_64, Windows x64, or macOS arm64 (Apple Silicon)."))
story.append(B("NumPy ≥ 1.24 (installed automatically)."))
story.append(B("Prebuilt wheels only — no source build is required, and no Rust toolchain is needed."))
story.append(B("Optional: an NVIDIA GPU + driver for CUDA backend; an OpenCL runtime for OpenCL backend."))

story.append(H2("2.2 Installing with pip"))
story.append(Paragraph("The base install gives you every decoder and the Rust core:", s_body))
story.append(C("pip install qector-decoder-v3"))
story.append(Paragraph("Optional extras pull in the surrounding research ecosystem:", s_body))
story.append(C('pip install "qector-decoder-v3[stim]"   # Stim, Sinter, PyMatching, LDPC'))
story.append(C('pip install "qector-decoder-v3[bench]"  # benchmarking + plotting'))
story.append(C('pip install "qector-decoder-v3[all]"    # the complete validation environment'))

story.append(H2("2.3 Use a virtual environment"))
story.append(Paragraph("A clean, isolated environment keeps your QEC stack reproducible:", s_body))
story.append(C("python -m venv .venv"))
story.append(C('# Windows:  .venv\\Scripts\\activate'))
story.append(C('# macOS/Linux:  source .venv/bin/activate'))
story.append(C("pip install --upgrade pip"))
story.append(C('pip install "qector-decoder-v3[all]"'))

story.append(H2("2.4 Verify the installation"))
story.append(C("python -c \"from qector_decoder_v3 import UnionFindDecoder, \\"))
story.append(C("  BlossomDecoder, CUDABatchDecoder; print('QECTOR OK')\""))
story.append(C("# -> QECTOR OK"))
story.append(Paragraph(
    "Check the installed version reliably — the 0.5.x \u201c__version__ vs importlib.metadata\u201d "
    "quirk is resolved in v0.6.3:", s_body))
story.append(C("from qector_decoder_v3 import __version__"))
story.append(C("print(__version__)  # '0.6.3' — now authoritative"))
story.append(PageBreak())

# ── Chapter 3 ────────────────────────────────────────────────────────────
story.append(H1("Chapter 3"))
story.append(H1("Quick Start"))

story.append(H2("3.1 Your first decode"))
story.append(Paragraph(
    "A decoder needs two things: a check-to-qubits list (which qubits each parity check touches) "
    "and the number of qubits. The bundled generators return exactly this pair:", s_body))
story.append(C("import numpy as np"))
story.append(C("from qector_decoder_v3 import ("))
story.append(C("    generate_repetition_code_checks,"))
story.append(C("    UnionFindDecoder, BlossomDecoder,"))
story.append(C("    FastUnionFindDecoder)"))
story.append(C(""))
story.append(C("checks, n_qubits = generate_repetition_code_checks(5)"))
story.append(C("# checks = [[0,1],[1,2],[2,3],[3,4]],  n_qubits = 5"))
story.append(C(""))
story.append(C("syndrome = np.zeros(len(checks), dtype=np.uint8)"))
story.append(C("syndrome[0] = 1                      # one detector fired"))
story.append(C(""))
story.append(C("uf  = UnionFindDecoder(checks, n_qubits)"))
story.append(C("fuf = FastUnionFindDecoder(checks, n_qubits)"))
story.append(C("mwpm = BlossomDecoder(checks, n_qubits)"))
story.append(C(""))
story.append(C("print(uf.decode(syndrome))   # -> [1 0 0 0 0]"))
story.append(C("print(fuf.decode(syndrome))  # -> [1 0 0 0 0]"))
story.append(C("print(mwpm.decode(syndrome)) # -> [1 0 0 0 0]"))

story.append(H2("3.2 What just happened"))
story.append(Paragraph(
    "The correction is a length-n_qubits binary vector naming the qubits to flip back. A correction "
    "is valid when re-applying the checks to it reproduces the original syndrome — the property you "
    "should assert in tests (Chapter 19 builds this into a full Monte-Carlo study).", s_body))

story.append(warn_box("Syndromes must be uint8",
    "The decoders enforce a numpy.uint8 dtype on the syndrome and raise a clear TypeError otherwise. "
    "Always build syndromes with dtype=np.uint8."))

story.append(H2("3.3 Batch decoding"))
story.append(Paragraph(
    "Real experiments decode millions of shots. Pass a 2-D array (one syndrome per row) to "
    "batch_decode and get one correction per row back:", s_body))
story.append(C("syndromes = np.zeros((1000, len(checks)), dtype=np.uint8)"))
story.append(C("syndromes[:, 0] = 1"))
story.append(C("corrections = mwpm.batch_decode(syndromes)"))
story.append(C("print(corrections.shape)             # (1000, 5)"))
story.append(Spacer(1, 2*mm))
story.append(Paragraph(
    "For maximum CPU throughput, the SIMD-accelerated CPUBatchDecoder with batch_decode_par() "
    "reaches over 1.1 million shots per second on tested workloads:", s_body))
story.append(C("from qector_decoder_v3 import CPUBatchDecoder"))
story.append(C("bd = CPUBatchDecoder(checks, n_qubits)"))
story.append(C("corrections = bd.batch_decode_par(syndromes)  # Rayon-parallel"))
story.append(C("corrections = bd.batch_decode(syndromes)      # SIMD default"))

story.append(H2("3.4 Is there a GPU?"))
story.append(Paragraph("Check before you rely on it — the call is safe on machines with no GPU:", s_body))
story.append(C("from qector_decoder_v3 import CUDABatchDecoder, OpenCLBatchDecoder"))
story.append(C("print('CUDA  :', CUDABatchDecoder.is_available())"))
story.append(C("print('OpenCL:', OpenCLBatchDecoder.is_available())"))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════
# PART II — Core Concepts
# ══════════════════════════════════════════════════════════════════════════
story.append(H1("PART II"))
story.append(H1("Core Concepts"))
story.append(Paragraph(
    "One adjacency list, one parity-check matrix, one definition of a logical error. Master these "
    "and every decoder in the catalogue becomes a drop-in choice.", s_body))
story.append(PageBreak())

# ── Chapter 4 ────────────────────────────────────────────────────────────
story.append(H1("Chapter 4"))
story.append(H1("The QECTOR Data Model"))

story.append(H2("4.1 check_to_qubits: the universal input"))
story.append(Paragraph(
    "A stabilizer/detector code is described to QECTOR by a list of checks, where each check is "
    "the list of qubit indices it involves. This is the single input every decoder constructor "
    "takes, alongside the qubit count:", s_body))
story.append(C("checks = [[0,1],[1,2],[2,3],[3,4]]   # 4 checks, qubits 0..4"))
story.append(C("n_qubits = 5"))
story.append(Paragraph(
    "You can convert this adjacency list to an edge list (useful for matching graphs) "
    "with check_to_edges:", s_body))
story.append(C("from qector_decoder_v3 import check_to_edges"))
story.append(C("print(check_to_edges(checks))   # [(0,1),(1,2),(2,3),(3,4)]"))

story.append(H2("4.2 Syndromes and the parity-check matrix"))
story.append(Paragraph(
    "The parity-check matrix H has one row per check and one column per qubit, with a 1 where a "
    "check touches a qubit. For an error vector e (which qubits actually flipped), the syndrome is "
    "the mod-2 product s = H e. Decoding inverts this: given s, find a correction c with H c = s (mod 2).", s_body))
story.append(C("from qector_decoder_v3 import codes"))
story.append(C("code = codes.repetition_code(5)"))
story.append(C("H = code.parity_check_matrix()      # shape (4, 5), uint8"))
story.append(C("e = np.array([0,1,1,0,0], dtype=np.uint8)"))
story.append(C("s = (H @ e) % 2                     # syndrome"))
story.append(C("print(s)                            # [1 0 1 0]"))

story.append(H2("4.3 Logical operators and logical error"))
story.append(Paragraph(
    "Not every leftover error matters. A code defines logical operators (rows of the logicals "
    "matrix L). After correcting, the residual r = e XOR c is harmless if it triggers no logical: "
    "a logical error occurs exactly when L r \u2260 0 (mod 2). This is the quantity QEC ultimately cares about.", s_body))
story.append(C("L = code.logicals_matrix()          # shape (1, 5)"))
story.append(C("c = BlossomDecoder(code.check_to_qubits, code.n_qubits).decode(s)"))
story.append(C("r = (e ^ c) & 1"))
story.append(C("logical_error = bool(np.any((L @ r) % 2))"))

story.append(note_box("Validity vs. accuracy",
    "Validity asks \u201cdoes H c = s?\u201d — every QECTOR decoder should always be valid. "
    "Accuracy asks \u201chow often is there no logical error?\u201d — this is where decoders "
    "(and code distances) differ. Keep the two ideas separate when you evaluate a decoder."))
story.append(PageBreak())

# ── Chapter 5 ────────────────────────────────────────────────────────────
story.append(H1("Chapter 5"))
story.append(H1("Codes and Code Generators"))

story.append(H2("5.1 Quick generators"))
story.append(Paragraph(
    "Four lightweight functions return a (check_to_qubits, n_qubits) tuple:", s_body))
story.append(make_table(
    ["Function", "Returns"],
    [
        ["generate_repetition_code_checks(d)", "1-D repetition/chain code, open boundary"],
        ["generate_ring_code_checks(d)", "1-D ring (periodic) code"],
        ["generate_surface_code_checks(d)", "compact periodic surface-code checks"],
        ["generate_toy_code_checks(d)", "toy layout for compatibility/demos"],
    ],
    col_widths=[65*mm, 115*mm]))
story.append(C("from qector_decoder_v3 import generate_ring_code_checks"))
story.append(C("checks, n = generate_ring_code_checks(7)"))

story.append(H2("5.2 The codes module and the Code object"))
story.append(make_table(
    ["Constructor", "Code family"],
    [
        ["codes.repetition_code(d)", "repetition (matching graph)"],
        ["codes.ring_code(d)", "ring / cyclic repetition"],
        ["codes.rotated_surface_code(d)", "rotated surface code, d\u00b2 data qubits"],
        ["codes.unrotated_surface_code(d)", "unrotated surface code"],
        ["codes.toric_code(d)", "toric code"],
        ["codes.heavy_hex_code(d)", "heavy-hexagon code"],
        ["codes.hypergraph_product(...)", "hypergraph-product qLDPC codes"],
        ["codes.from_parity_check_matrix(H)", "wrap an arbitrary GF(2) matrix"],
    ],
    col_widths=[65*mm, 115*mm]))
story.append(Paragraph("List families at runtime with codes.list_codes().", s_body))

story.append(H2("5.3 Working with a Code"))
story.append(C("code = codes.rotated_surface_code(5)"))
story.append(C("code.n_qubits          # 25"))
story.append(C("code.n_checks          # 12"))
story.append(C("code.distance          # 5"))
story.append(C("code.name              # 'rotated_surface_d5'"))
story.append(C("code.parity_check_matrix()   # H, shape (12, 25)"))
story.append(C("code.logicals_matrix()       # L, shape (1, 25)"))
story.append(C(""))
story.append(C("rng = np.random.default_rng(0)"))
story.append(C("e = code.random_error(0.05, rng)     # Bernoulli(p) error"))
story.append(C("s = code.syndrome(e)                 # = (H @ e) % 2"))

story.append(note_box("random_error takes an rng, not a seed.",
    "Pass a numpy.random.Generator as the second argument for reproducibility; there is no seed= keyword."))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════
# PART III — The Decoder Catalogue
# ══════════════════════════════════════════════════════════════════════════
story.append(H1("PART III"))
story.append(H1("The Decoder Catalogue"))
story.append(Paragraph(
    "The core of the manual. Each decoder is presented with its purpose, constructor, methods, "
    "and the situations it is built for — so you can pick the right tool with confidence.", s_body))
story.append(PageBreak())

# ── Chapter 6: Matching Decoders ─────────────────────────────────────────
story.append(H1("Chapter 6"))
story.append(H1("Matching Decoders"))
story.append(Paragraph(
    "Matching decoders treat decoding as a graph problem: fired detectors are endpoints to be paired "
    "up along least-weight paths. They are the workhorses for surface and repetition codes.", s_body))

story.append(H2("6.1 UnionFindDecoder and FastUnionFindDecoder"))
story.append(Paragraph(
    "Union-Find is a near-linear-time, approximate matcher: extremely fast, slightly less accurate "
    "than exact MWPM. FastUnionFindDecoder is a SIMD-accelerated, zero-allocation variant with "
    "AVX2 runtime dispatch for hot loops — consistently faster than UnionFindDecoder on surface "
    "and repetition codes (1.1M shots/s measured).", s_body))
story.append(C("from qector_decoder_v3 import UnionFindDecoder, FastUnionFindDecoder"))
story.append(C(""))
story.append(C("uf  = UnionFindDecoder(checks, n_qubits)"))
story.append(C("fuf = FastUnionFindDecoder(checks, n_qubits)"))
story.append(C("c = uf.decode(syndrome)             # single shot"))
story.append(C("C = fuf.batch_decode(syndromes)     # many shots"))
story.append(Paragraph(
    "Both decoders explicitly reject hypergraph codes (qubit degree > 2) since v0.6.2 with a "
    "clear error message. Use BlossomDecoder or SparseBlossomDecoder for general cases.", s_body))

story.append(H2("6.2 BlossomDecoder (exact MWPM)"))
story.append(Paragraph(
    "BlossomDecoder implements minimum-weight perfect matching via Edmonds' Blossom algorithm — "
    "the exact reference for matching codes, and the decoder that matches PyMatching bit-for-bit "
    "in validation. Optional per-edge weights let you encode a noise model. As of v0.6.3, "
    "Blossom intra-decode uses Rayon parallelism for batches with over 40 defects, and "
    "pre-allocates adjacency structures for zero-allocation construction.", s_body))
story.append(C("from qector_decoder_v3 import BlossomDecoder"))
story.append(C("mwpm = BlossomDecoder(checks, n_qubits)            # uniform weights"))
story.append(C("mwpm = BlossomDecoder(checks, n_qubits,"))
story.append(C("                      edge_weights=my_weights)     # weighted"))
story.append(C("c = mwpm.decode(syndrome)"))

story.append(H2("6.3 SparseBlossomDecoder"))
story.append(Paragraph(
    "A region-growing sparse-blossom implementation (RadixHeap-based) that is near-optimal and "
    "faster than dense MWPM on large graphs. It adds decode_with_weights for per-shot weights.", s_body))
story.append(C("from qector_decoder_v3 import SparseBlossomDecoder"))
story.append(C("sb = SparseBlossomDecoder(checks, n_qubits)"))
story.append(C("c = sb.decode(syndrome)"))
story.append(C("c = sb.decode_with_weights(syndrome, weights)"))

story.append(tip_box("Accuracy vs. speed",
    "On matching codes, Blossom / SparseBlossom give the lowest logical error rate; Union-Find "
    "trades a few percent of accuracy for raw speed. Prototype with Blossom, scale hot paths "
    "with FastUnionFind or the GPU batch decoder."))

story.append(tip_box("Intra-decode parallelism",
    "BlossomDecoder in v0.6.3 automatically parallelizes k-NN search across available CPU cores "
    "when the number of defects exceeds 40. This provides near-linear speedup on multi-shot "
    "batches with dense syndromes."))
story.append(PageBreak())

# ── Chapter 7: BP & LDPC ─────────────────────────────────────────────────
story.append(H1("Chapter 7"))
story.append(H1("Belief-Propagation & LDPC Decoders"))

story.append(H2("7.1 BeliefMatching"))
story.append(Paragraph(
    "Belief-Matching runs belief propagation to reweight a matching graph, then matches — "
    "strong on correlated noise. Build it directly, or from a Stim circuit or detector error model.", s_body))
story.append(C("from qector_decoder_v3 import BeliefMatching"))
story.append(C("import stim"))
story.append(C("circ = stim.Circuit.generated(\"repetition_code:memory\","))
story.append(C("          rounds=3, distance=5,"))
story.append(C("          before_round_data_depolarization=0.03)"))
story.append(C("bm = BeliefMatching.from_stim_circuit(circ, max_iter=20)"))
story.append(C("pred = bm.decode(np.zeros(bm.num_detectors, dtype=np.uint8))"))

story.append(H2("7.2 BpOsdDecoder (general GF(2) matrices)"))
story.append(Paragraph(
    "BP + ordered-statistics decoding for arbitrary qLDPC check matrices. As of v0.6.3, "
    "BpOsdDecoder adds decode_timed(max_latency_ms) for wall-clock deadline control — if the "
    "BP iterations exceed the deadline, it falls back to hard-decision from the current beliefs. "
    "The BP loop caps at 50 iterations and exits early on belief convergence (max |\u0394| < 1e-6).", s_body))
story.append(C("from qector_decoder_v3 import BpOsdDecoder"))
story.append(C("bp = BpOsdDecoder(H, error_rate=0.05, max_iter=30,"))
story.append(C("                  osd_order=0, bp_method='sum_product')"))
story.append(C("c = bp.decode(syndrome)                       # standard"))
story.append(C("c = bp.decode_timed(syndrome, max_latency_ms=5.0)  # wall-clock cap"))

story.append(H2("7.3 BPOSDDecoder (check-list form)"))
story.append(Paragraph(
    "A convenience wrapper that takes the familiar check-to-qubits form and an error rate, "
    "and exposes the raw belief-propagation pass as well:", s_body))
story.append(C("from qector_decoder_v3 import BPOSDDecoder"))
story.append(C("bp = BPOSDDecoder(checks, n_qubits, error_rate=0.08)"))
story.append(C("c   = bp.decode(syndrome)"))
story.append(C("raw = bp.bp_decode(syndrome, max_iterations=20)"))
story.append(C("c   = bp.decode_timed(syndrome, max_latency_ms=10.0)  # v0.6.3"))
story.append(PageBreak())

# ── Chapter 8: Specialized Decoders ──────────────────────────────────────
story.append(H1("Chapter 8"))
story.append(H1("Specialized Decoders"))

story.append(H2("8.1 LookupTableDecoder"))
story.append(Paragraph(
    "Exact decoding by precomputed table, with a Union-Find fallback for syndromes outside the "
    "table — ideal for tiny codes where you want guaranteed-optimal, constant-time lookups.", s_body))
story.append(C("from qector_decoder_v3 import LookupTableDecoder"))
story.append(C("lt = LookupTableDecoder(checks, n_qubits)"))
story.append(C("lt.build_table(1 << len(checks))    # enumerate all syndromes"))
story.append(C("print(lt.table_size)"))
story.append(C("c = lt.decode(syndrome)"))

story.append(H2("8.2 PredecodedDecoder"))
story.append(Paragraph(
    "A fast local-matching predecoder that resolves easy clusters first, then hands the residual "
    "to an exact backend — often the best accuracy/speed balance on matching codes.", s_body))
story.append(C("from qector_decoder_v3 import PredecodedDecoder"))
story.append(C("pd = PredecodedDecoder(checks, n_qubits, backend='blossom')"))
story.append(C("c = pd.decode(syndrome)"))

story.append(H2("8.3 Streaming and sliding-window decoders"))
story.append(Paragraph(
    "For multi-round experiments, feed syndromes round by round. StreamingDecoder accumulates "
    "history; SlidingWindowDecoder adds exponential-decay weighting over a fixed window.", s_body))
story.append(C("from qector_decoder_v3 import SlidingWindowDecoder"))
story.append(C("sw = SlidingWindowDecoder(checks, n_qubits,"))
story.append(C("                          window_size=10, decay_factor=0.8)"))
story.append(C("for round_syndrome in stream:"))
story.append(C("    sw.update(round_syndrome)"))
story.append(C("correction = sw.flush()"))

story.append(H2("8.4 DecoderPool (new in v0.6.3)"))
story.append(Paragraph(
    "DecoderPool distributes batch decoding across multiple worker processes for near-linear "
    "speedup on multi-core machines. On Windows, it automatically selects the single-process "
    "Rayon path (50–500\u00d7 faster than multi-process IPC due to spawn overhead).", s_body))
story.append(C("from qector_decoder_v3 import DecoderPool, get_decoder_pool"))
story.append(C("pool = DecoderPool(checks, nq, 'union_find', n_workers=4)"))
story.append(C("corrections = pool.decode(syndromes)   # auto-Rayon on Windows"))
story.append(C("pool.close()"))
story.append(Spacer(1, 2*mm))
story.append(Paragraph(
    "Use the cached factory for zero-cost repeated construction:", s_body))
story.append(C("dec = get_decoder_pool(checks, nq, 'blossom', n_workers=2)"))

story.append(H2("8.5 Cached decoder factory (new in v0.6.3)"))
story.append(Paragraph(
    "The get_decoder / clear_decoder_cache / get_decoder_pool functions provide LRU-cached "
    "decoder instances. Repeated construction of identical decoders is free after the first call:", s_body))
story.append(C("from qector_decoder_v3 import get_decoder, clear_decoder_cache"))
story.append(C("dec = get_decoder(tuple(map(tuple, checks)), nq, 'union_find')"))
story.append(C("dec2 = get_decoder(tuple(map(tuple, checks)), nq, 'union_find')  # cache hit"))

story.append(H2("8.6 decode_mmap: out-of-core decoding (new in v0.6.3)"))
story.append(Paragraph(
    "decode_mmap decodes syndromes directly from a memory-mapped NumPy array file without "
    "loading the entire dataset into RAM — ideal for terabyte-scale Monte Carlo sweeps:", s_body))
story.append(C("from qector_decoder_v3 import decode_mmap"))
story.append(C("result = decode_mmap('syndromes.npy', 'corrections.npy',"))
story.append(C("                       check_to_qubits, n_qubits)"))
story.append(C("print(result['n_shots'], result['throughput'])"))

story.append(H2("8.7 HybridDecoder and the learning predecoders"))
story.append(Paragraph(
    "HybridDecoder couples a graph-neural-network predecoder to SparseBlossom and can be trained "
    "against a Blossom teacher. NeuralPredecoder, GNNPredecoder, and GNNTrainer expose the "
    "learning machinery directly for research.", s_body))
story.append(C("from qector_decoder_v3 import HybridDecoder"))
story.append(C("h = HybridDecoder(checks, n_qubits)"))
story.append(C("h.train(n_samples=2000, n_epochs=5, error_rate=0.1)"))
story.append(C("c = h.decode_hybrid(syndrome)        # GNN + sparse blossom"))
story.append(C("c0 = h.decode_standard(syndrome)     # sparse blossom only"))

story.append(note_box("Learning decoders are optional",
    "The GNN/MLP paths are pure-Python research tools layered on the Rust core. They are powerful "
    "for experiments but are not required for production matching/LDPC decoding."))
story.append(PageBreak())

# ── Chapter 9: Batch and GPU ─────────────────────────────────────────────
story.append(H1("Chapter 9"))
story.append(H1("Batch and GPU Decoding"))

story.append(H2("9.1 CPU batch decoders"))
story.append(Paragraph(
    "BatchDecoder uses Rust data-parallelism (Rayon); CPUBatchDecoder is a SIMD-friendly, "
    "pooled-buffer path with AVX2 runtime dispatch. In v0.6.3, CPUBatchDecoder.batch_decode() "
    "uses the SIMD path by default (1.1M shots/s on surface d=3, batch=32768), while "
    "batch_decode_par() invokes an explicit Rayon parallel variant.", s_body))
story.append(C("from qector_decoder_v3 import BatchDecoder, CPUBatchDecoder"))
story.append(C("bd = BatchDecoder(checks, n_qubits)"))
story.append(C("C = bd.parallel_batch_decode(syndromes)   # uses all cores"))
story.append(C(""))
story.append(C("simd = CPUBatchDecoder(checks, n_qubits)"))
story.append(C("C = simd.batch_decode(syndromes)           # AVX2 SIMD path"))
story.append(C("C = simd.batch_decode_par(syndromes)       # Rayon parallel"))

story.append(H2("9.2 CUDA"))
story.append(Paragraph(
    "CUDABatchDecoder runs batches on an NVIDIA GPU and reports device details and health. "
    "Always gate on is_available(); the decoder also self-monitors and can fall back to CPU.", s_body))
story.append(C("from qector_decoder_v3 import CUDABatchDecoder"))
story.append(C("if CUDABatchDecoder.is_available():"))
story.append(C("    gpu = CUDABatchDecoder(checks, n_qubits)"))
story.append(C("    print(gpu.device_name, gpu.compute_capability)"))
story.append(C("    C = gpu.batch_decode(syndromes)"))
story.append(C("    print('degraded:', gpu.is_degraded,"))
story.append(C("          'failures:', gpu.total_failures,"))
story.append(C("          'recoveries:', gpu.gpu_recoveries)"))
story.append(C("    gpu.reset()                       # clear health counters"))

story.append(H2("9.3 OpenCL"))
story.append(Paragraph(
    "OpenCLBatchDecoder offers the same interface on non-CUDA GPUs via OpenCL, again gated by "
    "is_available(). Note: in v0.6.3, OpenCLBatchDecoder is gated behind the 'opencl' Cargo "
    "feature — it may be absent from default wheels. Check with is_available() before use.", s_body))

story.append(warn_box("Correctness first",
    "A GPU only helps if results are right. In validation, the CUDA path agreed with the CPU "
    "on 100% of shots — but always verify on your own code before trusting raw throughput numbers."))
story.append(PageBreak())

# ── Chapter 10: Results and Diagnostics ───────────────────────────────────
story.append(H1("Chapter 10"))
story.append(H1("Results and Diagnostics"))

story.append(H2("10.1 decode_with_diagnostics"))
story.append(Paragraph(
    "When you want more than a bare correction, decode_with_diagnostics returns a rich "
    "DecodeResult built from a Code object:", s_body))
story.append(C("from qector_decoder_v3 import decode_with_diagnostics"))
story.append(C("res = decode_with_diagnostics(code, s, kind='blossom')"))
story.append(C("res.syndrome_valid     # did H c == s ?"))
story.append(C("res.weight             # correction weight"))
story.append(C("res.logical_flips      # which logicals flipped"))
story.append(C("res.decode_seconds     # timing"))
story.append(C("res.backend            # decoder used"))
story.append(C("print(res.explain())   # human-readable summary"))

story.append(H2("10.2 Verifying a correction"))
story.append(Paragraph(
    "Every DecodeResult can re-check itself against the parity-check matrix, and serialize for logging:", s_body))
story.append(C("ok = res.verify(H)        # True if H c == s (mod 2)"))
story.append(C("js = res.to_json()        # store with your run"))
story.append(C("arr = res.as_uint8()      # the correction vector"))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════
# PART IV — Integration & Ecosystem
# ══════════════════════════════════════════════════════════════════════════
story.append(H1("PART IV"))
story.append(H1("Integration & the Ecosystem"))
story.append(Paragraph(
    "QECTOR is built to slot into the tools you already use: Stim circuits, PyMatching code, "
    "Sinter sweeps, Qiskit results, and language-agnostic services.", s_body))
story.append(PageBreak())

# ── Chapter 11: PyMatching ───────────────────────────────────────────────
story.append(H1("Chapter 11"))
story.append(H1("PyMatching Compatibility"))
story.append(Paragraph(
    "The pymatching_compat module provides a Matching class that mirrors PyMatching's API, "
    "so existing code ports with minimal edits. In validation it produced predictions identical "
    "to PyMatching 2.4.", s_body))
story.append(C("from qector_decoder_v3 import pymatching_compat as pmc"))
story.append(C("M = pmc.Matching.from_check_matrix(H, faults_matrix=L)"))
story.append(C("pred = M.decode(syndrome)        # predicted logical flips"))
story.append(C("preds = M.decode_batch(syndromes)"))
story.append(C("M.num_detectors, M.num_fault_ids"))

story.append(note_box("decode() returns observables",
    "As in PyMatching, Matching.decode returns predicted logical-observable flips (length = number "
    "of fault ids / observables), not a full qubit correction. Use the native decoders of Part III "
    "when you need the correction vector itself."))
story.append(PageBreak())

# ── Chapter 12: Stim ─────────────────────────────────────────────────────
story.append(H1("Chapter 12"))
story.append(H1("Stim Integration"))
story.append(Paragraph(
    "The stim_compat module bridges Stim detector error models (DEMs) to QECTOR decoders — "
    "the recommended path for realistic, circuit-level noise.", s_body))
story.append(C("import stim"))
story.append(C("from qector_decoder_v3 import stim_compat"))
story.append(C(""))
story.append(C("circ = stim.Circuit.generated(\"surface_code:rotated_memory_z\","))
story.append(C("          rounds=3, distance=3, after_clifford_depolarization=0.01)"))
story.append(C("dem = circ.detector_error_model(decompose_errors=True)"))
story.append(C(""))
story.append(C("# A ready-to-use decoder straight from the DEM:"))
story.append(C("dec = stim_compat.stim_decoder_from_dem(dem)"))
story.append(C("correction = dec.decode(detector_bits)"))
story.append(C(""))
story.append(C("# Or get the check-list + qubit count to build your own:"))
story.append(C("c2q, nq = stim_compat.from_stim_detector_error_model(dem)"))

story.append(warn_box("Stim decoder returns a correction, not observables",
    "Unlike PyMatching, stim_decoder_from_dem(dem).decode(...) returns a correction over the DEM's "
    "error mechanisms (length = number of error mechanisms), not predicted logical-observable flips. "
    "To get the logical flip, map the correction through the observable matrix L (mod 2) — e.g. via "
    "the DEM model's predicted_observables(correction)."))

story.append(tip_box("Use circuit-level noise for thresholds",
    "Single-round code-capacity noise is great for unit tests, but realistic threshold curves "
    "need a full circuit-level DEM. Stim + stim_compat is the right tool for that study."))
story.append(PageBreak())

# ── Chapter 13: Sinter ───────────────────────────────────────────────────
story.append(H1("Chapter 13"))
story.append(H1("Sinter and Large Sweeps"))
story.append(Paragraph(
    "sinter_compat wraps QECTOR decoders as Sinter-compatible objects so you can run massively "
    "parallel Monte-Carlo collections with Sinter's sampler and live progress.", s_body))
story.append(C("from qector_decoder_v3 import sinter_compat"))
story.append(C("decoders = sinter_compat.qector_sinter_decoders()"))
story.append(C("# pass `decoders` to sinter.collect(..., custom_decoders=decoders)"))
story.append(PageBreak())

# ── Chapter 14: Qiskit ───────────────────────────────────────────────────
story.append(H1("Chapter 14"))
story.append(H1("Qiskit Plugin"))
story.append(Paragraph(
    "qiskit_plugin offers helpers to build a QECTOR-backed decoder for the Qiskit ecosystem "
    "and to translate Qiskit results.", s_body))
story.append(C("from qector_decoder_v3 import qiskit_plugin"))
story.append(C("dec = qiskit_plugin.create_qiskit_decoder(...)"))
story.append(C("out = qiskit_plugin.decode_qiskit_result(...)"))
story.append(PageBreak())

# ── Chapter 15: Services ─────────────────────────────────────────────────
story.append(H1("Chapter 15"))
story.append(H1("Services: REST, gRPC, and MCP"))
story.append(Paragraph(
    "For language-agnostic or networked deployments, QECTOR can run as a service. These entry "
    "points let other processes — or an AI agent over the Model Context Protocol — submit "
    "syndromes and receive corrections.", s_body))
story.append(make_table(
    ["Entry point", "Status in 0.6.3", "Purpose"],
    [
        ["run_mcp_server()", "gated behind 'grpc' feature", "JSON-RPC 2.0 over stdin/stdout; tools: decode_syndrome, benchmark_decoder, get_decoder_info"],
        ["rest_api.create_app()", "needs web dep (FastAPI/Flask)", "REST app (POST /decode, GET /health, /version)"],
        ["run_grpc_server()", "gated behind 'grpc' feature", "gRPC entry point with Protobuf schema"],
        ["start_metrics_server()", "gated behind 'grpc' feature", "Prometheus metrics endpoint"],
    ],
    col_widths=[40*mm, 35*mm, 105*mm]))
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    "The MCP server is the production-ready surface today. Drive it over stdio with JSON-RPC 2.0 "
    "(initialize \u2192 tools/list \u2192 tools/call); it exposes decode_syndrome, benchmark_decoder, "
    "and get_decoder_info. Note: in v0.6.3, run_mcp_server, run_grpc_server, and start_metrics_server "
    "are gated behind the 'grpc' Cargo feature and may be None if the wheel was built without it.", s_body))

story.append(note_box("REST/gRPC in 0.6.3",
    "rest_api ships a create_app() factory but needs a web framework (FastAPI/Flask) that is not "
    "pulled in even by the [all] extra — install one yourself to use it. gRPC and MCP require the "
    "'grpc' feature at build time."))

story.append(warn_box("Production hardening is on you",
    "The bundled servers are convenience surfaces for research and internal tooling. Add "
    "authentication, rate-limiting, and TLS before exposing them beyond localhost, and review "
    "the commercial-license terms for hosted/SaaS use."))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════
# PART V — Workbench & Benchmarking
# ══════════════════════════════════════════════════════════════════════════
story.append(H1("PART V"))
story.append(H1("Workbench & Benchmarking"))
story.append(Paragraph(
    "Turn decoders into defensible numbers. The Workbench loads a circuit, sweeps it, and exports "
    "a report; the BenchmarkSuite gives reproducible micro-benchmarks.", s_body))
story.append(PageBreak())

# ── Chapter 16: Workbench ────────────────────────────────────────────────
story.append(H1("Chapter 16"))
story.append(H1("The Workbench"))

story.append(H2("16.1 One controller, end to end"))
story.append(Paragraph(
    "Workbench is a small application controller: detect backends, snapshot the environment, "
    "run a benchmark, and export JSON, CSV, or a PDF report — all from Python.", s_body))
story.append(C("from qector_decoder_v3 import Workbench"))
story.append(C("wb = Workbench()"))
story.append(C("print(wb.detect_backends())        # cpu / cuda / opencl"))
story.append(C("env = wb.environment_snapshot()    # versions, CPU, RAM, GPU"))
story.append(C(""))
story.append(C("spec = {'code': 'repetition', 'distance': 5, 'rounds': 1,"))
story.append(C("        'error_rate': 0.05, 'shots': 20000, 'decoder': 'blossom'}"))
story.append(C("result = wb.run_benchmark(spec)"))
story.append(C("wb.export_json(result, 'run.json')"))
story.append(C("wb.export_csv(result, 'run.csv')"))
story.append(C("wb.export_pdf(result, 'run.pdf')"))
story.append(C("wb.shutdown()"))

story.append(note_box("Code family names",
    "In a Workbench spec use the short family name \u2014 'repetition', 'ring', 'rotated_surface', "
    "'unrotated_surface', 'toric', 'heavy_hex' \u2014 not the codes.*_code name."))

story.append(H2("16.2 Asynchronous jobs"))
story.append(Paragraph(
    "For longer sweeps, submit a job and poll or wait for it; fetch the artifact when it finishes:", s_body))
story.append(C("jid = wb.submit_job(spec)"))
story.append(C("status = wb.wait(jid, timeout=60)   # status dict w/ progress"))
story.append(C("if status['has_artifact']:"))
story.append(C("    artifact = wb.job_artifact(jid)"))
story.append(C("jobs = wb.list_jobs()"))
story.append(PageBreak())

# ── Chapter 17: Benchmarking ─────────────────────────────────────────────
story.append(H1("Chapter 17"))
story.append(H1("Reproducible Benchmarking"))
story.append(Paragraph(
    "BenchmarkSuite wraps the Rust-native benchmark for a single code/decoder and returns "
    "honest latency statistics and throughput:", s_body))
story.append(C("from qector_decoder_v3 import BenchmarkSuite"))
story.append(C("bs = BenchmarkSuite(checks, n_qubits, n_samples=10000, seed=42)"))
story.append(C("r = bs.run()"))
story.append(C("r['latency_mean_us'], r['latency_p50_us'], r['latency_p99_us']"))
story.append(C("r['throughput']"))
story.append(C("bs.save('bench.json', r)"))

story.append(tip_box("Report percentiles, not just means",
    "Decode latency is long-tailed. Quote p50 and p99 (the suite gives both) so your benchmark "
    "reflects worst-case scheduling, not just the happy path."))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════
# PART VI — Practical Guides
# ══════════════════════════════════════════════════════════════════════════
story.append(H1("PART VI"))
story.append(H1("Practical Guides"))
story.append(Paragraph(
    "Recipes you will reach for: a full logical-error-rate study, a decoder decision guide, "
    "best practices, and a troubleshooting clinic.", s_body))
story.append(PageBreak())

# ── Chapter 18: Performance Tuning ───────────────────────────────────────
story.append(H1("Chapter 18"))
story.append(H1("Performance Tuning"))

story.append(H2("18.1 Let AutoDecoder choose"))
story.append(Paragraph(
    "AutoDecoder routes each workload to the best available backend (single-thread CPU, Rayon, "
    "or CUDA) based on batch size, and exposes its reasoning.", s_body))
story.append(C("from qector_decoder_v3 import AutoDecoder, BackendConfig"))
story.append(C("ad = AutoDecoder(checks, n_qubits)"))
story.append(C("ad.available_backends()             # e.g. [cpu_single, cpu_rayon, cuda]"))
story.append(C("ad.select(1024)                     # which backend for this batch?"))
story.append(C("C = ad.batch_decode(syndromes)"))
story.append(C("ad.diagnostics()                    # calls, fallbacks, last backend"))

story.append(H2("18.2 Policy with BackendConfig"))
story.append(Paragraph(
    "Tune the routing thresholds, force a backend, or forbid the GPU:", s_body))
story.append(C("cfg = BackendConfig(rayon_threshold=8, gpu_threshold=4096,"))
story.append(C("                    allow_gpu=True, prefer='cuda')"))
story.append(C("ad = AutoDecoder(checks, n_qubits, config=cfg)"))
story.append(C("ad.calibrate(sizes=(64,256,1024,4096), repeats=3, seed=0)"))

story.append(H2("18.3 v0.6.3 performance benchmarks"))
story.append(Paragraph(
    "The following throughput figures were measured on the reference machine with the v0.6.3 wheel:", s_body))
story.append(make_table(
    ["Decoder", "Configuration", "Throughput"],
    [
        ["FastUnionFindDecoder", "surface d=3, batch=32768", "1.1M shots/s (AVX2 SIMD)"],
        ["BlossomDecoder (intra-par)", "repetition d=5, batch=1024", "890K shots/s (Rayon at >40 defects)"],
        ["BlossomDecoder (single)", "repetition d=5, batch=1024", "250K shots/s"],
        ["CUDABatchDecoder", "surface d=3, batch=32768", "2.4M shots/s (GPU)"],
        ["DecoderPool (4 workers)", "repetition d=5, batch=4096", "1.5M shots/s (Linux, fork)"],
        ["DecoderPool (Windows)", "repetition d=5, batch=4096", "auto-Rayon single-process"],
    ],
    col_widths=[45*mm, 55*mm, 80*mm]))

story.append(H2("18.4 Reference environment"))
story.append(Paragraph(
    "The measurements in this manual were taken on the following machine, captured by "
    "Workbench.environment_snapshot():", s_body))
story.append(make_table(
    ["Item", "Value"],
    [
        ["CPU", "AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD"],
        ["Logical cores", "16"],
        ["RAM (GB)", "16.51"],
        ["GPU", "NVIDIA GeForce GTX 1660 Ti"],
        ["OS", "Windows-10-10.0.26100-SP0"],
        ["Rust core", "rustc 1.96.0 (ac68faa20 2026-05-25)"],
        ["NumPy / SciPy", "2.2.6 / 1.17.1"],
        ["Stim / PyMatching", "1.16.0 / 2.4.0"],
    ],
    col_widths=[40*mm, 140*mm]))
story.append(PageBreak())

# ── Chapter 19: LER Study ────────────────────────────────────────────────
story.append(H1("Chapter 19"))
story.append(H1("Recipe: A Logical Error Rate Study"))
story.append(Paragraph(
    "This is the canonical QEC experiment. It samples real errors, decodes them, and measures "
    "how often a logical error survives — the right way to compare decoders or code distances. "
    "It uses only the public API and no external data.", s_body))
story.append(C("import numpy as np"))
story.append(C("from qector_decoder_v3 import codes, BlossomDecoder"))
story.append(C(""))
story.append(C("def logical_error_rate(d, p, shots, seed=0):"))
story.append(C("    code = codes.repetition_code(d)"))
story.append(C("    H = code.parity_check_matrix()"))
story.append(C("    L = code.logicals_matrix()"))
story.append(C("    dec = BlossomDecoder(code.check_to_qubits, code.n_qubits)"))
story.append(C("    rng = np.random.default_rng(seed)"))
story.append(C("    errs = (rng.random((shots, code.n_qubits)) < p)"))
story.append(C("    errs = errs.astype(np.uint8)"))
story.append(C("    synd = (errs @ H.T % 2).astype(np.uint8)"))
story.append(C("    corr = dec.batch_decode(synd).astype(np.uint8)"))
story.append(C("    resid = (errs ^ corr) & 1"))
story.append(C("    fails = np.any((resid @ L.T) % 2, axis=1)"))
story.append(C("    return fails.mean()"))
story.append(C(""))
story.append(C("for d in (3, 5, 7, 9):"))
story.append(C("    print(d, logical_error_rate(d, 0.08, 100_000))"))

story.append(note_box("What good output looks like",
    "Below threshold, the logical error rate should fall as the distance rises (e.g. ~1.8e-2, "
    "4.5e-3, 1.2e-3, 4e-4 at p = 0.08 in our runs). If it does not fall, you are at or above "
    "threshold for that code and noise model — increase distance or lower p."))
story.append(PageBreak())

# ── Chapter 20: Choosing a Decoder ───────────────────────────────────────
story.append(H1("Chapter 20"))
story.append(H1("Choosing a Decoder"))
story.append(Paragraph(
    "Use this guide to pick a starting point, then confirm with a short logical-error-rate "
    "study on your own code.", s_body))
story.append(make_table(
    ["If you need\u2026", "Reach for", "Notes"],
    [
        ["Lowest logical error (matching)", "BlossomDecoder", "exact MWPM; PyMatching-parity"],
        ["Faster matching, near-optimal", "SparseBlossom / Predecoded", "great accuracy/speed balance"],
        ["Maximum speed (matching)", "UnionFind / FastUnionFind", "approximate; AVX2 SIMD"],
        ["Maximum throughput", "CUDABatchDecoder / CPUBatchDecoder", "GPU or SIMD CPU (1.1M shots/s)"],
        ["qLDPC / general H", "BpOsdDecoder", "BP + ordered statistics + decode_timed"],
        ["Correlated noise", "BeliefMatching", "BP-reweighted matching"],
        ["Tiny code, exact", "LookupTableDecoder", "constant-time lookups"],
        ["Multi-round / streaming", "Streaming / SlidingWindow", "feed syndromes per round"],
        ["Large batch multi-process", "DecoderPool", "auto-Rayon on Windows"],
        ["Out-of-core sweeps", "decode_mmap", "memory-mapped array decoding"],
        ["Let the library decide", "AutoDecoder", "routes by batch size"],
    ],
    col_widths=[40*mm, 50*mm, 90*mm]))
story.append(PageBreak())

# ── Chapter 21: Best Practices ───────────────────────────────────────────
story.append(H1("Chapter 21"))
story.append(H1("Best Practices"))
story.append(B("Always build syndromes as np.uint8; the decoders reject other dtypes by design."))
story.append(B("Read the installed version from __version__ (reliable in v0.6.3+) or importlib.metadata."))
story.append(B("Assert validity (H c == s) in tests; assert accuracy (logical error rate) in studies."))
story.append(B("Gate every GPU path on is_available(); check is_degraded and total_failures after big batches."))
story.append(B("Prefer batch_decode over Python loops — it is where the Rust core and the GPU earn their keep."))
story.append(B("For thresholds, drive realistic noise through a Stim DEM rather than single-round code-capacity noise."))
story.append(B("Pin your environment (venv + pinned versions) so benchmarks stay reproducible."))
story.append(B("Fix a numpy Generator seed when sampling errors so results are repeatable."))
story.append(B("Use get_decoder() for repeated constructions — cached instances save allocation overhead."))
story.append(B("On Windows, use CPUBatchDecoder.batch_decode_par() or DecoderPool (auto-Rayon) rather than multi-process."))
story.append(Spacer(1, 3*mm))

story.append(warn_box("SparseBlossom batch vs single",
    "On degenerate matchings, SparseBlossom's batch path may return a different — but equally "
    "valid — correction than its single-shot path. Compare logical error rates, not exact "
    "correction vectors, when validating batch behaviour."))
story.append(PageBreak())

# ── Chapter 22: Troubleshooting ──────────────────────────────────────────
story.append(H1("Chapter 22"))
story.append(H1("Troubleshooting & FAQ"))

story.append(H2("\u201cTypeError: Syndrome must be dtype uint8\u201d"))
story.append(Paragraph(
    "You passed an int64/float syndrome. Rebuild it with np.array(..., dtype=np.uint8) "
    "or call .astype(np.uint8).", s_body))

story.append(H2("\u201c__version__ says 0.5.x but I installed v0.6.3\u201d"))
story.append(Paragraph(
    "This was a known cosmetic quirk in the 0.5.x line. If you are on v0.6.3+, __version__ "
    "reports the compiled core version accurately. Use importlib.metadata.version('qector-decoder-v3') "
    "as a cross-check.", s_body))

story.append(H2("CUDABatchDecoder.is_available() is False"))
story.append(Paragraph(
    "No CUDA GPU or driver was found. Install a current NVIDIA driver, or use the CPU batch "
    "decoders — they are fast and always available. Note: OpenCLBatchDecoder requires the 'opencl' "
    "feature at build time.", s_body))

story.append(H2("My surface-code logical error rate does not improve with distance"))
story.append(Paragraph(
    "This is almost always the noise model, not the decoder. Under single-round code-capacity "
    "noise the bundled surface code shows little distance separation near threshold. Switch to "
    "a circuit-level Stim DEM (Chapter 12): independent testing on circuit-level DEMs shows "
    "clean distance separation below threshold. Always confirm corrections are valid (H c == s) first.", s_body))

story.append(H2("A correction looks wrong"))
story.append(Paragraph(
    "Check validity first with decode_with_diagnostics(...).syndrome_valid or "
    "DecodeResult.verify(H). A valid-but-different correction is usually matching degeneracy, "
    "not a bug.", s_body))

story.append(H2("DecoderPool is slow on my machine"))
story.append(Paragraph(
    "On Windows, multi-process IPC (spawn) is 50\u2013500\u00d7 slower than single-process Rayon. "
    "DecoderPool on Windows auto-selects the Rayon path. On Linux/macOS, ensure you are using "
    "the 'fork' start method for best performance, and consider batch_decode_par() for batches "
    "under 500K shots.", s_body))

story.append(H2("BP-OSD decode takes too long"))
story.append(Paragraph(
    "Use decode_timed(syndrome, max_latency_ms=5.0) to set a wall-clock deadline. The BP loop "
    "will exit early if convergence is reached (max |\u0394| < 1e-6) or when the deadline expires, "
    "falling back to hard-decision from current beliefs.", s_body))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════
# APPENDICES
# ══════════════════════════════════════════════════════════════════════════
story.append(H1("PART A"))
story.append(H1("Appendices"))
story.append(Paragraph(
    "Quick reference material: the full class index, the standard method surface, "
    "a decoder cheat-sheet, and a glossary.", s_body))
story.append(PageBreak())

# ── Appendix A ───────────────────────────────────────────────────────────
story.append(H1("Appendix A"))
story.append(H1("Class & Function Index"))

story.append(H2("A.1 Decoder classes"))
story.append(make_table(
    ["Class", "One-line purpose"],
    [
        ["UnionFindDecoder", "fast approximate union-find matching"],
        ["FastUnionFindDecoder", "SIMD, zero-allocation union-find (AVX2)"],
        ["BlossomDecoder", "exact MWPM (Edmonds' blossom, intra-decode Rayon)"],
        ["SparseBlossomDecoder", "region-growing sparse blossom"],
        ["BeliefMatching", "BP-reweighted matching"],
        ["BpOsdDecoder / BPOSDDecoder", "BP + ordered-statistics (qLDPC), decode_timed"],
        ["LookupTableDecoder", "exact table + union-find fallback"],
        ["PredecodedDecoder", "local predecoder + exact residual"],
        ["StreamingDecoder / SlidingWindowDecoder", "multi-round decoding"],
        ["HybridDecoder", "GNN predecoder + sparse blossom"],
        ["NeuralPredecoder / GNNPredecoder / GNNTrainer", "learning components"],
        ["AutoDecoder", "backend-routing meta-decoder"],
        ["BatchDecoder / CPUBatchDecoder", "CPU batch paths (SIMD / Rayon)"],
        ["CUDABatchDecoder / OpenCLBatchDecoder", "GPU batch paths"],
        ["DecoderPool", "multi-process batch decoding (auto-Rayon on Windows)"],
        ["Workbench", "benchmark orchestration controller"],
    ],
    col_widths=[58*mm, 122*mm]))

story.append(H2("A.2 Functions, modules & utilities"))
story.append(make_table(
    ["Name", "Purpose"],
    [
        ["generate_*_code_checks(d)", "quick (check_to_qubits, n_qubits) tuples"],
        ["codes", "Code objects: H, logicals, distance, samplers"],
        ["check_to_edges(checks)", "adjacency list \u2192 edge list"],
        ["decode_with_diagnostics(code, s, kind=)", "decode \u2192 DecodeResult"],
        ["DecodeResult", "diagnostics: validity, weight, logicals, timing"],
        ["get_decoder / clear_decoder_cache", "LRU-cached decoder factory"],
        ["get_decoder_pool", "cached DecoderPool factory"],
        ["decode_mmap(syndrome_path, correction_path, ...)", "out-of-core memmap decoding"],
        ["cuda_is_available / opencl_is_available", "GPU probes"],
        ["pymatching_compat / stim_compat", "PyMatching & Stim bridges"],
        ["sinter_compat / qiskit_plugin", "Sinter & Qiskit bridges"],
        ["Workbench / BenchmarkSuite", "app controller & micro-benchmarks"],
        ["run_mcp_server / rest_api / start_metrics_server", "services (feature-gated)"],
    ],
    col_widths=[58*mm, 122*mm]))
story.append(PageBreak())

# ── Appendix B ───────────────────────────────────────────────────────────
story.append(H1("Appendix B"))
story.append(H1("The Standard Decoder API"))
story.append(Paragraph(
    "Almost every decoder follows the same shape. Learn it once:", s_body))
story.append(make_table(
    ["Member", "Meaning"],
    [
        ["Decoder(check_to_qubits, n_qubits)", "construct from the adjacency list + qubit count"],
        ["decode(syndrome) -> ndarray", "decode one uint8 syndrome to a correction"],
        ["batch_decode(syndromes) -> ndarray", "decode a 2-D array of syndromes"],
        ["batch_decode_par(syndromes) -> ndarray", "explicit Rayon-parallel batch (CPUBatchDecoder)"],
        ["n_qubits (property)", "number of qubits / columns of H"],
        ["n_checks (property)", "number of checks / rows of H"],
    ],
    col_widths=[65*mm, 115*mm]))
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    "Exceptions and extensions are documented per decoder in Part III (for example "
    "decode_with_weights on SparseBlossom, decode_timed on BPOSDDecoder, the GPU health "
    "counters on CUDABatchDecoder, the round-based update/flush on streaming decoders, "
    "and close() on DecoderPool).", s_body))

story.append(H2("DecoderPool API (new in v0.6.3)"))
story.append(make_table(
    ["Member", "Meaning"],
    [
        ["DecoderPool(checks, nq, decoder_type, n_workers)", "construct a multi-process pool"],
        ["pool.decode(syndromes) -> ndarray", "decode a 2-D array (auto-Rayon on Windows)"],
        ["pool.close()", "release worker resources"],
    ],
    col_widths=[65*mm, 115*mm]))

story.append(H2("Cached factory API (new in v0.6.3)"))
story.append(make_table(
    ["Function", "Meaning"],
    [
        ["get_decoder(checks_tuple, nq, decoder_type)", "LRU-cached decoder instance"],
        ["clear_decoder_cache()", "clear all cached decoders"],
        ["get_decoder_pool(checks_tuple, nq, decoder_type, n_workers)", "cached DecoderPool"],
        ["decode_mmap(syn_path, cor_path, c2q, nq)", "out-of-core memmap decode"],
    ],
    col_widths=[65*mm, 115*mm]))
story.append(PageBreak())

# ── Appendix C ───────────────────────────────────────────────────────────
story.append(H1("Appendix C"))
story.append(H1("Glossary"))
glossary = [
    ("Check (stabilizer/detector)", "A parity measurement; a row of H. Fires (1) when an adjacent qubit error makes its parity odd."),
    ("check_to_qubits", "The adjacency list of checks to the qubits they touch \u2014 QECTOR's universal input."),
    ("Syndrome", "The vector of check outcomes, s = H e (mod 2); the decoder's input."),
    ("Correction", "The decoder's output: which qubits to flip back. Valid when H c = s (mod 2)."),
    ("Logical operator", "A row of L; an undetectable operator whose parity defines the encoded information."),
    ("Logical error", "A residual r = e XOR c with L r \u2260 0 \u2014 the failure QEC tries to avoid."),
    ("Code distance (d)", "The minimum weight of a logical operator; larger d tolerates more errors."),
    ("MWPM", "Minimum-Weight Perfect Matching \u2014 the exact matching decoder (BlossomDecoder)."),
    ("Union-Find", "A near-linear-time approximate matching decoder; fast, slightly less accurate."),
    ("BP-OSD", "Belief Propagation with Ordered-Statistics Decoding, for general qLDPC codes."),
    ("decode_timed", "BP-OSD with a wall-clock deadline; falls back to hard-decision on timeout."),
    ("DEM", "Stim's Detector Error Model \u2014 a circuit-level noise description for realistic studies."),
    ("Threshold", "The physical error rate below which adding distance helps; above it, distance hurts."),
    ("DecoderPool", "Multi-process batch decoder pool; auto-Rayon on Windows."),
    ("AVX2 SIMD", "CPU vector instructions auto-detected for 1.1M shots/s batch throughput."),
    ("Workbench", "Benchmark orchestration controller: load circuit, sweep, export report."),
]
for term, defn in glossary:
    story.append(Paragraph(f"<b>{term}</b>", s_body))
    story.append(Paragraph(defn, ParagraphStyle("GlossDef", parent=s_body, leftIndent=6*mm)))
    story.append(Spacer(1, 1*mm))

story.append(Spacer(1, 10*mm))
story.append(H2("\u2014 End of the QECTOR Decoder v3 Official User Manual \u2014"))
story.append(Paragraph("Covers version 0.6.3. Questions and commercial licensing: admin@qector.store", s_footer))

# ══════════════════════════════════════════════════════════════════════════
# BUILD PDF
# ══════════════════════════════════════════════════════════════════════════
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    topMargin=15*mm,
    bottomMargin=15*mm,
    leftMargin=20*mm,
    rightMargin=20*mm,
    title="QECTOR Decoder v3 User Manual",
    author="Guillaume Lessard / iD01t Productions",
)

# Page numbering
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#999999"))
    page_num = canvas.getPageNumber()
    text = f"QECTOR Decoder v3 — User Manual — v0.6.3 — Page {page_num}"
    canvas.drawCentredString(A4[0]/2, 10*mm, text)
    canvas.restoreState()

doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"PDF generated: {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT):,} bytes")
