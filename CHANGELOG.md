# Changelog

All notable changes to QECTOR Decoder v3 are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning. Every benchmark artifact is stamped with the git commit and
environment so report figures trace back to a specific build.

## [0.7.0] — UNRELEASED

The repo version is 0.7.0; the latest version on **PyPI is 0.6.9**. Nothing below
has been published. `src/*.rs` is `.gitignore`d, so `git log v0.6.9..HEAD` shows
none of the Rust work recorded here — it is verified by `cargo test` and by
reading the tree.

### Benchmarks — corrected artifacts, and what they say
- **`official_benchmark_results.{json,csv,md,pdf}` were regenerated.** The first
  v0.7.0 versions reproduced the defect that got six pre-v0.7.0 artifacts
  withdrawn (todo6 A1-03), and additionally stamped `_provenance`'s canned
  methodology note — which asserts `ler.estimate_ler_circuit_level` scoring and
  `ler.assert_comparable` validation — into files produced by neither. They
  reported "LER" as a syndrome-consistency check `(H·ĉ = s)`, which reads 0.000%
  for every decoder at every distance because it never asks whether the logical
  observable flipped; they timed QECTOR through a native batch call against
  PyMatching through a 20-shot Python loop extrapolated to 100,000; and their
  LER chart plotted hardcoded analytic curves rather than measurements.
- **What replaced them.** Every row now comes from
  `ler.estimate_ler_circuit_level` — one circuit, one decomposed DEM, one
  detector/observable sample set per cell, one `decode_batch` resolver for
  QECTOR, PyMatching and ldpc alike — and `ler.assert_comparable` gates the rows
  before writing. Nothing is extrapolated: 63 cells that exceeded the per-cell
  decode budget are recorded as *not measured*, with their probe rate and
  projected cost.
- **The numbers, per-cell and not generalised.** `qector_blossom` and PyMatching
  2 returned identical logical-failure counts on identical samples at `d = 3`
  and `d = 5` (1891 and 1596 in 100,000). On throughput PyMatching led at every
  distance measured, consistent with this project's long-standing note that it
  leads on plain MWPM. Both statements are in the artifacts and in the README.
- `scripts/run_custom_comparison_benchmark.py` gained `--from-json`, which
  re-renders the reports from a stamped artifact without re-measuring, so a
  presentation fix cannot silently move the numbers underneath a citation.

### Added — CLI, diagnostics, and ecosystem entry points
- **`qector` CLI** (`decode` / `bench` / `serve`) and **`qector-doctor`**, a
  15-check environment diagnostic that reports *why* a backend is unavailable
  instead of letting it fail at decode time. Both registered in
  `pyproject.toml:[project.scripts]`. `qector-doctor` must be invoked as a console
  script or `python -m`; run as a bare script from inside its own package
  directory it puts the raw `.pyd` on `sys.path`, shadowing the `__init__.py`
  wrapper that supplies constructor defaults, and then misreports `native-core`
  and `decode` as FAIL.
- **Sinter decoder entry points** — `qector_blossom`, `qector_belief`,
  `qector_unionfind`, `qector_bposd`, `qector_unionfind_unweighted` — so
  `sinter.collect()` resolves them without a `custom_decoders=` argument. The
  **qiskit-qec plugin** is registered the same way.
- **`qector_decoder_v3.pymatching`** exists as a real submodule, so
  `from qector_decoder_v3.pymatching import Matching` works. The attribute
  spelling already did; the module did not.

### Added — decoder families
- **`AmbiguityClusterDecoder`** — BP, partition on `|LLR|`, DFS-cluster the
  ambiguous mechanisms, enumerate each cluster exactly.
- **`TwoStageDecoder`** — decode the X sector, propagate, decode the Z sector,
  with any of blossom / unionfind / bposd / sparse_blossom per stage. It requires
  `check_types`: a DEM does not record which sector a detector belongs to.
- **`ColourCodeDecoder`** — BP-OSD over the **undecomposed** hypergraph DEM.
  Matching is not a correct colour-code decoder: a colour-code mechanism can light
  three detectors at once and has no graphlike decomposition, and Stim's
  `detector_error_model(decompose_errors=True)` raises outright on
  `color_code:memory_xyz` at `d ≥ 5`. Any colour-code "matching" decoder is either
  decoding a different code than it claims or silently dropping mechanisms.
- **Relay-BP** (`BpMethod::Relay`) — layered serial schedule over the exact
  sum-product update, so each check sees the freshest messages. Selected with
  `bp_method="relay"` on `BPOSDDecoder` and `BpOsdDecoder`.

### Added
- **`QECTOR_LICENSE_FILE`, plus `~/.qector/license.key`.** The core read only
  `QECTOR_LICENSE_KEY`, so pointing a deployment at a key *file* left it silently
  on Community tier. Resolution order is now env var, then file path, then the
  conventional `~/.qector/license.key`; a UTF-8 BOM and trailing newline are
  stripped so a key written by PowerShell redirection works unmodified. A
  `QECTOR_LICENSE_FILE` that is set but unreadable reports an **invalid** key
  rather than degrading to Community — the silent downgrade was the hardest
  version of this failure to diagnose. `set_license_key_file()` is the Python
  equivalent, and the pure-Python fallback resolves keys identically so the two
  layers cannot disagree.
- **`SpaceTimeDecoder` and `CUDABpOsdDecoder` are reachable from Python.** Both
  were registered in `lib.rs` and shipped in every wheel, but `__init__.py` never
  bound them. `SpaceTimeDecoder` was additionally declared in the type stub, so
  type checkers accepted calls that raised `AttributeError` at runtime.
- **`generate_parity_check_matrix()` and `flush_usage()`.** Both had a native
  implementation *and* a pure-Python fallback; neither was ever bound to a
  module-level name, leaving the native function unreachable and the fallback
  dead code.
- **`DemModel.make_decoder` covers every shipped decoder family.** It handled 5
  of 9; `lookup_table`, `hybrid_cascade`, `ambiguity_cluster` and `two_stage`
  could not be built from a detector error model at all — the entry point real
  circuit-level workloads use. `DemModel.DECODER_KINDS` enumerates them.
- **`batch_decode` on `TwoStageDecoder` and `AmbiguityClusterDecoder`,** reusing
  one scratch buffer across the batch.
- **Weighted cluster growth on the GPU.** `CUDABatchDecoder` and
  `OpenCLBatchDecoder` now take an optional `edge_weights` argument (the DEM's
  `log((1-p)/p)` matching weights) and run the same adaptive time-step growth as
  `uf_core::grow_weighted`. Both kernels produce identical logical error rates,
  which is the cross-check that the port is faithful. Omitting the weights keeps
  the original integer growth, bit-identical to the CPU `UnionFindDecoder`.
  `UfGraph::edge_lengths()` exposes the normalised lengths both backends upload.
- **`scripts/full_decoder_benchmark.py`** — benchmarks every decoder family
  against syndromes sampled from a Stim circuit, reporting LER with Wilson
  intervals, latency, and syndrome faithfulness, with a per-decoder time budget
  that trims (and honestly records) the shot count instead of stalling a sweep.
  **`scripts/generate_benchmark_pdf.py`** renders its JSON to a PDF; it performs
  no measurements of its own.
- **Kernel-compile gate for CUDA** (`test_cuda_kernels_compile`) plus a CUDA
  BP-OSD decode test.

### Fixed
- **The CUDA BP-OSD kernel did not compile, so `CUDABpOsdDecoder` could never be
  constructed.** A refactor replaced the `local_pos` linear scan with O(1)
  inverse indices in two of three call sites; the third still called the deleted
  helper. Every CUDA test treated construction failure as "no GPU here" and
  skipped, so a hard compile error looked like a missing device on a machine with
  a working GPU. Availability probes now distinguish a kernel source defect from
  an environment gap and fail rather than skip.
- **`HybridCascadeDecoder` ignored its edge weights.** Weights reached the
  Blossom escalation decoder but not the Union-Find pre-filter — and the
  pre-filter *accepts* most circuit-level syndromes, so its unweighted answer is
  what callers received. On `surface_code:rotated_memory_x` at circuit-level
  p=0.005 this cost roughly **2.3× the logical error rate** (d=9: 0.0452 → 0.0171)
  at no speed benefit. *Provenance:* development run, artifact not retained. The
  fix is corroborated by `benchmark_results/full_decoder_benchmark.json`
  (2026-07-30, QECTOR 0.7.0, p=0.005, rounds=d, seed 20260730), where
  `qector:hybrid_cascade` scores LER 0.0126 at d=11 against 0.0438 for the
  unweighted CUDA Union-Find — but that run does not contain a pre-fix row, so it
  confirms the resulting accuracy, not the 2.3× ratio itself.
- **`AutoRouter` ignored `priority`, and `explain()` no longer described
  `decode()`.** The native-auto path ran ahead of the Python policy layer
  unconditionally, so `AutoRouter(priority="accuracy")` behaved identically to
  the default and the "never route a hyperedge code to a matching decoder"
  invariant — enforced in that policy layer — was bypassed for qLDPC codes. The
  native path is now taken only for balanced-priority graphlike problems, and
  `explain()` consults the same predicate.
- **`TwoStageDecoder.decode` and `AmbiguityClusterDecoder.decode` returned
  `bytes`** while the other seven families return a `uint8` ndarray, so neither
  was substitutable for the decoder it was meant to replace.
- **`set_license_key()` silently accepted invalid keys.** It swallowed the
  verifier's rejection and set `QECTOR_LICENSE_KEY` anyway, so an expired,
  revoked, or malformed key surfaced later as an unexplained tier cap. It now
  raises `ValueError` and leaves the environment untouched.
- **`.env` was never read**, because `python-dotenv` was not declared as a
  dependency and the import failure was swallowed — so Stripe fulfillment
  reported "STRIPE_SECRET_KEY is not configured in environment or .env file" on
  machines where it plainly was. The dependency is declared, and the error names
  the real cause when the file goes unread.
- **The GPU decoders were weight-blind, and it cost accuracy, not speed.** With
  no way to receive the DEM's matching weights, both GPU backends scored the
  *unweighted* logical error rate — the same figure as every other unweighted
  path. **Measured cost of being weight-blind**, on
  `surface_code:rotated_memory_x`, circuit-level p=0.005, rounds=d, d=11:
  unweighted CUDA Union-Find LER **0.0438** and unweighted OpenCL **0.0434**,
  against **0.0149** for the weighted CPU `UnionFindDecoder` and **0.0062** for
  PyMatching — roughly 3×. Speed was never the problem: unweighted OpenCL decodes
  that same d=11 at **9.6 µs/shot** against 138 µs/shot for the weighted CPU core
  and 82 µs/shot for PyMatching.
  *Provenance:* `benchmark_results/full_decoder_benchmark.json`, generated
  2026-07-30 by `scripts/full_decoder_benchmark.py` on QECTOR 0.7.0 /
  Windows 11 / AMD Zen 2 / Python 3.11.9 / numpy 2.2.6, seed 20260730, ≤20,000
  shots per cell. The harness trims shots per decoder to a time budget and records
  the trimmed count, so rows differ in precision — compare `shots` and
  `ler_ci95_*`, not `ler` alone. The file is under the `.gitignore`d
  `benchmark_results/`, so it is reproducible but not distributed.
  *Withdrawn from an earlier draft of this entry:* the figures "0.04525 → 0.00900,
  weighted CPU 0.01400, PyMatching 26 µs/shot". The weighted-GPU LER of 0.00900
  has no surviving artifact and no row in the run above, and that run measures
  PyMatching at 82 µs/shot, not 26. The qualitative conclusion — weights matter far
  more than throughput here — is unchanged and is what the retained run supports.
  *Known remaining gap:* the weighted kernel scans all `E` edges per growth
  round, where the CPU walks only the frontier of odd clusters, so the weighted
  GPU path is accurate but ~10x slower than the weighted CPU path. Porting the
  frontier to the kernel is the outstanding work.
- **Prometheus latency quantiles froze permanently.** The cached sorted view was
  invalidated only when the FIFO reservoir evicted, i.e. not until 8192 samples
  had been recorded — so any process that read `get_latency_quantiles()` before
  then kept serving that first snapshot forever, and p50/p90/p99 never moved
  again. Every push now invalidates the cache. The accompanying
  `eviction_invalidates_sorted_cache` test also asserted something arithmetically
  impossible (that one outlier in 8192 samples moves p99); it is replaced by two
  tests that check the real property, including below the eviction threshold.
- **`build_maturin.cmd` built the wrong checkout.** It `cd`'d to a hard-coded
  absolute path on one machine; it now uses its own directory and reports build
  failures instead of always writing `DONE`.
- **`generate_report_from_json.py` could misalign every column.** It emitted
  cells by key-presence per row while taking headers from a caller-supplied list,
  so a row whose shape differed from the caller's assumption shifted all values
  one column left. Headers and cells now derive from one spec.
- OpenCL `batch_decode` reallocated four device buffers per call, including
  Union-Find scratch proportional to `batch_size × graph size` — roughly 900 MB
  of allocate-and-free per call at batch 20k. Buffers are now reused across
  calls, mirroring `CudaWorkspace`.

### Changed
- Ruff and Clippy are clean across the repo and all feature combinations
  (`--no-default-features`, `cuda`, `opencl`, default).
- `test_license_included.py` asserts the licence's *terms* rather than one
  revision's wording, so the deliberate move to PolyForm Noncommercial 1.0.0 no
  longer reads as a failure.
- `test_bposd_batch_decode` verifies BP-OSD's batch path agrees with per-shot
  decoding; it previously asserted the method did not exist.
- The `benchmarks_session` harnesses are documented as **not** decoder
  performance measurements: they feed uniform-random bit patterns to a
  boundaryless ring code, where ~49% of inputs have odd defect parity and admit
  no correction, so every backend measured ~2 ms/shot of non-terminating cluster
  growth. This is why GPU paths previously appeared slower than CPU.

### Added — earlier in the 0.7.0 cycle (2026-07-28)
- **Real Ed25519 license verification** in Rust core (v2 + QECT-PRO/ENT signed payloads, production key matches Python PEM), offline CRL at ~/.qector/revoked.txt, expiry enforcement, QECTOR_ENFORCE=1 hard gate.
- **MCP decoder cache** keyed by code + decoder_type + tuning env with honest cache_hit reporting, working clear_decoder_cache, per-tier decode timeout budgets, and async startup pre-warm.
- **Rust NativeAutoDecoder** with distance/noise/batch-aware backend selection, CPUBatch routing for batch_size > 1024, and license tier enforcement.
- **Stripe metered billing**: real HTTP flush via ureq with 1s/2s/4s backoff retry.
- **AutoDecoder AUTO_NATIVE backend** and license tier check in the Python fallback chain.
- **Workbench CLI** with --license-key, license_info command, and record_shots wiring.
- **benchmarking.decoder_type parameter** and support for all 11 decoder backends.
- **9 real MCP tools** (no phantom): decode_syndrome, batch_decode, decode_hyperedge, benchmark_decoder, get_decoder_info, get_backend_health, clear_decoder_cache, get_server_env, recommend_decoder.
- **MCP security**: stdin reader enforces 10 MB max_content_length (-32600 on oversize), syndrome length validation (-32602), strict decoder_type enum validation (-32602).
- **REST API security**: bind 127.0.0.1 by default, X-Request-ID logging, per-IP 120 req/min rate limiter.
- **Metrics server**: default bind 127.0.0.1:9090.
- **Arena allocator** + thread-local scratch in FastUnionFindDecoder (eliminates 15 per-shot Vec allocations).
- **AVX2 bitmask clear** with runtime detection and O(words) popcount.
- **DecoderArena::with_capacity** and benchmark test.
- **CI workflows**: release-build.yml (smoke test before publish) and tests.yml (full Rust + Python matrix).

### Changed — earlier in the 0.7.0 cycle (2026-07-28)
- UnionFindDecoder.decode/decode_into return Result instead of panicking on syndrome length mismatch.
- SparseBlossomDecoder: 3 production unwraps replaced with graceful Option/break.
- MWPM get_pr() returns Option, callers degrade gracefully.
- BenchmarkResult includes Wilson 95% CI, n_unfaithful, unfaithful_rate fields.
- AutoRouter ships with use_native_auto=True and _try_native_auto() routing.

### Fixed — earlier in the 0.7.0 cycle (2026-07-28)
- MCP syndrome length mismatch returns -32602 instead of SIGABRT (panic=abort).
- Hyperedge per-decoder gate: Blossom/SparseBlossom/BPOSD accept hyperedge codes; UnionFind/FastUnionFind reject with -32602.
- Invalid decoder_type returns -32602 in <50ms.

### Fixed — crash safety in the Rust core
Under `panic = "abort"` every one of these aborted the host process rather than
raising a catchable Python exception:
- `grpc_server.rs:316,325,354,369` — `decoder_cache.lock().unwrap()`: a single
  panicking cache client poisoned the mutex and took down every later caller.
  Now propagated with `map_err`.
- `cuda_bp_osd.rs:249,299` — the same pattern on `workspace.lock()`;
  `cuda_batch.rs` had always used `map_err` here, so the two files disagreed.
- `cuda_batch.rs` — every CUDA async error return (`cu_memcpy_htod_async`,
  `cu_launch_kernel`, `cu_memcpy_dtoh_async`, `cu_stream_synchronize`,
  `cu_stream_destroy`) was discarded with `let _ =`, so a failed launch returned
  corrupt corrections as `Ok`. The `let _ = cu_*` count is now zero.
- `cuda_workspace.rs:134` — `pointers()` aborted the host; it now returns
  `Result<_, String>` via `ok_or_else`. This is a signature change, and
  `cargo clippy --features full` exiting 0 is what confirms every caller was
  updated.
- `ler_benchmark.rs:266` — `Bernoulli::new(phys_error).unwrap()` panicked on NaN
  or out-of-range `phys_error`; now `.ok()`, with the fall-through documented.
- `cascade_decoder.rs:159` — `BPOSDDecoder::new(…).expect("invalid BP-OSD input")`
  inside a constructor that returns `Self` and therefore could not fail gracefully.

*Verification note:* the CUDA files do not compile under `--no-default-features`.
Any claim about them is gated on `--features full`, or it is unverified.

### Changed — benchmark methodology
- **The four comparison tables in `README.md` are withdrawn.** They put
  code-capacity QECTOR and circuit-level PyMatching in one table, and cited
  artifacts under `benchmark_results/` — a `.gitignore`d path that has never been
  in a published commit or wheel, so the numbers were never checkable. The README
  carries the full notice; the numbers are retracted, not deleted.
- `ler.assert_comparable` tags each run with its noise model and refuses
  cross-model comparisons, so this class of error cannot recur silently.
- `scripts/regenerate_benchmark_artifacts.py` rewritten. It previously had no
  argument parser, so *any* invocation — `--help` included — immediately launched
  a 1.6M-decode run. It now requires an explicit `--yes`, supports `--dry-run`,
  and embeds a provenance block (methodology, git commit, tree-dirty flag,
  parameters, dependency versions, caveats) in the artifact so a future reader
  cannot mistake its methodology for something else. **The full publication run
  has not been performed**, so no regenerated artifact exists yet.

### Validation
Measured on the working tree 2026-07-31:

| Gate | Result |
|---|---|
| `cargo test --no-default-features` | 303 passed, 0 failed |
| `cargo test --features full` | 323 passed, 0 failed, 7 ignored (hardware-only) |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | exit 0 |
| `cargo clippy --features full --all-targets -- -D warnings` | exit 0 |
| Full Python suite | **no trustworthy baseline** — the last full run was interrupted and none has completed on a quiesced tree since |

## [0.6.9] - 2026-07-26

### Added
- **v2 licence tokens carrying `tier` and `exp`.** Legacy tokens sign only
  `receipt_id:email`, so a 60-day evaluation was cryptographically identical to
  a perpetual licence. Format `v2.{claims_b64}.{sig}` over
  `{rid, email, tier, exp}`; the signature covers the encoded claims segment
  verbatim and is verified *before* the claims are parsed. `license_claims()`
  returns verified, unexpired claims so callers can gate on tier.
  Legacy 2- and 3-part tokens continue to verify unchanged, and a v2 token fails
  **closed** on installs predating v2 support.
- **Tuning environment variables documented** in the README: `QECTOR_BLOSSOM_K_MULT`,
  `QECTOR_BLOSSOM_INTRA_PAR`, `QECTOR_BLOSSOM_INTRA_THREADS`,
  `QECTOR_CUDA_DEVICE_ID`, `QECTOR_OPENCL_DEVICE_ALLOW` — with which of them can
  change results rather than only throughput.
- **Task A — Exact log-domain BP**: `BpMethod{MinSum,Exact}` enum in `bp_osd.rs`, default `Exact`. `phi(x) = -ln(tanh(x/2))` via hybrid exact (<0.25) + 65536-entry interpolated LUT ([0.25, 20]) + 0.0 (≥20). Deterministic reliability ranking (`rel_key` 1e-6 quantization + index tie-break) fixes float-noise OSD-0 basis flipping. PyO3 `bp_method` kwarg (`"exact"`/`"min_sum"`).
- **Task B — Higher-order OSD**: true combination-sweep OSD-1/2 in `bp_osd.rs` — flip subsets of W≤12 least-reliable selected columns, residual GF(2) re-solve with syndrome pre-subtraction, min-weight faithful candidate wins. PyO3 `osd_order` kwarg (0 default preserved, non-breaking).
- **Task C — GNN-enhanced belief matching**: `GNNBeliefMatcher` class in `belief_matching.py` — end-to-end GNN-guided MWPM pipeline (`DetectorGraph` → `GNNPredecoder.predict_with_node_probs` → per-edge weights → max-per-qubit fan-in → `SparseBlossomDecoder.decode_with_weights` → faithfulness fallback). Optional synthetic training (`train_samples`/`error_rate`/`train_epochs`/`seed`). `decode_with_gnn` one-shot helper. Both re-exported at package top level.
- **`BPOSDDecoder` Python wrapper** now forwards `bp_method`/`osd_order` kwargs (additive, default-preserving).
- **`backend.py`**: batch-decode 1D-output reshape fix (handles decoders returning flat arrays for 2D input).
- **`stripe_integration.py`**: `create_license_token` import fallback when `generate_license_keys` is absent.

### Fixed
- **`BeliefMatching.from_numpy_h` produced a decoder that returned an empty
  array for every syndrome.** The constructor built `hyper_obs` with zero rows,
  and `decode` returns `hyper_obs @ hard` — so any `BeliefMatching` built from a
  raw parity-check matrix silently decoded nothing and reported no error.
  `edge_obs` carried the identical defect, breaking the matching fallback for
  exactly the syndromes BP found hardest. Both matrices now carry qubit
  incidence, so `decode` returns a length-`n_qubits` correction; verified
  faithful (`H @ corr == syndrome`) on all 16 repetition-code d=5 syndromes.
- **`verify_license_token` raised instead of rejecting malformed input.** The
  `except RuntimeError` clauses caught neither `binascii.Error` nor
  `UnicodeDecodeError` (both `ValueError` subclasses), so a garbage token
  propagated an exception to the caller rather than returning `False`. Any
  caller passing attacker-influenced input crashed. Now catches
  `(InvalidSignature, ValueError, TypeError)`; nine adversarial inputs are
  locked by tests.
- **Checkout no longer pins `payment_method_types=["card"]`.** Pinning it
  disables Stripe dynamic payment methods, so Link, Apple/Google Pay and local
  methods never render and every buyer must type a card number. Omitting the
  parameter lets Stripe show the highest-converting eligible methods per
  customer, configured from the Dashboard.
- **blossom.rs boundary bug**: boundary node matched without `boundary_spt`, panicking on odd-defect boundary-less codes.

### Performance / internals (dev.md items, this cycle)
- f32 GNN stack (Task 2.1); seeded GNN init (6.5); hybrid hot-path allocations (6.7); word-packed GF(2) solver `src/gf2.rs` shared by BP-OSD and Blossom (1.1); hyperedge `BestEffortVerified` policy (7.2); mwpm dense fallback threshold (4.1); RadixHeap SmallVec buckets (4.4); sliding-window bit-packed history (5.2); latency quantiles (5.4); NoHashHasher lookup table (2.3); safetensors strict shape guard (2.4).

## [0.6.8] - 2026-07-22

### Fixed
- **v0.6.7 was completely unimportable**: `__init__.py` unconditionally accessed `_native_module.HybridCascadeDecoder`, which is absent from the compiled module. All 18 native-module lookups now use `_guard("ClassName")` — missing symbols return a callable stub that raises `RuntimeError` at decode-time. Import always succeeds.
- **CI YAML broken**: smoke-test `run:` step had unindented Python code inside a literal block scalar, breaking GitHub's parser. No workflow triggers worked (`workflow_dispatch`, tag push, pull_request). Fixed by indenting the inline Python to match the content level.

### Added
- **CI smoke test before publish**: installs wheel, imports `qector_decoder_v3`, instantiates `sparse_blossom` decoder. Catches unimportable wheels before they reach PyPI.

## [0.6.7] - 2026-07-20

### Fixed
- **`indices[self.n_checks..]` panic in `bp_osd.rs` (`BPOSDDecoder.decode`)**: crashed the entire Python process (unrecoverable Rust panic across the PyO3 boundary, not a catchable exception) on hyperedge check structures, e.g. the 18-check/9-qubit rotated surface code. Reproduced deterministically in an isolated process, patched, rebuilt, reinstalled, and re-verified via the exact original repro (exit 0, correct output). Confirmed specific to hyperedge structures — the same call path runs cleanly on a graphlike repetition code both before and after the patch.
- **NaN `error_rate` panic in `bp_osd.rs`**: an unclamped NaN error rate silently poisoned belief propagation, burning through all 50 iterations before crashing (~133s wall time). Fixed by clamping `error_rate` at the constructor. Re-verified via the exact original repro: now returns in ~17s — matching the baseline runtime of every other clean script this session — instead of crashing at 133s.
- **`_opencl_health_check()`'s child-process probe script referenced an undefined `_np` name.** The probe script imports `numpy` as `np` in the child process, but the probe line referenced `_np` (the parent module's private alias, not defined in that child scope), raising a silent `NameError` on every invocation. This unconditionally set `_OPENCL_HEALTH_CACHE = False`, meaning `opencl_is_available()` always reported `False` regardless of real hardware or driver support. Fixed by using `np` consistently in the probe script. Verified directly against the shipped `__init__.py`: the embedded child-process script now reads `import numpy as np` and uses `np.asarray(...)` / `np.array(...)` throughout the probe, with no `_np` reference anywhere in that scope. (A same-machine black-box call to `opencl_is_available()` alone can't confirm this fix, since this machine has no OpenCL hardware and would correctly return `False` either way — the source-level check above is what closes the question.)
- **`SparseBlossomDecoder::grow_regions` / `RadixHeap` — bit-identical to `BlossomDecoder`.** Re-confirmed empirically: 10/10 trials with genuinely syndrome-reachable errors (constructed via `H @ error`, not hand-picked) on the 18-check hyperedge code produced bit-for-bit identical corrections between the two decoders. Note: this repo's own tests named "bitperfect" (`test_sparse_vs_blossom_bitperfect_ring`, `..._ring_20`) do not actually assert cross-decoder equality — they only check each decoder's own output independently satisfies the syndrome. The bit-identical claim rests on this session's empirical test, not on those two Rust tests.
- **`BPOSDDecoder`'s `bp_decode_timed` (Rust) / `decode_timed` (Python) — deadline honored before the first iteration.** Confirmed at three independent levels: source shows `Instant::now()` initialized before the loop with the deadline check as the literal first statement of each iteration; both relevant Rust unit tests pass (`test_bp_decode_timed_converges_and_matches_untimed`, `test_bp_decode_timed_respects_zero_latency_deadline`); and live timing on the 18-check hyperedge code shows `max_latency_ms=0.0` stabilizing at ~0.02ms vs. a generous budget's consistent ~0.68ms — a clean ~30x gap.

### Retracted
- **LER benchmark's rotated-surface generator does *not* emit a graphlike code.** A prior draft of this entry claimed `generate_surface_code_checks` was fixed to emit "a proper two-half (X + Z) graphlike code." Re-tested directly against the live 0.6.7 install: `generate_surface_code_checks(3)` still returns an 18-check, 9-qubit matrix where every qubit participates in 8 checks (graphlike requires ≤2). Unchanged from prior versions — still a rank-4-of-18 hyperedge code, exactly as its own docstring describes ("periodic/toric surface code," "hyperedges," "rank-deficient"). No code change was made here this cycle; the claim was inaccurate and is retracted, not fixed.

### Added
*(Not independently re-tested this session — carried over as-is from a prior draft of this entry.)*
- `SparseBlossomDecoder.k_nearest_via_radix` — public event-driven candidate-edge discovery backed by a new `RadixHeap<u32, HeapEvent>` structure exposed to downstream callers that need fine control over the candidate set.
- MCP server (`mcp_server`) now exposes 5 new tools: `decode_syndrome_blossom`, `batch_decode_blossom`, `run_ler_benchmark`, plus expanded `get_decoder_info` listing all 11 decoder families.

### Quality
- Cross-decoder syndrome-validity test suite (`src/cross_decoder_tests.rs`) covers UF / FastUF / LookupTable / SparseBlossom / BP-OSD / SlidingWindow / Streaming / Hybrid.
- SafeTensors loader now has a full round-trip test suite covering generic + runtime dispatch, dtype mismatch, missing tensors, and shape round-trip.
- Dead-code warnings eliminated across the crate (8 → 0).
- `cargo test --lib`: 142 passed, 0 failed, 0 ignored — re-run directly this session (finished in 14.91s), unchanged after this session's two `bp_osd.rs` edits. `cargo check --all-targets` not re-run this session.

## [0.6.6] - 2026-07-12

### Fixed
- **Critical: package import broken on every published v0.6.5 wheel.** `python/qector_decoder_v3/__init__.py` had a leftover, unguarded `_RustOpenCLBatchDecoder = _native_module.OpenCLBatchDecoder` line (a duplicate of the properly try/except-guarded assignment later in the file). Since the public CI release wheels are built with `--no-default-features --features cuda` (no `opencl` feature), the compiled module never has this attribute, so `import qector_decoder_v3` raised `AttributeError` immediately on a completely clean install. This was masked in local development because default-feature builds include `opencl`. Root cause found and reproduced by testing a fresh `pip install qector-decoder-v3==0.6.5` in an isolated venv and by rebuilding locally with the exact CI feature flags. Fixed by removing the dead duplicate line; verified the corrected package imports cleanly under the exact CI build configuration in a clean venv, with `cuda_is_available()` / `opencl_is_available()` correctly returning `False` rather than crashing.
- **v0.6.5 is not usable and should not be installed** — use v0.6.6 or later.

## [0.6.5] - 2026-07-10

### Fixed
- **mypy clean**: Resolved all 8 type errors across `decode_mmap.py`, `decoder_pool.py`, and `belief_matching.py` — strict type checking now passes on the full Python layer.
- **Test imports**: `test_comprehensive_suite.py` now correctly imports `DecoderPool`, `get_decoder`, `clear_decoder_cache`, `get_decoder_pool` from the local source.
- **CI resilience**: Ensured v0.6.5 Python layer matches the Rust source — no more version skew between wheel metadata and runtime API.
- **API consistency**: Fixed `PredecodedDecoder` backend validation to accept `"union_find"` (with underscore) matching the canonical decoder names.
- **Test suite NameError**: `test_comprehensive_suite.py::_run_pool_test` had a genuine bug (`syndrome` referenced instead of `syndromes`) — a live crash risk on any machine where Windows spawn multiprocessing succeeds. Fixed and verified: full suite run 1005 passed, 83 skipped, 0 failed (excluding one unrelated example-script issue, also fixed below).
- **ruff clean**: Full repo now passes `ruff format --check` and `ruff check` with zero errors; `.venv`, `.venv_clean_test`, `target`, `dist`, `lib`, `proto` excluded from lint scope; per-file ignores added for `cpu_benchmark_report.py` and `test_exports.py`.
- **`examples/example_batch.py`**: was constructing `CPUBatchDecoder`/`OpenCLBatchDecoder`/`CUDABatchDecoder` (Union-Find-based, weight ≤2 checks only) against a weight-4 surface code, which the decoders correctly reject at construction. Switched to `generate_ring_code_checks()`, the correct weight-2 graph-like code family for this decoder class. Verified: `python/tests/test_examples.py` passes (1 passed in 154.39s), and the script runs end-to-end.
- **CI secret injection**: Regenerated and verified the `RUST_SRC_B64_1/2/3` GitHub Actions secrets (byte-identical round-trip checked before upload). Confirmed the full 15-wheel build succeeds end-to-end. *Correction (2026-07-31):* this entry originally described the matrix as "Linux/Windows/macOS x86_64/aarch64 x Python 3.9-3.13". No aarch64 wheel has ever been published. The 15 wheels are CPython 3.9–3.13 x `win_amd64` / `manylinux_2_17_x86_64` / `macosx_11_0_arm64` — checked against the live PyPI JSON API.

### Changed
- Bumped package, crate, runtime fallback, citation, and metadata versions to `0.6.5` across `pyproject.toml`, `Cargo.toml`, `python/qector_decoder_v3/__init__.py`, `CITATION.cff`, `codemeta.json`, `README.md`, `PYPI_README.md`, docs, and examples.

## [0.6.4] - 2026-07-10

### Fixed
- **CI secrets updated**: Rust source injected at build time now matches the v0.6.4 Python layer. The v0.6.3 wheel was built with stale Rust source (missing `LERBenchmark` and other v0.6.3 Rust changes) — it has been superseded by v0.6.4.

## [0.6.3] - 2026-07-10

### Added
- **BP-OSD convergence cap**: 50-iteration max, early-exit on belief convergence (max |Δ| < 1e-6), `decode_timed(max_latency_ms)` for tail-latency control.
- **AVX2 SIMD transpose + gather**: CPU batch decoder auto-detects AVX2 via `is_x86_feature_detected!` — 1.1M shots/s on surface d=3, batch=32768.
- **Blossom intra-decode Rayon parallelism**: k-NN search parallelized via `into_par_iter()` when n_defects > 40.
- **DecoderPool**: Multi-process batch decoding with auto-Rayon fallback on Windows. *(The "50–500× faster than multi-process IPC" figure this entry originally carried is withdrawn: no artifact, workload, or batch size was ever recorded for it. The Rayon fallback is still the right default on Windows — Windows spawn-based IPC pays a process-creation and pickling cost per batch that Rayon does not — but the magnitude is unmeasured.)*
- **Cached decoder factory**: `get_decoder()` / `clear_decoder_cache()` / `get_decoder_pool()` — zero construction cost after first call.
- **`decode_mmap`**: Out-of-core decoding via memory-mapped NumPy arrays.
- **`DecodeResult` / `decode_with_diagnostics`**: Structured decode results with per-shot diagnostic metadata.
- **`Workbench`**: High-level orchestration for multi-decoder comparison and benchmarking.
- **Comprehensive test suite**: `test_comprehensive_suite.py` — 200+ scenario tests across all decoder families.

### Changed
- `FastUnionFindDecoder` docstring updated: "Consistently faster than UnionFindDecoder on surface and repetition codes (1.1M shots/s)".
- `run_mcp_server` gated behind `grpc` feature; `OpenCLBatchDecoder`/`opencl_is_available` gated behind `opencl`.
- CPUBatch `batch_decode()` now calls SIMD path by default; `batch_decode_par()` for explicit Rayon variant.
- Bumped package to 0.6.3 across all metadata files.

### Fixed
- `bposd.py` line 118: CRW consistency bug in belief tracking.
- DecoderPool on Windows: auto-selects single-process Rayon path instead of broken multi-process IPC.
- Memory layout optimizations: aligned Vecs, pre-reserved capacity in Blossom construction.

## [0.6.2] - 2026-07-06

### Added
- v0.6.2 release notes: `CHANGELOG_v0.6.2.md`.

### Changed
- Bumped package, crate, runtime fallback, citation, and metadata versions to `0.6.2` across `pyproject.toml`, `Cargo.toml`, `python/qector_decoder_v3/__init__.py`, `CITATION.cff`, `codemeta.json`, `README.md`, `PYPI_README.md`, docs, and examples.

### Fixed
- Hardened Union-Find decoder input validation and error handling in `python/qector_decoder_v3/__init__.py`.
- Expanded regression coverage for hypergraph rejection and relaxed latency validation.

## [0.6.1] - 2026-07-05

### Fixed
- **README.md**: the "Belief-matching accuracy mode" example called
  `BeliefMatching(check_to_qubits, n_qubits, error_rate=0.005)`, which does
  not match the real constructor (`BeliefMatching(matrices, max_iter=30,
  bp_shortcut=False)`) and raises `TypeError: unexpected keyword argument
  'error_rate'` if run verbatim. Replaced with a self-contained example using
  `BeliefMatching.from_stim_circuit(circuit)`, verified by executing it
  end-to-end against the published `0.6.0` wheel.
- Audited every class instantiation in every `*.md` file in the repo against
  the real `__init__` signatures (not just import-name existence, which the
  `0.6.0` audit covered) — this was the only mismatch found. `BpOsdDecoder`
  and the Sinter integration example were checked and confirmed correct.

## [0.6.0] - 2026-07-05

### Fixed
- **README.md / PYPI_README.md**: the Stim detector-error-model workflow
  example referenced `qector_decoder_v3.stim_compat.stim_circuit_to_check_matrix`,
  a function that does not exist (it was superseded by
  `from_stim_detector_error_model` during the 0.5.9 cleanup, without the
  docs being updated). Both quick-start examples now import
  `from_stim_detector_error_model` and build the `check_to_qubits` mapping
  from a real `stim.DetectorErrorModel` (`circuit.detector_error_model(...)`),
  matching the documented function's actual signature.
- **Python 3.9 compatibility**: replaced PEP 604 `X | None` union syntax with
  `typing.Optional`/`typing.Union` in `backend.py`, `qiskit_plugin.py`,
  `stim_compat.py`, and `__init__.py`. This syntax requires Python 3.10+ and
  would raise `TypeError` at import time on 3.9, contradicting the package's
  own `requires-python = ">=3.9"` and the `smoke-import-py3.9` CI job.
- Hardened `test_clean_venv_install.py`'s qiskit-absent smoke test to also
  stub out `qiskit`, not just `stim`/`pymatching`.
- Version-string consistency: bumped `pyproject.toml`, `Cargo.toml`,
  `Cargo.lock`, `python/qector_decoder_v3/__init__.py`, `CITATION.cff`, and
  `codemeta.json` to `0.6.0`, and updated all plain-text version labels in
  `INSTALL.md`, `README.md`, `PYPI_README.md`, `docs/GPU_AND_CUPY.md`,
  `docs/SERVICE_API_SCHEMA.md`, and the `examples/` scripts.

## [0.5.9] - 2026-07-02

### Added
- **CuPy-accelerated GPU backend** (`gpu_backend.py`, `bp_cupy.py`): batched
  belief-propagation / BP-OSD decoding on NVIDIA GPUs via CuPy, with automatic
  NumPy fallback on machines without a GPU. See `docs/GPU_AND_CUPY.md` and
  `examples/example_cupy_bp.py`.
- **Decoder auto-routing** (`routing.py`): automatic backend selection (CPU /
  native CUDA / CuPy) based on batch size and hardware availability. See
  `examples/example_auto_routing.py`.
- **Streaming / sliding-window sessions** (`streaming.py`): incremental,
  multi-round decoding sessions with window + commit semantics for long-running
  syndrome streams. See `examples/example_streaming_session.py`.
- Corresponding test suites: `test_gpu_backend.py`, `test_bp_cupy.py`,
  `test_routing.py`, `test_streaming.py`.

### Removed
- Superseded `advanced.py` module and its dedicated tests
  (`test_advanced_decoders.py`, `test_beliefmatching_bridge.py`,
  `test_kimi_findings.py`, `test_stim_circuit_to_check_matrix.py`), folded into
  the new routing/streaming/GPU-backend surface.
- Superseded due-diligence bundle helper scripts (`finalize_bundle.py`,
  `run_due_diligence_wrapper.py`), superseded by `run_due_diligence_bundle.py`.

### Fixed
- `ruff format --check python/` was failing in CI (`tests / ruff-and-mypy`) on
  9 files; reformatted with `ruff format` (lint and mypy were already passing).
- Version bumped to `0.5.9` across `pyproject.toml`, `Cargo.toml`, `Cargo.lock`,
  and the Python runtime fallback version, since PyPI `0.5.7` was already
  published under the prior module layout and cannot be overwritten.

## [0.5.7] - 2026-06-30

### Fixed
- Aligned Python packaging, Cargo metadata, runtime fallback version, and PyPI release bundle at `0.5.7`.
- Verified the Windows CPython 3.11 wheel imports the compiled extension and reports `qector_decoder_v3.__version__ == "0.5.7"`.

## [0.5.0] - 2026-06-23

### Fixed
- **Blossom exactness at large distance (adaptive-k).** `BlossomDecoder` previously
  used a fixed `k=12` candidate cap, which undershot the optimum on large dense
  circuit-level graphs (d ≥ 13–15), producing heavier matchings and a markedly
  worse logical error rate than PyMatching at d=15. The candidate set is now
  **adaptive**, `k = max(12, 4·√n_defects)`, restoring exact-MWPM LER parity with
  PyMatching through **d=15** (`memory_x` and `memory_z`). Locked permanently by
  `test_blossom_adaptive_k_regression.py`, `test_blossom_d15_no_gap.py`,
  `test_blossom_candidate_set_contains_optimal.py`, `test_weight_gap_histogram.py`,
  and `test_defect_count_vs_weight_gap.py`.

### Added
- **QECTOR Workbench** (`qector_decoder_v3.workbench.Workbench`): headless,
  fully-tested controller to load `.stim`/`.dem` files, run cancelable benchmark
  jobs through a FIFO queue, and export JSON/CSV/PDF reports (charts built from
  real artifacts, no fabricated data). Backend detection + environment snapshot.
- **Evidence & reproduction scripts**: `run_due_diligence_bundle.py` (one-command
  evidence bundle with hashes + git commit), `belief_reference_compare.py`,
  `gpu_memory_profile.py`, `auto_backend_calibrate.py`, `leak_test.py`.
- **Provenance**: `benchmarking.capture_environment()` now records `git_commit`, so
  every JSON artifact and report figure points to the exact build it came from
  (replaces "Git commit: unknown").
- **Expanded validation suite** covering: exact-MWPM parity (memory_x/z, p-sweep,
  rounds-sweep), DEM-collapse mathematical equivalence + d=11/d=15 regression
  fixtures (50,484→6,718 and 132,426→17,862), logical-observable / stabilizer-coset
  correctness, belief-matching seed×p grid + reference cross-check, BP-OSD on
  BB[[72,12]]/BB[[144,12,12]]/HGP/bicycle, GPU CPU-bit-identity + fallback +
  calibration, latency percentiles + tail, and memory/leak profiling.
- **Documentation**: README "Validated scope", "When to use which decoder" decision
  matrix, and a permanent "Known limitations" section with honest latency ratios.

### Build
- Refreshed Rust dependencies (`rayon` 1.12, `fastrand` 2.4) and migrated the
  optional `grpc`/`full` stack to `tonic` 0.14 / `prost` 0.14 with a vendored
  `protoc` (`protoc-bin-vendored`), so gRPC builds need no system `protoc`. The
  default wheel features (`opencl`, `cuda` with CPU fallback) are unchanged.

## [0.4.0]

### Added
- `SparseBlossomDecoder` (region-growing, RadixHeap, exact DP for n ≤ 20 with
  Edmonds primal-dual fallback), bit-validated against `BlossomDecoder`.
- Ecosystem layer: `codes`, `dem`, `result`, `backend`, `pymatching_compat`,
  `benchmarking`; belief-matching and BP-OSD decoders; Stim/Sinter compatibility.
- Native CUDA (NVRTC + Driver API) and OpenCL batch decoders with CPU fallback.

### Fixed
- Stim DEM loading uses the correct detector graph (mechanisms = columns,
  detectors = rows), replacing the earlier `stim_compat` heuristic.

## [0.2.0]

- Python + Numba baseline decoder core (pre-Rust rewrite).
