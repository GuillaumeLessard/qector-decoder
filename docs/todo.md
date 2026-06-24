**QECTOR Decoder v3 — Complete Upgrade & Validation Plan**

Here is a direct, prioritized list of **all upgrades still needed** to make QECTOR a credible, well-rounded decoder library, plus the **specific tests** required to prove each one. This is based on all reports and evaluations so far.

### 1. Core Algorithmic Upgrades (Highest Priority)

| # | Upgrade Needed | Why It Matters | Tests Required to Prove It |
|---|----------------|----------------|----------------------------|
| 1 | **Make BlossomDecoder truly exact at all distances** (adaptive-k is a partial fix; full Edmonds primal-dual or better candidate selection needed for d≥17) | Current adaptive-k trades latency for correctness. At d=17+ it is still not guaranteed optimal without very large k. | `test_blossom_weight_optimality.py` extended to d=17,19,21 with 10k+ shots each + weight-gap histogram vs PyMatching on identical collapsed graphs |
| 2 | **Belief-matching performance improvement** (currently rebuilds graph per shot) | The 20–34% LER win is real, but per-shot cost makes it impractical for many workloads. | Benchmark `belief_matching` vs plain MWPM with realistic batch sizes (not just single-shot). Measure both LER and wall-clock time per logical error. Add regression test that belief LER advantage holds while showing latency multiplier. |
| 3 | **BP-OSD competitiveness** (currently ~10% behind `ldpc` package) | BP-OSD is one of the few paths that can decode true LDPC/qLDPC codes. Being 10% behind a specialized library limits its value. | Head-to-head on multiple LDPC families ([[72,12]], [[144,12,12]], hypergraph product codes) with correct logical metric. Target: within 3–5% of `ldpc` or document why the gap exists. |
| 4 | **UnionFind robustness or clear deprecation** | Still fails on ~1/3000 adversarial graphs. It is marketed as a "fast path" but cannot be trusted as general-purpose. | Expand `test_property_faithfulness.py` with more adversarial hypergraphs. Either make it pass 100% or clearly document "use only on surface/repetition codes, fall back to Blossom for safety". |

### 2. Performance & Scaling Upgrades

| # | Upgrade Needed | Why It Matters | Tests Required to Prove It |
|---|----------------|----------------|----------------------------|
| 5 | **Reduce latency gap vs PyMatching at d≥9** | Gap is now 14–24× at d=9–15 after the correctness fix. This is the biggest practical weakness. | Full latency + LER tables up to d=21 (or at least d=17) on both rotated surface and toric codes. Show clear "when to use QECTOR vs PyMatching" guidance. |
| 6 | **GPU wins at realistic batch sizes** (currently only wins decisively at batch ≥4096–16384) | Most real workloads use smaller batches. GPU overhead currently hurts more than it helps in many cases. | Realistic workload benchmark (batch sizes 64, 256, 1024, 4096) with end-to-end time including transfer. Show clear crossover points per distance. |
| 7 | **Tail latency characterization** | Current reports focus on p50. Real-time QEC needs p99/p99.9 behavior. | Add p99.9 and max latency to all benchmark tables. Add stress test with 100k consecutive decodes measuring tail. |
| 8 | **Memory scaling characterization up to d=21** | Peak memory and scaling slope are only shown to d=11. | Run memory profiling (both Python tracemalloc + native RSS) up to d=21. Publish power-law fits for each decoder. |

### 3. Ecosystem & Integration Upgrades

| # | Upgrade Needed | Why It Matters | Tests Required to Prove It |
|---|----------------|----------------|----------------------------|
| 9 | **Full multi-platform wheel publishing** | Currently mostly Windows-focused in reports. Users on Linux/macOS cannot easily install. | CI matrix that builds and tests wheels on Linux (manylinux), Windows, and macOS for Python 3.10–3.12. All tests must pass on the built wheels. |
| 10 | **Sinter integration at scale** | Currently only lightly tested. Sinter is the standard way the community benchmarks decoders. | Full `sinter.collect` runs up to d=15 with both `qector_belief` and `qector_blossom`. Compare LER and runtime vs PyMatching in the same harness. |
| 11 | **Stim DEM round-trip fidelity** | DEM collapse is critical but complex. Any future change risks breaking logical correctness. | Property-based test that round-trips (Stim circuit → DEM → QECTOR decode → logical observable check) on 500+ random circuits. |

### 4. Testing & Validation Upgrades (To Close Credibility Gaps)

| # | Upgrade Needed | Why It Matters | Tests Required to Prove It |
|---|----------------|----------------|----------------------------|
| 12 | **Independent machine + OS validation** | All current numbers come from one Windows machine. | Docker-based Linux test run that reproduces all LER tables within statistical noise. Document any differences. |
| 13 | **Higher distance validation (d=17–21)** | Competitive claims stop at d=15. Many interesting regimes are higher. | LER + latency tables at d=17 and d=21 (even if with reduced shots). Show that threshold behavior and parity hold. |
| 14 | **Adversarial + property-based testing expansion** | Current Hypothesis tests are good but limited. | Add tests for: very high defect density, boundary-heavy syndromes, degenerate codes, and hypergraph product codes. Target zero failures on 50k+ adversarial samples. |
| 15 | **Correctness regression suite for every decoder** | Many decoders (especially GPU, belief, BP-OSD) have had correctness issues in the past. | One master test that runs every decoder (including all GPU paths) on a fixed set of 10k seeded syndromes and asserts bit-identical + syndrome-faithful output. |

### 5. Documentation & Positioning Upgrades

| # | Upgrade Needed | Why It Matters | Tests / Proof Required |
|---|----------------|----------------|------------------------|
| 16 | **Clear "When to use which decoder" decision matrix** | Users currently get a long list of decoders with unclear trade-offs. | Add a decision table in README + docs showing: latency vs accuracy vs code family vs batch size. Back it with benchmark data. |
| 17 | **Honest performance positioning vs PyMatching + Fusion Blossom** | Current reports are better but still occasionally optimistic in tone. | Add a permanent "Known Limitations" section with the latest latency ratios at d=5,9,15. Update it on every release. |
| 18 | **Full API stability contract** | Public API (especially `pymatching_compat`, `AutoDecoder`, `belief_matching`) should not break between minor versions. | Add API stability tests + changelog enforcement in CI. |

### Summary Priority Order (Recommended)

**Phase 1 (Credibility)**
- Fix #1 (Blossom exactness at scale)
- Add tests #12, #13, #14 (independent validation + higher distance + adversarial)
- Improve documentation #16 + #17

**Phase 2 (Usability)**
- Improve belief-matching practicality (#2)
- Make GPU useful at realistic batches (#6)
- Strengthen BP-OSD (#3)

**Phase 3 (Production)**
- Multi-platform wheels + full CI (#9)
- Complete memory & tail latency characterization (#7, #8)
- Sinter integration at scale (#10)

Below is the **complete upgrade + proof/test roadmap** to move QECTOR from **strong internal technical asset** to **externally defensible, buyer-grade, research-grade QEC platform**.

The latest report is already strong: it shows exact LER parity with PyMatching through d=15 after the adaptive-k fix, belief-matching lower LER on tested workloads, GPU bit-identical to CPU across many batch sizes, native memory profiling, artifact hashes, and reproduction commands. 

# 0. Immediate credibility fixes

These are small but mandatory.

## Upgrade needed

```text id="cix04v"
Replace “Git commit: unknown” with the real commit hash.
Freeze exact dependency versions in requirements/lock files.
Archive old conflicting QECTOR reports.
Create one canonical release folder.
Add CHANGELOG entry for the d=15 adaptive-k fix.
Add “validated scope” section at the top of README.
```

## Tests / proof needed

```bash id="w3jjqn"
git rev-parse HEAD
pip freeze > artifacts/pip_freeze.txt
cargo metadata > artifacts/cargo_metadata.json
python scripts/generate_report_pdf.py
sha256sum benchmark_results/*
```

## Pass condition

```text id="d2nmoy"
Report, artifacts, git commit, package version, and hashes all point to the same build.
No stale docs contradict the current exact-MWPM / adaptive-k state.
```

---

# 1. External reproduction

This is the single biggest remaining credibility step.

## Upgrade needed

```text id="aj5j8v"
Docker/Linux reproduction path.
Fresh Windows reproduction path.
Fresh Ubuntu reproduction path.
Second physical machine reproduction.
Optional: third-party researcher reproduction.
```

## Tests / proof needed

```bash id="lvptk8"
docker build -t qector .
docker run --rm qector pytest python/tests -q

python scripts/competitive_stim_ler.py --distances 3 5 7 9 11 13 15 --shots 40000
python scripts/belief_extended.py
python scripts/d15_mismatch_audit.py --distance 15 --shots 40000
python scripts/gpu_extensive_test.py
python scripts/run_competitive_benchmark.py --code rotated_surface --distances 3 5 7 9 11 --trials 5000 --warmup 500
```

## Pass condition

```text id="f80sio"
Second machine reproduces qualitative claims:
QECTOR-Blossom LER equals PyMatching on tested MWPM workloads.
QECTOR-belief shows lower observed LER on tested belief workloads.
GPU is bit-identical to CPU when GPU is available.
Artifact hashes match when environment is identical, or tables match statistically when hardware differs.
```

---

# 2. Exact Blossom large-distance proof

The adaptive-k fix restored d=15 parity, but this must now be locked down permanently. The report says the previous d=15 gap was caused by a fixed k=12 candidate cap, and the adaptive-k fix restores parity at d=15. 

## Upgrade needed

```text id="az7w5n"
Permanent large-distance Blossom optimality suite.
Candidate-set diagnostics.
Weight-gap histogram generation.
Defect-count vs excess-weight scatter plot.
Regression test for the old k=12 failure.
```

## Tests / proof needed

```text id="u13q6w"
test_blossom_weight_optimality.py
test_blossom_adaptive_k_regression.py
test_blossom_d15_no_gap.py
test_blossom_candidate_set_contains_optimal.py
test_weight_gap_histogram.py
test_defect_count_vs_weight_gap.py
```

Run:

```bash id="o7f49k"
pytest python/tests/test_blossom_weight_optimality.py -q

python scripts/d15_mismatch_audit.py --distance 13 --shots 40000
python scripts/d15_mismatch_audit.py --distance 15 --shots 40000
python scripts/d15_mismatch_audit.py --distance 17 --shots 20000
python scripts/weight_gap_analysis.py --distances 13 15 17 --shots 3000
```

## Pass condition

```text id="g742xf"
For d=13 and d=15:
QECTOR LER == PyMatching LER within Wilson CI.
No systematic logical mismatch.
Median weight gap = 0.
p99 weight gap = 0 or documented tiny tolerance.
Old fixed-k failure is reproduced by regression fixture and fixed by adaptive-k.
```

---

# 3. Extended PyMatching comparison

The current report shows strong parity, but buyer-grade proof needs more bases, noise points, and distances.

## Upgrade needed

```text id="ddu0eh"
memory_x and memory_z up to d=15.
d=17 if runtime allows.
p-sweep for exact Blossom, not only belief.
rounds sweep: rounds=d, 2d, 3d.
Stim full DEM vs collapsed graph equivalence.
```

## Tests / proof needed

```text id="8jxex2"
test_pymatching_parity_memory_x.py
test_pymatching_parity_memory_z.py
test_pymatching_parity_p_sweep.py
test_pymatching_parity_rounds_sweep.py
test_pymatching_full_dem_vs_collapsed.py
```

Run:

```bash id="vx00u6"
python scripts/competitive_stim_ler.py \
  --bases memory_x memory_z \
  --distances 3 5 7 9 11 13 15 \
  --noise 0.005 \
  --shots 40000

python scripts/competitive_stim_ler.py \
  --distances 5 7 9 \
  --noise 0.002 0.004 0.006 0.008 0.010 \
  --shots 40000

python scripts/competitive_stim_ler.py \
  --rounds-mode d 2d 3d \
  --distances 3 5 7 9 \
  --shots 40000
```

## Pass condition

```text id="2yxewk"
QECTOR-Blossom LER overlaps PyMatching Wilson 95% CI on all claimed exact-MWPM regimes.
Any CI-disjoint failure is automatically flagged and moved into limitations/root-cause section.
```

---

# 4. Belief-matching proof expansion

This is the highest-value differentiator. The report already shows belief-matching beating PyMatching at d=5 and d=7, multi-seed robustness, p-sweep, memory_z validation, and seed×p grid. 

## Upgrade needed

```text id="m5wzpd"
Extend belief tests to d=9 and d=11 if runtime allows.
More seeds.
More p-values.
memory_x and memory_z full grid.
Compare against reference beliefmatching package beyond d=5.
Add latency/cost table for belief mode.
Add BP convergence diagnostics.
```

## Tests / proof needed

```text id="d80r4h"
test_belief_seed_sweep.py
test_belief_p_sweep.py
test_belief_seed_p_grid.py
test_belief_memory_z.py
test_belief_reference_package.py
test_belief_bp_convergence.py
test_belief_latency_cost.py
```

Run:

```bash id="bcw29d"
python scripts/belief_extended.py \
  --distances 3 5 7 9 \
  --probs 0.002 0.004 0.005 0.006 0.008 0.010 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --shots 10000

python scripts/competitive_belief_matching.py \
  --basis memory_x memory_z \
  --distances 3 5 7 9 \
  --noise 0.005 \
  --shots 20000

python scripts/belief_reference_compare.py \
  --distances 3 5 7 \
  --shots 10000
```

## Pass condition

```text id="wd6dth"
Belief-matching has lower pooled LER than PyMatching across the claimed seed×p grid.
d=3 may be parity/noise-dominated and must be labelled as such.
Belief latency is reported honestly as accuracy mode, not fast mode.
```

---

# 5. DEM collapse mathematical proof

DEM collapse is central: the report shows raw mechanisms collapse by about 7.3×–7.7×, including d=15 from 132,426 mechanisms to 17,862 collapsed edges. 

## Upgrade needed

```text id="s86ad3"
Formal probability-combination tests.
Observable-mask preservation tests.
Parallel-edge collapse tests.
Full DEM vs collapsed DEM equivalence tests.
Regression fixture for d=11 and d=15.
```

## Tests / proof needed

```text id="9pss6m"
test_dem_collapse_probability.py
test_dem_collapse_observable_masks.py
test_dem_collapse_parallel_edges.py
test_dem_collapse_full_vs_collapsed_pymatching.py
test_dem_collapse_regression_d11.py
test_dem_collapse_regression_d15.py
```

Cases:

```text id="96eoev"
same detector pair, same observable mask
same detector pair, different observable mask
boundary edge
hyperedge mechanism
zero probability
tiny probability
probability near 0.5
many parallel mechanisms
nested repeat blocks
shift_detectors
```

## Pass condition

```text id="ktrb6g"
PyMatching(full DEM) == PyMatching(collapsed DEM) on LER and observables.
Observable masks are never merged incorrectly.
d=11 fixture remains 50,484 → 6,718.
d=15 fixture remains 132,426 → 17,862.
```

---

# 6. Logical observable proof

Syndrome faithfulness is not enough. The report correctly says the logical metric must avoid `c != e`, because stabilizer-equivalent corrections are harmless. 

## Upgrade needed

```text id="9xos2g"
Predicted observable tests.
Logical coset tests.
Stabilizer-equivalence tests.
Stim observable agreement tests.
BP-OSD residual rowspace tests.
```

## Tests / proof needed

```text id="pbhmv8"
test_predicted_observables.py
test_logical_coset_equivalence.py
test_stabilizer_equivalent_corrections.py
test_stim_observable_agreement.py
test_bposd_rowspace_metric.py
```

## Pass condition

```text id="38gxje"
Correction may differ from true error by stabilizer without counting as logical failure.
predicted_observables(correction) matches Stim sampled observables.
BP-OSD logical failure uses residual not in stabilizer rowspace.
```

---

# 7. BP-OSD / LDPC expansion

The current report shows QECTOR BP-OSD is within about 10% of `ldpc` on `[[72,12]]`, but this is still a secondary claim. 

## Upgrade needed

```text id="wseqvc"
More shots.
More LDPC/qLDPC code families.
BB [[144,12,12]] validation.
HGP code validation.
OSD order sweep.
BP algorithm sweep.
Runtime comparison vs ldpc.
```

## Tests / proof needed

```text id="5514jj"
test_bposd_bb72.py
test_bposd_bb144.py
test_bposd_hypergraph_product.py
test_bposd_bicycle_family.py
test_bposd_osd_orders.py
test_bposd_bp_modes.py
test_bposd_vs_ldpc_runtime.py
```

Run:

```bash id="978jz0"
python scripts/bposd_ldpc_benchmark.py \
  --codes bb72 bb144 hgp bicycle \
  --probs 0.005 0.01 0.02 0.03 0.05 \
  --shots 10000 \
  --osd-orders 0 1 2 w \
  --bp-modes sum_product min_sum normalized_min_sum
```

## Pass condition

```text id="bna7uv"
CSS condition Hx Hz^T = 0 verified.
k logical qubits verified.
Corrections syndrome-faithful.
QECTOR LER within declared tolerance of ldpc.
No claim that QECTOR beats ldpc unless it actually does.
```

---

# 8. GPU proof expansion

The report already shows CUDA/OpenCL bit-identical to CPU across 36 configurations and batch sizes up to 65,536 on GTX 1660 Ti. 

## Upgrade needed

```text id="vp9ky7"
Test more GPUs.
Test CPU-only fallback.
Test no-CUDA install.
Test AutoDecoder crossover.
Test degraded GPU mode.
Add VRAM profiling.
```

## Tests / proof needed

```text id="ab7kyi"
test_cuda_cpu_bit_identical.py
test_opencl_cpu_bit_identical.py
test_gpu_fallback.py
test_auto_decoder_calibrate.py
test_gpu_no_silent_slowdown.py
test_gpu_memory_profile.py
```

Run:

```bash id="26e12o"
python scripts/gpu_extensive_test.py \
  --distances 3 5 7 9 11 13 \
  --batches 1 64 1024 4096 16384 65536

python scripts/gpu_memory_profile.py \
  --distances 5 9 13 \
  --batch 65536

python scripts/auto_backend_calibrate.py \
  --distances 3 5 7 9 11 13 \
  --batches 64 256 1024 4096 16384 65536
```

## Pass condition

```text id="lq8wbi"
GPU output bit-identical to CPU.
GPU is used only when faster.
Fallback produces CPU-identical correction.
No CPU install requires CUDA/OpenCL.
VRAM and RSS stay bounded.
```

---

# 9. Latency, cold path, and tail latency

Page 23 reports p50/p90/p95/p99 and cold path. Keep this as a regression suite. 

## Upgrade needed

```text id="i13nq6"
Benchmark regression thresholds.
p99 tracking.
Cold-path tracking.
Environment-specific baselines.
Performance budget reports.
```

## Tests / proof needed

```text id="e6na4r"
test_latency_percentiles_monotonic.py
test_cold_path_present.py
test_benchmark_syndrome_faithful_gate.py
test_performance_regression_smoke.py
test_batch_vs_single_decode.py
```

Run:

```bash id="1mox48"
python scripts/run_competitive_benchmark.py \
  --code rotated_surface \
  --distances 3 5 7 9 11 \
  --decoders union_find blossom sparse_blossom \
  --trials 5000 \
  --warmup 500 \
  --out benchmark_results/min_competitive
```

## Pass condition

```text id="zldac9"
p50 <= p90 <= p95 <= p99.
Every run syndrome_faithful == true.
Cold path reported separately.
No >2x slowdown from environment-specific baseline unless flagged.
```

---

# 10. Native memory and leak proof

The report now includes native RSS memory profiling and Python tracemalloc. 

## Upgrade needed

```text id="u3gf2d"
Repeated-run leak tests.
Native Rust allocation profile.
GPU-host memory profile.
VRAM profile.
Long-run benchmark memory stability.
```

## Tests / proof needed

```text id="df4iqq"
test_no_python_memory_growth.py
test_no_native_rss_leak.py
test_long_run_decode_memory.py
test_gpu_memory_bounded.py
```

Run:

```bash id="stsbxu"
python scripts/native_memory_profile.py \
  --decoders cpu_batch blossom fast_union_find cuda_batch \
  --distances 5 9 13 \
  --batch 16384

python scripts/leak_test.py \
  --decoder blossom \
  --distance 11 \
  --iterations 100000

python scripts/leak_test.py \
  --decoder cuda_batch \
  --distance 9 \
  --batch 4096 \
  --iterations 1000
```

## Pass condition

```text id="u8b5tl"
RSS does not grow unbounded.
Python peak allocation remains flat in hot path.
GPU memory returns to baseline after decode jobs.
No leak over long repeated decode loops.
```

---

# 11. Streaming / sliding-window proof

Architecture lists Streaming/SlidingWindow, but the report still mainly validates static circuit-level decoding.

## Upgrade needed

```text id="rxkil2"
Streaming decoder correctness.
Flush behavior.
History-size behavior.
Sliding-window equivalence.
Measurement-error time-edge tests.
```

## Tests / proof needed

```text id="tmz1rj"
test_streaming_batch_equivalence.py
test_streaming_flush.py
test_streaming_history_size.py
test_sliding_window_rounds.py
test_measurement_error_time_edges.py
```

## Pass condition

```text id="hmopnl"
Streaming output matches equivalent batch decode where mathematically comparable.
Flush clears state.
No stale syndrome carryover.
Memory stays bounded with number of rounds.
```

---

# 12. SparseBlossom proof and scope

SparseBlossom is near-optimal by design, not exact. Keep it honest.

## Upgrade needed

```text id="pj35u5"
Weight-gap distribution.
Boundary-pairing regression.
Dense-syndrome behavior.
Tail-latency analysis.
Clear docs saying not exact MWPM.
```

## Tests / proof needed

```text id="i9su7f"
test_sparse_blossom_faithfulness.py
test_sparse_blossom_weight_gap.py
test_sparse_blossom_boundary_pairing.py
test_sparse_blossom_dense_syndromes.py
```

## Pass condition

```text id="wdrsfb"
Always syndrome-faithful on supported tested graphs.
Weight gap <= documented bound on small exhaustive tests.
No marketing claim that SparseBlossom is exact.
```

---

# 13. UnionFind proof and scope

UnionFind is fast but approximate, and the report already states adversarial failures exist. 

## Upgrade needed

```text id="47z2zj"
Supported-graph faithfulness suite.
Adversarial graph failure fixtures.
Approximate-decoder warning.
Predecoder/triage positioning.
```

## Tests / proof needed

```text id="rnsfo5"
test_unionfind_surface_faithfulness.py
test_unionfind_repetition_faithfulness.py
test_unionfind_toric_faithfulness.py
test_unionfind_adversarial_failure_documented.py
```

## Pass condition

```text id="92888u"
Faithful on supported QEC matching graphs.
Known adversarial failures are reproduced and documented.
No claim that UnionFind is exact on arbitrary graphs.
```

---

# 14. API stability and input validation

## Upgrade needed

```text id="xm6ro9"
Stable public API.
Type hints.
Docstrings.
Clean exceptions.
Non-contiguous NumPy array support.
Batch shape validation.
```

## Tests / proof needed

```text id="uyf5ol"
test_public_api_imports.py
test_type_hints.py
test_numpy_dtypes.py
test_noncontiguous_arrays.py
test_invalid_inputs.py
test_batch_shapes.py
test_error_messages.py
```

Cases:

```text id="zzbqum"
uint8 arrays
bool arrays
int arrays
Fortran arrays
non-contiguous slices
wrong syndrome length
negative qubit index
empty graph
invalid check matrix
invalid probabilities
```

## Pass condition

```text id="hayhrp"
No panic.
No segfault.
Clear ValueError or TypeError.
Output format consistent across decoders.
```

---

# 15. Result object / diagnostics proof

## Upgrade needed

```text id="g62rgm"
DecodeResult JSON stability.
Sparse correction output.
Bit-packed correction output.
Timing metadata.
Backend metadata.
Fallback status.
```

## Tests / proof needed

```text id="z6z1ji"
test_decode_result_json_roundtrip.py
test_decode_result_sparse_indices.py
test_decode_result_bitpacked.py
test_decode_result_timing.py
test_decode_result_backend_metadata.py
test_decode_result_fallback_status.py
```

## Pass condition

```text id="1tkhzy"
Dense ↔ sparse ↔ bit-packed roundtrip works.
JSON export/import stable.
Timing and backend fields are present.
Fallback status is explicit.
```

---

# 16. Packaging and install proof

## Upgrade needed

```text id="jx01b1"
Clean wheels.
CPU-only install.
Optional CUDA/OpenCL extras.
Python 3.9–3.12 support.
Windows/Linux/macOS support.
No forced GPU dependency.
```

## Tests / proof needed

```text id="av0vhe"
test_clean_venv_install.py
test_wheel_import.py
test_cpu_no_cuda_required.py
test_optional_extras.py
test_version_consistency.py
test_license_included.py
```

Run:

```bash id="odx7zq"
python -m venv clean
clean/Scripts/pip install qector-decoder-v3
python -c "import qector_decoder_v3; print(qector_decoder_v3.__version__)"

pip install "qector-decoder-v3[stim]"
pip install "qector-decoder-v3[bench]"
pip install "qector-decoder-v3[cuda]"
pip install "qector-decoder-v3[opencl]"
```

## Pass condition

```text id="2kwo7q"
CPU install works without CUDA/OpenCL.
Extras install only what they claim.
Version matches pyproject/Cargo/report.
License included.
```

---

# 17. CI/CD proof

## Upgrade needed

```text id="8i3fzy"
GitHub Actions matrix.
Rust tests.
Python tests.
Benchmark smoke artifacts.
Docker build.
Wheel build.
Coverage report.
```

## Tests / proof needed

```yaml id="v7l60s"
windows-latest: py39, py310, py311, py312
ubuntu-latest: py39, py310, py311, py312
macos-latest: py39, py310, py311, py312
```

Commands:

```bash id="ga2fpv"
pytest python/tests -q
maturin build --release
python scripts/run_competitive_benchmark.py --trials 100 --warmup 10
docker build -t qector .
```

## Pass condition

```text id="nptvcw"
All OS/Python combinations pass.
Benchmark smoke artifacts uploaded.
Wheels build.
Docker build passes.
```

---

# 18. Documentation execution proof

## Upgrade needed

```text id="1km4uf"
README examples executable.
Docs command snippets tested.
No stale function names.
No stale benchmark claims.
```

## Tests / proof needed

```text id="7n5xf7"
test_readme_examples.py
test_reproduce_commands.py
test_beyond_pymatching_examples.py
test_scaling_examples.py
test_correctness_audit_commands.py
```

## Pass condition

```text id="yln1j6"
Every code block in docs either runs or is marked no-test.
All CLI commands parse.
No stale API names.
```

---

# 19. Windows Workbench upgrade

If the Windows app is part of the value story, it needs to be real.

## Upgrade needed

```text id="mwhtnq"
QECTOR Workbench app.
Load .stim and .dem files.
Run benchmark jobs.
Display real charts from JSON artifacts.
Export PDF/CSV/JSON.
Show environment snapshot.
Backend detection.
Job queue with cancel/resume.
```

## Tests / proof needed

```text id="hnl7k0"
test_workbench_load_stim.py
test_workbench_load_dem.py
test_workbench_run_benchmark.py
test_workbench_cancel_job.py
test_workbench_export_pdf.py
test_workbench_export_csv_json.py
test_workbench_environment_snapshot.py
test_workbench_backend_detection.py
```

## Pass condition

```text id="g4q09z"
No fake data.
Every chart generated from real artifact.
Every export reproducible.
App survives long benchmark jobs.
Paths with spaces work on Windows.
```

---

# 20. Buyer due-diligence bundle

## Upgrade needed

```text id="kzokmt"
One command that produces all evidence.
All JSON/CSV/PDF artifacts.
All hashes.
All environment data.
All source commit info.
```

## Tests / proof needed

```bash id="24nu4a"
python scripts/run_due_diligence_bundle.py --out qector_evidence_bundle
```

Expected output:

```text id="4xvmxp"
full_report.pdf
correctness_audit.json
competitive_stim_ler.json/csv/md
belief_extended.json/csv/md
belief_grid.json
stim_ler_d13_d15.json
stim_ler_memz.json
weight_gap_analysis.json
gpu_extensive.json
native_memory.json
d15_mismatch_audit.csv
environment.json
pip_freeze.txt
cargo_metadata.json
git_commit.txt
sha256sums.txt
```

## Pass condition

```text id="v9k0bn"
A third party can inspect one folder and reproduce the core claims.
No hidden manual table edits.
All report figures trace to machine-readable artifacts.
```

---

# Priority roadmap

## Priority 1 — must do now

```text id="zqso8w"
1. Add real Git commit hash.
2. Run Docker/Linux reproduction.
3. Run second physical machine reproduction.
4. Lock d=15 adaptive-k fix with permanent regression.
5. Generate due-diligence bundle.
```

## Priority 2 — strengthen claims

```text id="b8r5vk"
6. Extend exact-MWPM parity to memory_x/z, d=17 if possible.
7. Extend belief-matching to d=9 and broader seed×p grid.
8. Add full DEM collapse mathematical tests.
9. Add logical observable/coset test suite.
10. Add native memory and leak tests.
```

## Priority 3 — product value

```text id="vqfmi3"
11. Build QECTOR Workbench.
12. Add PDF/CSV/JSON export from app.
13. Add benchmark job queue.
14. Add backend calibration UI.
15. Add reproducible evidence bundle button.
```

## Priority 4 — commercial readiness

```text id="5x9yt3"
16. Clean repo.
17. Clean license/IP review.
18. Patent review for belief-matching / adaptive-k / DEM collapse if novel.
19. External lab reproduction.
20. Buyer memo + technical appendix.
```

---

# Final state to prove “all”

To say QECTOR is fully proven within its claimed scope, you need this exact evidence:

```text id="tsks7d"
415+ tests passing
20x stability pass
Docker/Linux reproduction pass
Second-machine reproduction pass
QECTOR-Blossom LER parity with PyMatching through d=15
d=15 adaptive-k regression locked
Belief-matching lower pooled LER across seed×p grid
memory_x and memory_z validated
DEM collapse full-vs-collapsed equivalence
logical observable correctness
GPU bit-identical to CPU
cold/hot/p99 latency tables
native + Python memory profiles
artifact hashes
real git commit hash
clean install/wheel tests
due-diligence bundle
```

That is the proof chain.

Once those are complete, the strongest safe public claim becomes:

```text id="6fvxio"
QECTOR Decoder v3 is a reproducible Rust/Python QEC decoder platform with Stim/Sinter/PyMatching-compatible workflows, exact-MWPM LER parity against PyMatching on tested circuit-level workloads through d=15, a belief-matching accuracy mode that lowers observed LER on tested correlated workloads, BP-OSD/qLDPC support, bit-identical CPU/GPU batch decoding, and artifact-level reproducibility.
```
