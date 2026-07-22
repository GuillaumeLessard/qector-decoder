"""Integration smoke test for the v0.6.7 cascade / blossom / bposd changes."""
import numpy as np
import qector_decoder_v3 as qd

print("version:", qd.__version__)

# --- 1. HybridCascadeDecoder: batch_decode + batch_decode_timed -------------
def ring_code(n):
    return [[i, (i + 1) % n] for i in range(n)]

checks = ring_code(15)
cascade = qd.HybridCascadeDecoder(checks, 15)
syndromes = np.zeros((6, 15), dtype=np.uint8)
syndromes[0, 3] = 1; syndromes[0, 4] = 1      # adjacent pair -> UF pre-filter
syndromes[2, 0] = 1; syndromes[2, 3] = 1      # 3-hop pair -> UF pre-filter
syndromes[4, 2] = 1; syndromes[4, 4] = 1
syndromes[4, 9] = 1; syndromes[4, 11] = 1     # two pairs
corr = cascade.batch_decode(syndromes)
assert corr.shape == (6, 15), corr.shape
H = np.zeros((15, 15), dtype=np.uint8)
for ci, qs in enumerate(checks):
    H[ci, qs] = 1
for i in range(6):
    assert np.array_equal((H @ corr[i]) & 1, syndromes[i]), f"row {i} not faithful"
print("cascade.batch_decode: OK (all rows syndrome-faithful)")
assert hasattr(cascade, "batch_decode_timed")
corr_t = cascade.batch_decode_timed(syndromes, 50.0)
assert corr_t.shape == (6, 15)
print("cascade.batch_decode_timed: OK")

# --- 2. BlossomDecoder on a large (d >= 9-sized) code -----------------------
big = ring_code(100)  # 100 checks -> large_code path (d >= 9 trigger)
blossom = qd.BlossomDecoder(big, 100)
syn = np.zeros(100, dtype=np.uint8)
for c in (3, 11, 27, 42, 60, 78):
    syn[c] = 1
corr_b = blossom.decode(syn)
Hb = np.zeros((100, 100), dtype=np.uint8)
for ci, qs in enumerate(big):
    Hb[ci, qs] = 1
assert np.array_equal((Hb @ corr_b) & 1, syn), "blossom large-code not faithful"
print("blossom.decode on 100-check code (d>=9 parallel path): OK, faithful")

# --- 3. BPOSDDecoder decode_timed -------------------------------------------
bposd = qd.BPOSDDecoder(checks, 15, 0.08)
syn2 = np.zeros(15, dtype=np.uint8)
syn2[5] = 1; syn2[6] = 1
c2 = bposd.decode_timed(syn2, 50.0)
assert c2.shape == (15,)
print("bposd.decode_timed: OK")

# --- 4. Benchmark helper routing --------------------------------------------
assert hasattr(bposd, "decode_timed"), "benchmark scripts rely on decode_timed"
print("ALL SMOKE TESTS PASSED")
