#!/usr/bin/env python3
"""
Simple benchmark runner for QECTOR Decoder v3
"""
import time
import numpy as np
from qector_decoder_v3 import BlossomDecoder, BatchDecoder

print("Running basic benchmarks...")

# Small batch
t = time.perf_counter()
batch = BatchDecoder([[0,1],[1,2],[2,3]], 4)
syndromes = np.random.randint(0,2,(4096,3), dtype=np.uint8)
_ = batch.parallel_batch_decode(syndromes)
print(f"Batch 4096: { (time.perf_counter()-t)*1000 :.2f} ms")

print("Benchmark done.")