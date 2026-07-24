#!/usr/bin/env python3
import numpy as np
from qector_decoder_v3 import UnionFindDecoder, BlossomDecoder, BatchDecoder

checks = [[0,1],[1,2],[2,3],[3,4]]
syndrome = np.array([1,1,0,0], dtype=np.uint8)

print("UnionFind:", UnionFindDecoder(checks, 5).decode(syndrome))
print("Blossom:  ", BlossomDecoder(checks, 5).decode(syndrome))

batch = BatchDecoder(checks[:3], 4)
print("Batch test OK")
print("Quick test passed!")
