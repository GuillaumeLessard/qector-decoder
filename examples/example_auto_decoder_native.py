#!/usr/bin/env python3
"""Demonstrate NativeAutoDecoder."""

import numpy as np
from qector_decoder_v3 import NativeAutoDecoder, codes

code = codes.rotated_surface_code(3)
dec = NativeAutoDecoder(code.check_to_qubits, code.n_qubits, distance=3, noise_rate=0.08, batch_size=1, is_qldpc=False)

syndrome = np.zeros(len(code.check_to_qubits), dtype=np.uint8)
correction = dec.decode(syndrome)

print(f"Correction shape: {correction.shape}")
print(f"Correction: {correction}")
assert correction.shape == (code.n_qubits,)
