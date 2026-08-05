"""
QECTOR Decoder v3 - Qiskit Integration Tutorial Example.

Demonstrates decoding quantum error correction results from Qiskit.
"""

import numpy as np
from qector_decoder_v3.qiskit_plugin import (
    create_qiskit_decoder,
    decode_qiskit_syndrome,
)


def main():
    print("=== QECTOR Decoder v3 Qiskit Integration Tutorial ===")

    # 1. Direct syndrome decoding from a parity-check matrix
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    syndrome = np.array([1, 0], dtype=np.uint8)
    correction = decode_qiskit_syndrome(H, syndrome, decoder="blossom")
    print(f"Parity Check H shape: {H.shape}")
    print(f"Syndrome: {syndrome}")
    print(f"Decoded Correction: {correction}")
    assert np.array_equal((H @ correction) % 2, syndrome)
    print("Syndrome verification (H @ c == s): PASS\n")

    # 2. Raw dict / Qiskit result decoding
    result_dict = {"counts": {"0x0": 450, "0x1": 50}}
    decoder_fn = create_qiskit_decoder(code_distance=3)
    res = decoder_fn(result_dict)
    print(f"Qiskit Result metadata: {res['metadata']}")
    print(f"Corrections array shape: {res['correction'].shape}")
    print("Qiskit tutorial completed successfully!")


if __name__ == "__main__":
    main()
