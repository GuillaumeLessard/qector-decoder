import stim
import numpy as np
from qector_decoder_v3 import SpaceTimeDecoder

def run_instrumentation():
    d = 5
    r = 5
    p = 0.005
    shots = 2000

    # Build rotated memory Stim circuit
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=r,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )

    # The check structure below comes from `rotated_surface_code(d)`, not from
    # the circuit's DEM, so no detector error model is built here.
    # Simple check structure for d=5 space-time decoder
    # Let's extract spatial checks from code
    from qector_decoder_v3.codes import rotated_surface_code
    code = rotated_surface_code(d)
    c2q = code.check_to_qubits()
    n_qubits = code.n_qubits
    
    # CSS check types: half X (False), half Z (True)
    check_types = [i % 2 == 1 for i in range(len(c2q))]
    p_data = [p] * n_qubits
    p_meas = [p] * len(c2q)

    decoder = SpaceTimeDecoder(
        check_to_qubits=c2q,
        check_types=check_types,
        n_rounds=r,
        p_data=p_data,
        p_meas=p_meas,
        n_qubits=n_qubits,
    )

    sampler = circ.compile_detector_sampler()
    det_samples = sampler.sample(shots=shots)

    fallback_count = 0
    total_decodes = 0

    for sample in det_samples:
        decoder.reset()
        # Reshape detectors into r rounds
        # Each detector in Stim circuit corresponds to spatial checks across rounds
        # Ingest rounds:
        n_checks = len(c2q)
        # Partition detectors by round
        for round_idx in range(r):
            # simulate raw syndrome round
            round_synd = sample[round_idx * n_checks : (round_idx + 1) * n_checks].astype(np.uint8)
            decoder.update(round_synd)
        
        decoder.decode_history()
        if decoder.last_decode_used_fallback:
            fallback_count += 1
        total_decodes += 1

    rate = fallback_count / total_decodes
    print(f"Fallback activations: {fallback_count}/{total_decodes} ({rate:.2%})")

if __name__ == "__main__":
    run_instrumentation()
