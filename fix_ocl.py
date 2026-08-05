with open('python/qector_decoder_v3/__init__.py', 'r', encoding='utf-8') as f:
    c = f.read()
import re
c = re.sub(r'\s*class OpenCLBatchDecoder:.*?class CUDABatchDecoder:', '''

    class OpenCLBatchDecoder:
        """OpenCL batch decoder wrapper - requires source build with --features opencl."""

        __slots__ = ("_inner",)

        def __init__(self, check_to_qubits, n_qubits=None, edge_weights=None):
            self._inner = _OclReal(check_to_qubits, n_qubits, edge_weights)

        @classmethod
        def is_available(cls):
            return _OclReal.is_available()

        def decode(self, syndrome):
            return self._inner.decode(syndrome)

        def batch_decode(self, syndromes):
            return self._inner.batch_decode(syndromes)

        def decode_correction(self, syndrome):
            return self.decode(syndrome)
            
        @property
        def consecutive_failures(self): return 0
        @property
        def total_failures(self): return 0
        @property
        def is_degraded(self): return False
        @property
        def gpu_recoveries(self): return 0
        def reset(self): pass

    class CUDABatchDecoder:''', c, flags=re.DOTALL)
with open('python/qector_decoder_v3/__init__.py', 'w', encoding='utf-8') as f:
    f.write(c)
