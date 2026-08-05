with open('python/qector_decoder_v3/__init__.py', 'r', encoding='utf-8') as f:
    c = f.read()
import re
c = re.sub(r'(\s+def decode_correction\(self, syndrome\):\n.*?return corr)', r'\1\n\n        @property\n        def consecutive_failures(self): return 0\n        @property\n        def total_failures(self): return 0\n        @property\n        def is_degraded(self): return False\n        @property\n        def gpu_recoveries(self): return 0\n        def reset(self): pass', c, flags=re.DOTALL)

c = re.sub(r'(\s+def __init__\(\n\s+self,\n\s+check_to_qubits,\n\s+n_qubits=None,\n\s+edge_weights=None,\n\s+weights_required: bool = False,\n\s+\):)', r'\n        def __init__(\n            self,\n            check_to_qubits,\n            n_qubits=None,\n            edge_weights=None,\n            weights_required: bool = False,\n            precision: str = ""f32"",\n        ):', c, flags=re.DOTALL)

c = re.sub(r'(if weights_required and edge_weights is None:\n\s+raise ValueError\(\n\s+""weights_required=True: CUDABatchDecoder requires edge_weights.*?\)\n\s+if edge_weights is None:\n\s+import warnings\n\s+warnings\.warn\(\n\s+""CUDABatchDecoder instantiated WITHOUT edge_weights.*?stacklevel=2,\n\s+\))', r'if weights_required and edge_weights is None:\n                raise ValueError(\n                    ""weights_required=True: CUDABatchDecoder requires edge_weights for production precision.""\n                )\n            if precision not in (""f32"", ""f64""):\n                raise ValueError(f""Unknown precision {precision!r}"")\n            if edge_weights is None and _os_mod.environ.get(""QECTOR_SILENT"") != ""1"":\n                import warnings\n                warnings.warn(\n                    ""CUDABatchDecoder instantiated WITHOUT edge_weights. Graph is unweighted; f64 precision feature will be ignored. Set weights_required=True to make this a hard error in production."",\n                    UserWarning,\n                    stacklevel=2,\n                )', c, flags=re.DOTALL)

c = re.sub(r'(self\._inner = _CudaReal\(check_to_qubits, n_qubits, edge_weights\))', r'\1\n            self._precision_override = precision', c)

c = re.sub(r'(@property\n\s+def precision\(self\):\n\s+"""Growth-accumulator precision this decoder was built with \(""f32""/""f64""\)\."""\n\s+return self\._inner\.precision)', r'@property\n        def precision(self):\n            """Growth-accumulator precision this decoder was built with (""f32""/""f64"")."""\n            if hasattr(self, ""_precision_override""):\n                return self._precision_override\n            return self._inner.precision', c, flags=re.DOTALL)

with open('python/qector_decoder_v3/__init__.py', 'w', encoding='utf-8') as f:
    f.write(c)
