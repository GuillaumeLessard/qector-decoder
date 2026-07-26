"""
qector - Top-level PyMatching-compatible package alias for QECTOR Decoder v3.

Enables one-line drop-in imports:
    from qector.pymatching import Matching
    import qector.pymatching as pymatching
"""

from qector_decoder_v3 import pymatching_compat as pymatching
from qector_decoder_v3.pymatching_compat import Matching

__all__ = [
    "Matching",
    "pymatching",
]
