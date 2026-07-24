"""
qector_decoder_v3.pymatching - Drop-in replacement for pymatching.

Usage:
    from qector_decoder_v3.pymatching import Matching
    m = Matching.from_detector_error_model(dem)
    predictions = m.decode_batch(shots)
"""

from .pymatching_compat import Matching

__all__ = ["Matching"]
