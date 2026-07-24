"""
qector.pymatching - Drop-in replacement for pymatching.

Usage:
    from qector.pymatching import Matching
    m = Matching.from_detector_error_model(dem)
    predictions = m.decode_batch(shots)
"""

from qector_decoder_v3.pymatching_compat import Matching

__all__ = ["Matching"]
