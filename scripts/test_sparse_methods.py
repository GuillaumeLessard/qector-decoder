import sys

sys.path.append("python")  # after site-packages: an installed build wins over the source tree
from qector_decoder_v3 import _RustSparseBlossomDecoder

print(dir(_RustSparseBlossomDecoder))
