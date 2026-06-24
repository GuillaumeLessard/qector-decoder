"""README quick-start is real and its code blocks are syntactically valid.

The README is the first thing a user copies from. This test:

* asserts the README mentions the package and a real decoder name,
* executes a representative example mirroring the quick-start (build a code,
  build a ``BlossomDecoder``, decode a reachable syndrome, check faithfulness),
* compiles every fenced ```python block as a syntax check, skipping any tagged
  ``# no-test``, and asserts at least one block compiles.
"""

import os
import re

import numpy as np

from qector_decoder_v3 import BlossomDecoder, codes


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise AssertionError(f"pyproject.toml not found walking up from {here}")
        cur = parent


def _readme_text():
    path = os.path.join(_repo_root(), "README.md")
    assert os.path.isfile(path), f"README.md missing at {path}"
    return open(path, "r", encoding="utf-8").read()


def test_readme_mentions_package_and_decoder():
    text = _readme_text()
    assert "qector_decoder_v3" in text, "README does not reference the import name"
    # The README documents at least one real decoder by name.
    decoder_names = ["UnionFindDecoder", "BlossomDecoder", "SparseBlossomDecoder",
                     "StreamingDecoder", "BatchDecoder"]
    assert any(name in text for name in decoder_names), (
        "README mentions no known decoder class"
    )


def test_readme_quickstart_example_runs():
    # Mirrors the README quick-start / ecosystem usage: a code, a decoder, a
    # reachable syndrome, and the core faithfulness invariant.
    code = codes.rotated_surface_code(5)
    H = code.parity_check_matrix()
    rng = np.random.default_rng(0)
    error = code.random_error(0.1, rng)
    syndrome = code.syndrome(error)

    decoder = BlossomDecoder(code.check_to_qubits, code.n_qubits)
    correction = np.asarray(decoder.decode(syndrome), np.uint8)
    assert correction.shape == (code.n_qubits,)
    assert np.array_equal((H @ correction) & 1, syndrome), "README example not faithful"


def test_readme_python_blocks_compile():
    text = _readme_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert blocks, "README contains no ```python fenced blocks"

    compiled = 0
    for i, block in enumerate(blocks):
        if "# no-test" in block:
            continue
        # Syntax-only check; we never exec README snippets (they may reference
        # placeholder variables like `syndrome_stream`).
        compile(block, f"<README block {i}>", "exec")
        compiled += 1

    assert compiled >= 1, "no compilable (non-skipped) python blocks in README"
