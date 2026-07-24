"""Pytest bootstrap for the qector_decoder_v3 test suite.

Loads the repo-root `.env` (Stripe keys, license private key) *before* any
test module imports `qector_decoder_v3.stripe_integration`, which reads the
environment at import time. This makes the suite CWD-independent - it works
whether pytest is invoked from the repo root, `python/`, or anywhere else.

The `.env` file is gitignored and never committed.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(_REPO_ROOT / ".env")

# The webhook tests exercise the *unauthenticated fallback* path by default
# (sig_header="" with no explicit webhook_secret). With the production
# STRIPE_WEBHOOK_SECRET present in the environment, the handler correctly
# demands a signature - so for the test session we drop it. Tests that need
# signature verification pass their own explicit webhook_secret.
os.environ.pop("STRIPE_WEBHOOK_SECRET", None)
