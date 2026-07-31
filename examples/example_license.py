#!/usr/bin/env python3
"""Demonstrate license key usage."""

from qector_decoder_v3 import get_license_info, set_license_key

set_license_key("QECT-PRO-test123")
info = get_license_info()
print(f"Tier: {info.get('tier')}")
print(f"Max distance: {info.get('max_distance')}")
