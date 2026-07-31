# QECTOR Licensing

## Tier Overview

| Tier | Max Distance | GPU | Pricing |
|---|---|---|---|
| Community | d≤7 | No | Free (source-available) |
| Pro | d≤19 | Yes | See https://qector.store/pricing |
| Enterprise | d≤63 | Yes | See https://qector.store/pricing |

## Key Formats

- **v2 tokens**: `v2.{claims_b64url}.{sig_b64url}` — signed Ed25519 with tier, exp, iat
- **Pro tokens**: `QECT-PRO-{payload}.{sig}` — signed Pro license
- **Enterprise tokens**: `QECT-ENT-{payload}.{sig}` — signed Enterprise license
- **Community tokens**: `QECT-COMM-{payload}` — unsigned community (no sig)
- **Legacy tokens**: `{rid}.{email_b64}.{sig}` — backward-compatible legacy format

## Setting a License Key

```python
from qector_decoder_v3 import set_license_key, get_license_info
set_license_key("QECT-PRO-...")
info = get_license_info()
print(f"Tier: {info['tier']}, Max distance: {info['max_distance']}")
```

## CRL (Certificate Revocation List)

Place revoked identifiers (one per line, hex-encoded) at `~/.qector/revoked.txt` or override with `QECTOR_CRL_PATH`.

## Metered Billing (Enterprise)

Enterprise deployments use Stripe metered billing:
```python
from qector_decoder_v3 import record_shots, get_accumulated_shots
record_shots(1000)
total = get_accumulated_shots()
```
