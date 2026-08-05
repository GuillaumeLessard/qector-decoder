# QECTOR Enterprise License Server Service Level Agreement (SLA)

This document specifies the availability, cryptographic verification, and operational guarantees for QECTOR Enterprise licensing.

---

## 1. Offline Cryptographic Autonomy (0-Latency / Offline First)

- **Ed25519 Standalone Verification**: License tokens (format `v2` with `QECT-PRO-` / `QECT-ENT-` prefixes) are signed using Ed25519 keypairs. Verification takes place **entirely offline** within the compiled Rust core (`src/license.rs`).
- **Zero Runtime Dependencies**: The decoder does **not** make blocking HTTP calls to a license server during decoding. Decoding latency is strictly local (0 ms license verification overhead).

---

## 2. Server Availability & Uptime Target

- **License Minting & Renewal API**: **99.9% Uptime Target** (monthly availability) for the Cloudflare Worker license fulfillment service (`https://qector.store`).
- **Revocation List (CRL) Sync**: Offline revocation checking via `~/.qector/revoked.txt` with optional non-blocking daily background refresh.

---

## 3. Commercial Support & Response SLAs

| Support Tier | Target Response Time | Contact Channel |
|---|---|---|
| **Community** | Best effort | GitHub Issues |
| **Pro** | < 24 business hours | `support@qector.store` |
| **Enterprise** | < 4 business hours / Dedicated SLA | `enterprise@qector.store` |
