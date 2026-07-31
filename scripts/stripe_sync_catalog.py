#!/usr/bin/env python3
"""Sync the QECTOR Stripe product catalog: descriptions, metadata, and meter.

What this fixes
---------------
1. **Empty descriptions.** Every live product had ``description: null``, so
   Checkout and the customer portal rendered a bare product name with no
   explanation of what the buyer receives.

2. **Missing ``qector_tier`` metadata.** ``stripe_integration.ensure_qector_products()``
   locates existing products by the ``qector_tier`` metadata key and creates a
   new product when it finds none. The live products were created by hand and
   carry no such key, so calling that function against the live account would
   have silently duplicated all six products and then created fresh prices under
   the duplicates. Stamping the metadata makes the function idempotent, which is
   what its docstring already promises.

3. **Missing usage meter.** ``src/stripe_billing.rs`` posts meter events named
   ``qec_syndrome_decodes``. No meter with that event name existed, so every
   metered flush in production was rejected by Stripe. This script creates it if
   absent.

Copy style
----------
Descriptions deliberately contain no dash characters of any kind, neither
hyphen, en dash, nor em dash. Ranges are written with the word "to" and
compound terms are spelled out.

Usage
-----
Dry run against test mode, printing every change without applying it::

    python scripts/stripe_sync_catalog.py --dry-run

Apply to test mode::

    python scripts/stripe_sync_catalog.py --apply

Apply to the live account (requires a key with Products and Billing Meters
write permission, which the current ``rk_live_`` restricted key does not have)::

    STRIPE_SECRET_KEY=sk_live_... python scripts/stripe_sync_catalog.py --apply --live

Optionally normalise product names to remove em dashes as well::

    python scripts/stripe_sync_catalog.py --apply --live --rename
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

try:
    import stripe
except ImportError:  # pragma: no cover
    sys.exit("the `stripe` package is required: pip install stripe")


METER_EVENT_NAME = "qec_syndrome_decodes"
METER_DISPLAY_NAME = "QEC Syndrome Decodes"

# Tier copy. Keyed by the `qector_tier` value that
# stripe_integration.PRICING uses, so the two stay in step.
#
# NOTE: no dash characters anywhere in this copy, by request.
CATALOG: dict[str, dict[str, str]] = {
    "evaluation": {
        "match": "Commercial Evaluation License",
        "name": "QECTOR Decoder v3: Commercial Evaluation License",
        "description": (
            "Sixty day commercial evaluation of the complete QECTOR Decoder v3 suite for up to "
            "three named users. Includes every decoder family in the platform: exact minimum "
            "weight perfect matching via Blossom, Sparse Blossom, Union Find, belief matching, "
            "and BP OSD for LDPC and qLDPC codes, together with the CPU batch and GPU "
            "accelerated decode paths. Ships with the full reproducibility toolkit so results "
            "can be independently verified: Stim and Sinter integration, PyMatching compatible "
            "interfaces, artifact hashed benchmark evidence, and circuit level logical error "
            "rate estimation with Wilson confidence intervals. Grants internal evaluation "
            "rights only. The full purchase price is credited toward any annual license taken "
            "within the evaluation window, so a successful evaluation costs nothing extra."
        ),
    },
    "solo_perpetual": {
        "match": "Solo / Indie Commercial (perpetual)",
        "name": "QECTOR Decoder v3: Solo and Indie Commercial, Perpetual",
        "description": (
            "Perpetual internal commercial license for a single named user, purchased once with "
            "no renewal. Grants permanent rights to the core decoder platform: exact Blossom "
            "and Sparse Blossom matching, Union Find, belief matching, BP OSD for LDPC and "
            "qLDPC codes, CPU batch decoding, and the GPU accelerated paths where hardware "
            "permits. Includes the reproducibility toolkit covering Stim and Sinter "
            "integration, PyMatching compatible interfaces, and circuit level logical error "
            "rate measurement. Intended for independent researchers, consultants, and solo "
            "practitioners who want durable rights without an annual commitment. Software "
            "updates released after the purchase date are not included. Redistribution, OEM "
            "embedding, and hosted API deployment are not included."
        ),
    },
    "solo_annual": {
        "match": "Solo / Indie Commercial (annual)",
        "name": "QECTOR Decoder v3: Solo and Indie Commercial, Annual",
        "description": (
            "Annual internal commercial license for a single named user, including twelve "
            "months of updates and new releases. Covers the complete decoder suite: exact "
            "Blossom and Sparse Blossom matching, Union Find, belief matching, BP OSD for LDPC "
            "and qLDPC codes, CPU batch decoding, and GPU accelerated decode paths. Includes "
            "the reproducibility toolkit with Stim and Sinter integration, PyMatching "
            "compatible interfaces, artifact hashed benchmark evidence, and circuit level "
            "logical error rate estimation with confidence intervals. Best suited to "
            "independent researchers and solo practitioners who want to stay current with "
            "ongoing decoder accuracy and performance work. Redistribution, OEM embedding, and "
            "hosted API deployment are not included."
        ),
    },
    "startup": {
        "match": "Startup / Growth Team",
        "name": "QECTOR Decoder v3: Startup and Growth Team",
        "description": (
            "Annual team license for up to ten named users at early stage quantum companies. "
            "Provides the complete decoder platform including the advanced BP OSD and LDPC "
            "workflows for qLDPC code families such as bivariate bicycle and hypergraph "
            "product codes, alongside exact and sparse matching decoders, Union Find, belief "
            "matching, and the CPU batch and GPU accelerated decode paths. Includes the full "
            "reproducibility toolkit: Stim and Sinter harness integration, PyMatching "
            "compatible interfaces, artifact hashed benchmark evidence, and circuit level "
            "logical error rate estimation with Wilson confidence intervals. Ninety days of "
            "priority email support are included from the activation date, covering "
            "integration questions and decoder selection guidance. Sized for internal "
            "prototype development and pre production research."
        ),
    },
    "professional": {
        "match": "Professional / Lab",
        "name": "QECTOR Decoder v3: Professional and Lab",
        "description": (
            "Annual license for up to twenty five named users, built for funded laboratories "
            "and commercial research and development teams. Grants full platform access "
            "including every decoder family, the advanced qLDPC and LDPC workflows, high "
            "throughput CPU batch processing, and the GPU accelerated decode paths. Includes "
            "the complete reproducibility and benchmarking toolkit with Stim and Sinter "
            "integration, PyMatching compatible interfaces, artifact hashed evidence bundles, "
            "and circuit level logical error rate estimation with confidence intervals for "
            "defensible published results. Carries credit for one Validation Report Package, a "
            "structured proof of value engagement in which decoder accuracy and throughput are "
            "measured on your own codes and noise model and delivered as a citable report."
        ),
    },
    "evaluation_legacy": {
        "match": "60 Days Commercial Evaluation License",
        "name": "QECTOR Decoder v3: Commercial Evaluation License, 60 Days (Legacy)",
        "description": (
            "Legacy listing for the sixty day commercial evaluation, retained so existing "
            "purchase links and historical invoices continue to resolve. New buyers should use "
            "the current Commercial Evaluation License listing instead, which carries the same "
            "sixty day internal evaluation rights across the full decoder suite and the same "
            "full credit toward any annual license. This product is not advertised on the "
            "pricing page and is kept active only for continuity of existing customer records."
        ),
    },
}


def _cli_key(live: bool) -> str:
    """Read the Stripe key: explicit env var first, else the CLI config."""
    env = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    if env:
        return env
    field = "live_mode_api_key" if live else "test_mode_api_key"
    # `shutil.which` so this resolves stripe.exe / stripe.cmd on Windows, where a
    # bare "stripe" argv[0] raises FileNotFoundError from CreateProcess.
    exe = shutil.which("stripe")
    if exe is None:
        raise SystemExit(
            "the Stripe CLI is not on PATH and STRIPE_SECRET_KEY is unset.\n"
            "Either set STRIPE_SECRET_KEY or install the CLI: https://stripe.com/docs/stripe-cli"
        )
    out = subprocess.run([exe, "config", "--list"], capture_output=True, text=True, check=False)
    for line in out.stdout.splitlines():
        if line.strip().startswith(field):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit(
        f"no Stripe key found. Set STRIPE_SECRET_KEY or run `stripe login`.\n"
        f"(looked for {field} in `stripe config --list`)"
    )


def sync_products(client, apply: bool, rename: bool) -> int:
    products = client.Product.list(active=True, limit=100).data
    print(f"found {len(products)} active products\n")
    changed = 0

    for tier, spec in CATALOG.items():
        target = next((p for p in products if spec["match"] in (p.name or "")), None)
        if target is None:
            print(f"  SKIP  {tier}: no product matching {spec['match']!r}")
            continue

        md = target.metadata
        md = md.to_dict() if hasattr(md, "to_dict") else dict(md or {})

        updates: dict = {}
        if (target.description or "") != spec["description"]:
            updates["description"] = spec["description"]
        if md.get("qector_tier") != tier:
            updates["metadata"] = {**md, "qector_tier": tier}
        if rename and target.name != spec["name"]:
            updates["name"] = spec["name"]

        if not updates:
            print(f"  OK    {tier}: already current ({target.id})")
            continue

        fields = ", ".join(sorted(updates))
        print(f"  {'APPLY' if apply else 'WOULD'} {tier}: {target.id} [{fields}]")
        if "description" in updates:
            print(f"          description: {len(spec['description'])} chars")
        if "metadata" in updates:
            print(f"          qector_tier={tier}  (makes ensure_qector_products idempotent)")
        if "name" in updates:
            print(f"          name: {target.name!r} -> {spec['name']!r}")

        if apply:
            client.Product.modify(target.id, **updates)
        changed += 1

    return changed


def sync_meter(client, apply: bool) -> bool:
    meters = client.billing.Meter.list(limit=100).data
    existing = next((m for m in meters if m.event_name == METER_EVENT_NAME), None)
    if existing is not None:
        print(f"\n  OK    meter {METER_EVENT_NAME!r} exists ({existing.id}, status={existing.status})")
        return False

    print(f"\n  {'APPLY' if apply else 'WOULD'} create meter {METER_EVENT_NAME!r}")
    print("          aggregation=sum  value_key=value  customer_key=stripe_customer_id")
    print("          (matches the body built by src/stripe_billing.rs)")
    if apply:
        m = client.billing.Meter.create(
            display_name=METER_DISPLAY_NAME,
            event_name=METER_EVENT_NAME,
            default_aggregation={"formula": "sum"},
            value_settings={"event_payload_key": "value"},
            customer_mapping={"type": "by_id", "event_payload_key": "stripe_customer_id"},
        )
        print(f"          created {m.id}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="target the live account (default: test)")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--dry-run", action="store_true", help="explicit dry run (the default)")
    ap.add_argument("--rename", action="store_true", help="also normalise product names to remove em dashes")
    args = ap.parse_args(argv)

    apply = args.apply and not args.dry_run
    mode = "LIVE" if args.live else "TEST"
    print(f"=== QECTOR Stripe catalog sync [{mode}] {'APPLY' if apply else 'DRY RUN'} ===\n")

    if apply and args.live:
        print("!! writing to the LIVE account !!\n")

    stripe.api_key = _cli_key(args.live)
    changed = sync_products(stripe, apply, args.rename)
    meter_changed = sync_meter(stripe, apply)

    print(f"\n{changed} product(s) {'updated' if apply else 'would change'}"
          f"{', meter created' if meter_changed and apply else ''}")
    if not apply:
        print("\nRe-run with --apply to write. Add --live to target the live account.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
