# Commercial Licensing

QECTOR Decoder v3 is source-available under the
**PolyForm Noncommercial License 1.0.0** (see `LICENSE`). Personal, academic,
educational and other non-commercial use is free. Universities, public research
organisations, charities and government institutions may use it free whatever
their source of funding.

A paid licence is required for commercial use, including:

- use by or for a company or startup, including internal R&D and operations;
- company work on government, defence or university contracts;
- paid consulting that uses QECTOR;
- hosted, SaaS or API deployments;
- OEM embedding, product integration and redistribution.

Plans and checkout: **[qector.store/pricing](https://qector.store/pricing/)**.
Questions: **[admin@qector.store](mailto:admin@qector.store)**.

## Plans

Prices are in US dollars; applicable tax is added at checkout. Every plan has a
14-day money-back guarantee on the first payment. Subscriptions renew
automatically and can be cancelled at any time.

| Plan | Price | Users | Includes |
| --- | --- | --- | --- |
| Free commercial trial | $0 for 14 days | Up to 5 | Pro features for evaluation. No card required; one trial per e-mail per year |
| Pro | $79 / month or $790 / year | 1 named user | All decoders, GPU (CUDA/OpenCL) batch decoding, internal production use, e-mail support |
| Pro Perpetual | $1,990 one-time | 1 named user | Pro rights for every 3.x release, 12 months of support |
| Team | $3,900 / year | Up to 10 | Everything in Pro, priority support, onboarding call |
| Lab | $9,900 / year | Up to 25 | Everything in Team, code-distance tier 63, support SLA, one Validation Report |
| Pilot | $1,500 one-time | Team | 60-day evaluation licence with an integration call; creditable toward Team or Lab within 90 days |
| Enterprise | From $25,000 / year | Unlimited | SaaS / hosted API and OEM options, named support engineer, signed agreement |

Academic institutions that want a paid plan, for support or to pass commercial
rights to an industry partner, receive 40% off Team and Lab.

Licences bought under earlier plans (Evaluation, Solo/Indie, Startup/Growth,
Professional) keep their original terms.

## Professional services

| Service | Price | Scope |
| --- | --- | --- |
| Validation Sprint | $3,750 one-time | 10–20 business days. A 60-day Pilot licence for up to 3 users, up to 3 of your workloads run with QECTOR, a reproducible artefact bundle (environment, input hashes, results, manifest), and a 60-minute review call |
| Validation Report | $5,000 one-time | Independent review of your decoding setup and its reproducibility (one included with Lab) |
| Workflow integration | $7,500–$12,000, scoped | Engineering to integrate QECTOR into your pipeline or custom LDPC/qLDPC code families |
| Priority support | $5,000 / year | Private support channel and priority bug review |
| Integration support | $10,000 / year | Installation, configuration and integration guidance |
| Custom engineering | $150 / hour | Fixes, scripts, reports and workflow adaptation |
| Technical advisory | $2,000 / day | Architecture review and QEC workflow consulting |

## Rights not included by default

Unless a written agreement grants them, no plan includes redistribution,
sublicensing, OEM, hosted API or SaaS rights, trademark or patent rights,
exclusivity, warranty or indemnification.

## Licence delivery and activation

Licences are delivered by e-mail as Ed25519-signed tokens, usually within
minutes of payment. Tokens are verified offline against a public key in the
package, with no licence server and no network access. Activation steps are in
the [README](README.md#activating-a-licence).

## Metered billing (Enterprise)

Enterprise agreements can include usage-based billing:

```python
from qector_decoder_v3 import record_shots, get_accumulated_shots
from qector_decoder_v3.stripe_integration import flush_metered_usage

record_shots(1000)
total = get_accumulated_shots()
```

Usage is reported to Stripe on job completion or with
`flush_metered_usage(customer_id, api_key)`. Setting `QECTOR_OFFLINE=1`
disables every outbound billing call.

*This document supersedes earlier commercial pricing statements for QECTOR Decoder v3.*
