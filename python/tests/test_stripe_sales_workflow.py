"""
test_stripe_sales_workflow.py - flawless sales-workflow surface:
products registry, ensure_qector_products idempotency, labs-bot notification
(success + failure isolation), and package-internal signing error path.
All network calls are mocked; no live API access in unit tests.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import qector_decoder_v3.stripe_integration as si
from qector_decoder_v3.license import create_license_token, verify_license_token
from qector_decoder_v3.stripe_integration import (
    PRICING,
    _notify_labs_bot,
    get_stripe_keys,
    handle_stripe_webhook_payload,
)


class TestPricingRegistry:
    def test_self_serve_tiers_present(self):
        assert "evaluation" in PRICING
        assert "solo_perpetual" in PRICING
        assert "solo_annual" in PRICING

    def test_amounts_match_commercial_md(self):
        assert PRICING["evaluation"]["amount_cents"] == 49900
        assert PRICING["solo_perpetual"]["amount_cents"] == 89900
        assert PRICING["solo_annual"]["amount_cents"] == 129900
        assert PRICING["solo_annual"]["interval"] == "year"

    def test_commercial_alias_matches_evaluation(self):
        assert PRICING["commercial"] is PRICING["evaluation"]

    def test_keys_snapshot_includes_bot_flag(self):
        keys = get_stripe_keys()
        assert "labs_bot_configured" in keys
        assert keys["secret_key_configured"] is True


class TestEnsureProducts:
    @patch("stripe.Price.create")
    @patch("stripe.Price.list")
    @patch("stripe.Product.create")
    @patch("stripe.Product.list")
    def test_creates_missing_products_and_prices(self, mock_plist, mock_pcreate, mock_price_list, mock_price_create):
        mock_plist.return_value = MagicMock(data=[])

        def _mk_product(**kwargs):
            m = MagicMock()
            m.id = "prod_" + kwargs["metadata"]["qector_tier"]
            m.metadata = kwargs["metadata"]
            return m

        mock_pcreate.side_effect = _mk_product
        mock_price_list.return_value = MagicMock(data=[])

        def _mk_price(**kwargs):
            m = MagicMock()
            m.id = "price_" + kwargs["product"]
            return m

        mock_price_create.side_effect = _mk_price

        ids = si.ensure_qector_products()
        assert set(ids) == {"evaluation", "solo_perpetual", "solo_annual"}
        assert mock_pcreate.call_count == 3
        assert mock_price_create.call_count == 3

    @patch("stripe.Price.create")
    @patch("stripe.Price.list")
    @patch("stripe.Product.create")
    @patch("stripe.Product.list")
    def test_idempotent_when_products_exist(self, mock_plist, mock_pcreate, mock_price_list, mock_price_create):
        existing = []
        for tier in ("evaluation", "solo_perpetual", "solo_annual"):
            m = MagicMock()
            m.id = f"prod_{tier}"
            m.metadata = {"qector_tier": tier}
            existing.append(m)
        mock_plist.return_value = MagicMock(data=existing)

        def _price_list(product, active, limit):
            spec = next(s for t, s in PRICING.items() if t != "commercial" and t in product)
            m = MagicMock()
            m.id = "price_existing"
            m.unit_amount = spec["amount_cents"]
            m.recurring = {"interval": spec["interval"]} if spec["interval"] else None
            if m.recurring:
                m.recurring = MagicMock(interval=spec["interval"])
            return MagicMock(data=[m])

        mock_price_list.side_effect = _price_list
        ids = si.ensure_qector_products()
        assert mock_pcreate.call_count == 0
        assert mock_price_create.call_count == 0
        assert all(v == "price_existing" for v in ids.values())

    def test_raises_without_secret_key(self):
        with patch.object(si, "STRIPE_SECRET_KEY", ""), pytest.raises(RuntimeError, match="STRIPE_SECRET_KEY"):
            si.ensure_qector_products()


class TestLabsBot:
    def test_returns_false_when_url_unset(self):
        with patch.object(si, "QECTOR_LABS_BOT_WEBHOOK_URL", ""):
            assert _notify_labs_bot("test") is False

    @patch("urllib.request.urlopen")
    def test_posts_discord_payload(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 204
        mock_urlopen.return_value.__enter__.return_value = resp
        with patch.object(si, "QECTOR_LABS_BOT_WEBHOOK_URL", "https://discord.test/hook"):
            ok = _notify_labs_bot("sale!", fields={"tier": "evaluation"})
        assert ok is True
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["content"] == "sale!"
        assert body["embeds"][0]["fields"][0]["name"] == "tier"

    @patch("urllib.request.urlopen", side_effect=OSError("network down"))
    def test_failure_never_raises(self, _mock):
        with patch.object(si, "QECTOR_LABS_BOT_WEBHOOK_URL", "https://discord.test/hook"):
            assert _notify_labs_bot("sale!") is False

    @patch("urllib.request.urlopen", side_effect=OSError("network down"))
    def test_webhook_fulfillment_survives_bot_outage(self, _mock):
        with patch.object(si, "QECTOR_LABS_BOT_WEBHOOK_URL", "https://discord.test/hook"):
            payload = json.dumps(
                {
                    "type": "checkout.session.completed",
                    "data": {"object": {"id": "cs_bot_outage", "customer_email": "a@b.c"}},
                }
            ).encode()
            result = handle_stripe_webhook_payload(payload=payload, sig_header="", webhook_secret="")
        assert result["issued"] is True
        assert verify_license_token(result["license_token"], "a@b.c") is True


class TestPackageInternalSigning:
    def test_sign_and_verify_roundtrip(self):
        token = create_license_token("rec_xyz", "Buyer@Qector.Store")
        assert verify_license_token(token, "buyer@qector.store") is True
        assert verify_license_token(token) is True  # self-contained 3-part

    def test_missing_private_key_is_loud(self):
        with (
            patch.dict(os.environ, {"QECTOR_LICENSE_PRIVATE_KEY_B64": ""}),
            pytest.raises(RuntimeError, match="QECTOR_LICENSE_PRIVATE_KEY_B64"),
        ):
            create_license_token("rec_nokey", "a@b.c")

    def test_no_demo_fallback_in_package(self):
        """The package must never mint tokens the production pubkey rejects."""
        token = create_license_token("rec_real", "real@qector.store")
        assert not token.endswith(".demo_token")
        assert verify_license_token(token, "real@qector.store") is True
