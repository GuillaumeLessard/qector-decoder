"""
QECTOR Decoder v3 - Stripe Webhook & License Fulfillment Server
Start server with: python stripe_webhook_server.py
Forward live/test events using Stripe CLI: stripe listen --forward-to localhost:8000/webhook
"""

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse

from qector_decoder_v3.stripe_integration import (
    get_stripe_keys,
    create_checkout_session,
    handle_stripe_webhook_payload,
)

app = FastAPI(
    title="QECTOR Decoder v3 License Server",
    description="Stripe Checkout and Ed25519 Cryptographic License Fulfillment Endpoint",
    version="0.6.7"
)


@app.get("/health")
def health_check():
    return {"status": "ok", "stripe": get_stripe_keys()}


@app.post("/create-checkout-session")
async def api_create_checkout_session(request: Request):
    try:
        data = await request.json()
        email = data.get("email", "")
        tier = data.get("tier", "commercial")
        if not email:
            raise HTTPException(status_code=400, detail="email is required")

        result = create_checkout_session(customer_email=email, license_tier=tier)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    try:
        result = handle_stripe_webhook_payload(payload=payload, sig_header=sig_header)
        return JSONResponse(content=result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("=== Starting QECTOR Stripe License Fulfillment Server on http://127.0.0.1:8000 ===")
    print("Stripe CLI forward command: stripe listen --forward-to localhost:8000/webhook")
    uvicorn.run(app, host="127.0.0.1", port=8000)
