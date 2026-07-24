"""Per-ENDPOINT price resolution from OpenRouter's public `/endpoints` API.

Billing is per ENDPOINT -- not per model, and not per provider name. On
openai/gpt-4o-mini the tags `azure` and `azure/swedencentral` are both served by
provider "Azure" yet price completion at 0.0000006 vs 0.00000066. So a pin must
name an exact tag, and an absent or ambiguous match fails rather than guessing a
rate.

`GET /api/v1/models/{author}/{slug}/endpoints` is PUBLIC (no key) and free -- it
is metadata, not inference, so resolving prices here costs nothing and is safe to
run before any paid call.

Shared by two callers that must agree on what a pinned route costs:
  * `scripts/capture_reasoning_usage_fixture.py` (D1) -- prices a billing capture.
  * `backend/reasoning_probe.py` -- bounds the PAID probe's spend ceiling. The
    probe must bound each call by ITS OWN endpoint's price: one uniform bound
    across all models is dominated by the priciest model in the registry and
    overstates the sweep's ceiling by an order of magnitude, which forces the
    maintainer to authorize far more headroom than the sweep can actually spend.

Rates are USD PER TOKEN (strings upstream); the curated registry is per MILLION.
Deriving a bound from the registry's model-wide `pricing` would be unsound: a
pinned endpoint's price can exceed the curated model-wide value, so a
registry-derived bound could UNDER-count and breach the authorized ceiling.
"""
from datetime import datetime, timezone

import httpx

from . import config

# Built from the CONFIGURED base URL, exactly like config.OPENROUTER_API_URL and
# openrouter_client's URLs: paid calls and the prices that bound them must come from
# the SAME service. Hard-coding openrouter.ai would price a different service than
# the one being billed whenever a relay/proxy is configured, and would be
# unreachable at all where openrouter.ai is blocked (audit §4.3).
OPENROUTER_ENDPOINTS_URL = f"{config.OPENROUTER_BASE_URL}/models/{{model}}/endpoints"


class EndpointPricingError(RuntimeError):
    """A pinned endpoint's rate could not be resolved with certainty. Callers must
    treat this as fatal for the route in question: guessing a rate would silently
    invalidate whatever guarantee (billing evidence, spend ceiling) depends on it."""


def fetch_endpoint_pricing(model, provider_tag, *, get=httpx.get):
    """Resolve the EXACT pinned endpoint and return its own pricing + provenance.

    The tag must match exactly -- no prefix or provider-name fallback -- and an
    absent or ambiguous match raises rather than guessing a rate.

    Rates are USD PER TOKEN (strings upstream); the curated registry is per MILLION.
    """
    url = OPENROUTER_ENDPOINTS_URL.format(model=model)
    resp = get(url, timeout=30.0)
    resp.raise_for_status()
    body = resp.json()
    data = body.get("data") if isinstance(body.get("data"), dict) else body
    endpoints = (data or {}).get("endpoints") or []

    matches = [e for e in endpoints if e.get("tag") == provider_tag]
    if not matches:
        raise EndpointPricingError(
            f"no endpoint on {model} has tag == {provider_tag!r}. Available tags: "
            f"{sorted(str(e.get('tag')) for e in endpoints)}. Pricing is per endpoint, so the "
            "pin must name an exact tag -- variant tags (e.g. 'azure/swedencentral') price "
            "differently from their base tag."
        )
    if len(matches) > 1:
        raise EndpointPricingError(
            f"tag {provider_tag!r} matched {len(matches)} endpoints on {model} -- ambiguous, "
            "so no single rate can be attributed to this capture"
        )

    endpoint = matches[0]
    pricing = endpoint.get("pricing") or {}

    def rate(key):
        value = pricing.get(key)
        return float(value) if value not in (None, "") else None

    prompt_rate = rate("prompt")
    completion_rate = rate("completion")
    if prompt_rate is None or completion_rate is None:
        raise EndpointPricingError(
            f"endpoint {provider_tag!r} on {model} publishes no prompt/completion pricing; "
            "cannot price this capture"
        )

    return {
        "source": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "units": "USD per token",
        # Endpoint identity -- the reading is valid for THIS route only.
        "tag": endpoint.get("tag"),
        "provider_name": endpoint.get("provider_name"),
        "endpoint_name": endpoint.get("name"),
        "endpoint_model_id": endpoint.get("model_id"),
        "endpoint_model_name": endpoint.get("model_name"),
        "prompt_per_token": prompt_rate,
        "completion_per_token": completion_rate,
        # A non-zero internal_reasoning price on THIS endpoint is an explicit,
        # named separate charge for reasoning tokens.
        "internal_reasoning_per_token": rate("internal_reasoning"),
        "pricing_raw": pricing,
    }
