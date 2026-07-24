"""Capture a REAL OpenRouter usage payload from a reasoning turn.

D1 (v1.3.0) must decide from EVIDENCE, not memory, whether reasoning tokens are
already inside `completion_tokens` (the meter is already truthful and D1's
breakdown is display-only) or billed SEPARATELY (D1 adds a provider-specific
reasoning line). This makes one raw-HTTP reasoning call (SDKs can normalize away
the fields we're testing) and writes tests/fixtures/reasoning_usage_fixture.json.

The decision is PROVIDER-SPECIFIC, so the capture:
  * REQUIRES a provider slug and pins it (`provider.order` + `allow_fallbacks:false`);
  * opts into router metadata and VERIFIES the served endpoint matches the pin;
  * records the pricing rates it used, with source + timestamp, so the gate can
    re-derive the verdict later from stored evidence rather than today's mutable
    registry.

If routing metadata is absent, has no selected endpoint, or names a different
provider than requested, the capture FAILS and leaves the existing fixture
untouched -- an unverified route is not evidence.

Verified against current OpenRouter docs (2026-07):
  * `X-OpenRouter-Metadata: enabled` opts into `openrouter_metadata`, whose
    `endpoints.available[]` entries carry `provider` / `model` / `selected`.
    Works for non-streaming requests. (docs/guides/features/router-metadata)
  * `usage.cost` is "the total amount charged to your account";
    `usage.cost_details.upstream_inference_cost` is the upstream provider's
    charge. `usage: {include: true}` is DEPRECATED and has no effect -- usage is
    always returned. (docs/cookbook/administration/usage-accounting)
  * `GET /api/v1/models/{author}/{slug}/endpoints` is PUBLIC (no key) and returns
    `data.endpoints[]`, each carrying `tag`, `provider_name`, `name`, `model_id`,
    `model_name` and its OWN `pricing` object (strings, USD *per token* -- the
    curated registry is per MILLION tokens, do not mix). Verified live 2026-07:
    openai/gpt-4o-mini returns tag `azure` and tag `azure/swedencentral` BOTH with
    provider_name "Azure" but completion rates 0.0000006 vs 0.00000066. Billing is
    therefore per ENDPOINT, not per model and not per provider name -- pricing must
    key on the exact `tag`.
  * Consequently the router's selected display name is checked against the pinned
    ENDPOINT's `provider_name` (so a variant tag like `azure/swedencentral`, served
    as "Azure", still verifies) while the PRICE comes from the exact tag.

`classify_billing` below is the SINGLE SOURCE of the inside/separate decision --
tests import it from here rather than re-implementing the thresholds.

Usage:
    OPENROUTER_API_KEY=... uv run python scripts/capture_reasoning_usage_fixture.py <model_id> <provider_slug>
"""
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Per-ENDPOINT pricing (public, no key). Rates differ between tags of the SAME
# provider, so this is the only authority that can price a pinned route.
OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "reasoning_usage_fixture.json"

# Rounding + rate drift allowance, as a fraction of the completion-priced cost.
# The separate-billing signal must clear 2x this to be distinguishable at all.
NOISE_FRACTION = 0.02
NOISE_FLOOR = 1e-6


class CaptureError(RuntimeError):
    """Capture could not produce trustworthy evidence; the fixture is left alone."""


def _normalize_provider(name):
    """Provider pins are slugs ('openai'); router metadata reports display names
    ('OpenAI'). Compare on a case/punctuation-insensitive basis."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def select_provider_from_metadata(data, expected_provider_name):
    """Return the provider display name of the SELECTED endpoint, verified against
    the pinned endpoint's `provider_name`.

    Checked against provider_name rather than the tag on purpose: a variant tag
    such as `azure/swedencentral` is served as "Azure", so tag-matching would
    reject a correct route. A genuinely different provider still fails.

    Raises CaptureError when routing metadata is absent, exposes no selected
    endpoint, or names a different provider.
    """
    metadata = data.get("openrouter_metadata")
    if not metadata:
        raise CaptureError(
            "response carried no `openrouter_metadata` -- the request must send "
            "'X-OpenRouter-Metadata: enabled' and the account must be able to opt in. "
            "Without it the served route is unverified, so this is not provider evidence."
        )

    available = (metadata.get("endpoints") or {}).get("available") or []
    selected = [e for e in available if e.get("selected")]
    if not selected:
        raise CaptureError(
            f"`openrouter_metadata` exposed no selected endpoint (available={len(available)}) "
            "-- cannot attribute this billing reading to a provider"
        )

    served = selected[0].get("provider")
    if _normalize_provider(served) != _normalize_provider(expected_provider_name):
        raise CaptureError(
            f"routing mismatch: the pinned endpoint belongs to provider "
            f"{expected_provider_name!r} but the request was served by {served!r}. A fallback "
            "route invalidates a per-endpoint billing reading; re-run with allow_fallbacks "
            "disabled or pin a tag on the provider that actually serves this model."
        )
    return served


def fetch_endpoint_pricing(model, provider_tag, *, get=httpx.get):
    """Resolve the EXACT pinned endpoint and return its own pricing + provenance.

    Billing is per endpoint: `azure` and `azure/swedencentral` are both provider
    "Azure" on openai/gpt-4o-mini yet price completion at 0.0000006 vs 0.00000066.
    So the tag must match exactly -- no prefix or provider-name fallback -- and an
    absent or ambiguous match fails rather than guessing a rate.

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
        raise CaptureError(
            f"no endpoint on {model} has tag == {provider_tag!r}. Available tags: "
            f"{sorted(str(e.get('tag')) for e in endpoints)}. Pricing is per endpoint, so the "
            "pin must name an exact tag -- variant tags (e.g. 'azure/swedencentral') price "
            "differently from their base tag."
        )
    if len(matches) > 1:
        raise CaptureError(
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
        raise CaptureError(
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


def classify_billing(
    *,
    reasoning_tokens,
    reported_cost,
    completion_priced_cost,
    completion_rate_per_token,
    internal_reasoning_rate_per_token=None,
):
    """SINGLE SOURCE of the D1 inside/separate decision. Tests import this.

    Order of evidence:
      1. A non-zero `internal_reasoning` price is an explicit separate charge.
      2. Otherwise compare the billed cost against the completion-priced cost:
           INSIDE   -> surcharge ~= 0
           SEPARATE -> surcharge ~= reasoning_tokens * completion_rate
         judged against the SIGNAL, never a flat % of completion cost: a small
         reasoning count makes the separate surcharge indistinguishable from
         noise, which must resolve to 'ambiguous', never a false 'inside'.

    Returns (relationship, detail); relationship is one of
    'no_reasoning' | 'unknown' | 'ambiguous' | 'inside' | 'separate'.
    """
    detail = {"surcharge": None, "expected_separate_surcharge": None, "noise": None}

    if not reasoning_tokens:
        return "no_reasoning", detail
    if internal_reasoning_rate_per_token:
        detail["internal_reasoning_rate_per_token"] = internal_reasoning_rate_per_token
        return "separate", detail
    if (
        reported_cost is None
        or completion_priced_cost is None
        or completion_rate_per_token is None
    ):
        return "unknown", detail

    surcharge = reported_cost - completion_priced_cost
    expected_separate = reasoning_tokens * completion_rate_per_token
    noise = max(NOISE_FLOOR, NOISE_FRACTION * completion_priced_cost)
    detail.update(
        surcharge=surcharge,
        expected_separate_surcharge=expected_separate,
        noise=noise,
    )

    if expected_separate <= 2 * noise:
        return "ambiguous", detail
    if surcharge <= noise:
        return "inside", detail
    if abs(surcharge - expected_separate) <= max(noise, 0.25 * expected_separate):
        return "separate", detail
    return "ambiguous", detail


CONCLUSIONS = {
    "no_reasoning": (
        "reasoning_tokens=0 -- this model/route did not reason; pick a probe-confirmed "
        "reasoning model or a different provider route and re-run"
    ),
    "unknown": (
        "missing billed cost or pricing -- cannot derive billing from cost; review and set "
        "billing_relationship manually"
    ),
    "ambiguous": (
        "the separate-billing signal is within noise, or the surcharge matches neither "
        "hypothesis -- capture a longer reasoning turn (more reasoning_tokens) or review "
        "manually; do NOT assume 'inside'"
    ),
    "inside": (
        "billed cost ~= completion-priced cost -> reasoning is INSIDE completion_tokens "
        "(already billed); D1's breakdown is display-only, DO NOT add to the total"
    ),
    "separate": (
        "reasoning is billed SEPARATELY on this route (explicit internal_reasoning price, "
        "or a billed surcharge matching reasoning_tokens x completion rate) -- D1 must add "
        "a provider-specific reasoning line"
    ),
}


def capture(model, provider_slug, api_key, *, post=httpx.post, get=httpx.get):
    """Run one pinned reasoning call and build the fixture. Raises CaptureError
    rather than returning unverified evidence.

    Resolves the pinned ENDPOINT first: its `provider_name` is what the router's
    selected display name is verified against, and its own pricing is the only
    rate used to classify."""
    price = fetch_endpoint_pricing(model, provider_slug, get=get)

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Reason step by step, then answer: a bat and ball cost $1.10; "
                       "the bat costs $1.00 more than the ball. How much is the ball?",
        }],
        "reasoning": {"effort": "high"},
        # Pin the upstream and forbid silent fallback so the billing reading is
        # attributable to a known route. `usage:{include:true}` is deprecated and
        # deliberately omitted -- usage is always returned.
        "provider": {"order": [provider_slug], "allow_fallbacks": False},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Metadata": "enabled",
    }

    resp = post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()

    provider_selected = select_provider_from_metadata(data, price["provider_name"])

    usage = data.get("usage", {})
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    prompt = usage.get("prompt_tokens", 0)
    reported_cost = usage.get("cost")
    upstream_cost = (usage.get("cost_details") or {}).get("upstream_inference_cost")

    completion_priced = (
        prompt * price["prompt_per_token"] + completion * price["completion_per_token"]
    )

    billing, detail = classify_billing(
        reasoning_tokens=reasoning,
        reported_cost=reported_cost,
        completion_priced_cost=completion_priced,
        completion_rate_per_token=price["completion_per_token"],
        internal_reasoning_rate_per_token=price["internal_reasoning_per_token"],
    )

    return {
        "_placeholder": False,
        "model": model,
        "provider_requested": provider_slug,
        "provider_selected": provider_selected,
        "routing_summary": (data.get("openrouter_metadata") or {}).get("summary"),
        "usage": usage,
        "cost": reported_cost,
        "upstream_inference_cost": upstream_cost,
        "price_authority": {**price, "completion_priced_cost": completion_priced},
        "classification_detail": detail,
        "billing_relationship": billing,
        "_conclusion": CONCLUSIONS[billing],
        "_note": (
            "billing_relationship comes from classify_billing() in "
            "scripts/capture_reasoning_usage_fixture.py -- the single source the D1 gate "
            "imports. The verdict is derived from the STORED price_authority rates of the "
            "EXACT pinned endpoint (matched on `tag`, USD per token, captured with source + "
            "fetched_at), never from model-wide or registry pricing, so it stays reproducible. "
            "Billing is per endpoint: variant tags of one provider price differently, which is "
            "why the price keys on `tag` while the router's selected display name is verified "
            "against that endpoint's `provider_name`."
        ),
    }


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        print(
            "error: usage: capture_reasoning_usage_fixture.py <model_id> <provider_slug>\n"
            "       both are required -- an unpinned capture cannot support a "
            "provider-specific billing decision",
            file=sys.stderr,
        )
        return 2
    model, provider_slug = sys.argv[1], sys.argv[2]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("error: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    try:
        fixture = capture(model, provider_slug, api_key)
    except CaptureError as exc:
        # Leave the existing fixture untouched: a failed capture must never
        # overwrite the placeholder (or a good prior capture) with junk.
        print(f"capture failed: {exc}", file=sys.stderr)
        print(f"{FIXTURE} left unchanged", file=sys.stderr)
        return 1

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=2), encoding="utf-8")

    price = fixture["price_authority"]
    print(f"model={model}  endpoint tag={price['tag']!r} provider_name={price['provider_name']!r} "
          f"({price['endpoint_name']})")
    print(f"router selected={fixture['provider_selected']!r} (verified against provider_name)")
    print(f"usage={fixture['usage'].get('completion_tokens')} completion tokens, "
          f"{(fixture['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens')} reasoning")
    print(f"cost={fixture['cost']}  completion_priced={price['completion_priced_cost']}")
    print(f"endpoint rates: prompt={price['prompt_per_token']}/tok completion={price['completion_per_token']}/tok "
          f"internal_reasoning={price['internal_reasoning_per_token']} (source {price['source']})")
    print(f"billing_relationship={fixture['billing_relationship']}")
    print(fixture["_conclusion"])
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
