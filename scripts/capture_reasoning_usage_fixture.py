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
  * `/api/v1/model/{author}/{slug}` returns a pricing object keyed
    `prompt` / `completion` / `request` / `image` / `web_search` /
    `internal_reasoning` / `input_cache_read` / `input_cache_write`, as STRINGS
    in USD *per token* (the curated registry is per MILLION tokens -- do not mix).
    (docs/guides/overview/models)
  * The reachable docs expose no per-provider pricing route; model pricing is
    published model-wide with a `top_provider`. We therefore record the rates we
    used plus the VERIFIED pinned provider the reading is scoped to, rather than
    claiming a per-provider rate we cannot source.

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
OPENROUTER_MODEL_URL = "https://openrouter.ai/api/v1/model/{model}"
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


def select_provider_from_metadata(data, provider_requested):
    """Return the provider name of the SELECTED endpoint, verified against the pin.

    Raises CaptureError when routing metadata is absent, exposes no selected
    endpoint, or names a provider other than the one requested.
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
    if _normalize_provider(served) != _normalize_provider(provider_requested):
        raise CaptureError(
            f"routing mismatch: pinned provider {provider_requested!r} but the request was "
            f"served by {served!r}. The billing verdict is provider-specific, so a fallback "
            "route invalidates the capture; re-run with allow_fallbacks disabled or pin the "
            "provider that actually serves this model."
        )
    return served


def fetch_price_authority(model, *, get=httpx.get):
    """Fetch the pricing OpenRouter publishes for this model, with provenance.

    Returns a dict of the exact rates used, their source and fetch time, so the
    gate can re-derive the verdict from stored evidence. Rates are USD PER TOKEN
    (strings upstream); the curated registry is per MILLION tokens.
    """
    url = OPENROUTER_MODEL_URL.format(model=model)
    resp = get(url, timeout=30.0)
    resp.raise_for_status()
    body = resp.json()
    entry = body.get("data") if isinstance(body.get("data"), dict) else body
    pricing = (entry or {}).get("pricing") or {}

    def rate(key):
        value = pricing.get(key)
        return float(value) if value not in (None, "") else None

    prompt_rate = rate("prompt")
    completion_rate = rate("completion")
    if prompt_rate is None or completion_rate is None:
        raise CaptureError(
            f"{url} returned no prompt/completion pricing; cannot price this capture"
        )

    return {
        "source": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "units": "USD per token",
        "prompt_per_token": prompt_rate,
        "completion_per_token": completion_rate,
        # A non-zero internal_reasoning price is an explicit, named separate
        # charge for reasoning tokens -- the strongest available signal.
        "internal_reasoning_per_token": rate("internal_reasoning"),
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
    rather than returning unverified evidence."""
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

    provider_selected = select_provider_from_metadata(data, provider_slug)

    usage = data.get("usage", {})
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    prompt = usage.get("prompt_tokens", 0)
    reported_cost = usage.get("cost")
    upstream_cost = (usage.get("cost_details") or {}).get("upstream_inference_cost")

    price = fetch_price_authority(model, get=get)
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
            "imports. The verdict is derived from the STORED price_authority rates (USD per "
            "token, captured with source + fetched_at), never from the mutable curated "
            "registry, so it stays reproducible. It is scoped to provider_selected, which "
            "was verified against provider_requested via openrouter_metadata."
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
    print(f"model={model}  provider_requested={provider_slug}  provider_selected={fixture['provider_selected']}")
    print(f"usage={fixture['usage'].get('completion_tokens')} completion tokens, "
          f"{(fixture['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens')} reasoning")
    print(f"cost={fixture['cost']}  completion_priced={price['completion_priced_cost']}")
    print(f"rates: prompt={price['prompt_per_token']}/tok completion={price['completion_per_token']}/tok "
          f"internal_reasoning={price['internal_reasoning_per_token']} (source {price['source']})")
    print(f"billing_relationship={fixture['billing_relationship']}")
    print(fixture["_conclusion"])
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
