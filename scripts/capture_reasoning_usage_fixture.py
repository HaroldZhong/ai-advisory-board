"""Capture a REAL OpenRouter usage payload from a reasoning turn.

D1 (v1.3.0) must decide from EVIDENCE, not memory, whether reasoning tokens are
already inside `completion_tokens` (OpenRouter norm -> the meter is already
truthful and D1's breakdown is display-only) or billed SEPARATELY (D1 adds a
provider-specific reasoning line). This script makes one raw-HTTP reasoning call
(SDKs can normalize away the fields we're testing -- brainstorm sec. 2.4 step 7)
with usage accounting on, then writes tests/fixtures/reasoning_usage_fixture.json.

Because the D1 decision is PROVIDER-SPECIFIC, the capture pins the provider (when
a slug is supplied) and always records the served route + the price evidence the
classifier used, so the fixture is reproducible rather than a one-off reading.

`classify_billing` below is the SINGLE SOURCE of the inside/separate decision --
tests import it from here rather than re-implementing the thresholds.

Standalone by design (no backend import): reads OPENROUTER_API_KEY from the env,
posts to the public endpoint, and reads pricing straight from the registry JSON.

Usage:
    OPENROUTER_API_KEY=... uv run python scripts/capture_reasoning_usage_fixture.py <model_id> [provider_slug]

Pick a <model_id> the Phase-A probe marked reasoning-capable, and pass the
[provider_slug] you want pinned (recommended -- support is provider-specific).
This script does NOT assert support; it records what actually came back.
"""
import json
import os
import sys
from pathlib import Path

import httpx

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "reasoning_usage_fixture.json"
REGISTRY = ROOT / "backend" / "model_registry.json"

# Rounding + registry-vs-actual rate drift allowance, as a fraction of the
# completion-priced cost. The separate-billing signal must clear 2x this to be
# distinguishable at all.
NOISE_FRACTION = 0.02
NOISE_FLOOR = 1e-6


def registry_pricing(model):
    """(input_per_M, output_per_M) from the curated registry, or None if the
    model isn't priced there."""
    models = json.loads(REGISTRY.read_text(encoding="utf-8")).get("models", [])
    entry = next((m for m in models if m.get("id") == model), None)
    if not entry:
        return None
    pricing = entry.get("pricing") or {}
    if pricing.get("input") is None or pricing.get("output") is None:
        return None
    return pricing["input"], pricing["output"]


def classify_billing(
    *,
    reasoning_tokens,
    completion_tokens,
    reported_cost,
    completion_priced_cost,
    output_rate,
):
    """SINGLE SOURCE of the D1 inside/separate decision. Tests import this.

    Two hypotheses predict different costs:
      INSIDE   -> surcharge ~= 0 (reasoning already inside completion_tokens)
      SEPARATE -> surcharge ~= reasoning_tokens * output_rate (charged on top)

    Classified against the SIGNAL, not a flat % of completion cost: a small
    reasoning count makes the separate surcharge indistinguishable from noise,
    which must resolve to 'ambiguous' -- never a false 'inside'.

    Returns (relationship, detail) where relationship is one of
    'no_reasoning' | 'unknown' | 'ambiguous' | 'inside' | 'separate'.
    """
    detail = {
        "surcharge": None,
        "expected_separate_surcharge": None,
        "noise": None,
    }

    if not reasoning_tokens:
        return "no_reasoning", detail
    if reported_cost is None or completion_priced_cost is None or output_rate is None:
        return "unknown", detail

    surcharge = reported_cost - completion_priced_cost
    expected_separate = reasoning_tokens / 1_000_000 * output_rate
    noise = max(NOISE_FLOOR, NOISE_FRACTION * completion_priced_cost)
    detail.update(
        surcharge=surcharge,
        expected_separate_surcharge=expected_separate,
        noise=noise,
    )

    # The two hypotheses must be far enough apart to tell apart at all.
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
        "missing reported cost or registry pricing -- cannot derive billing from cost; "
        "review and set billing_relationship to 'inside' or 'separate' manually"
    ),
    "ambiguous": (
        "the separate-billing signal is within noise, or the surcharge matches neither "
        "hypothesis -- capture a longer reasoning turn (more reasoning_tokens) or review "
        "manually; do NOT assume 'inside'"
    ),
    "inside": (
        "reported cost ~= completion-priced cost -> reasoning is INSIDE completion_tokens "
        "(already billed); D1's breakdown is display-only, DO NOT add to the total"
    ),
    "separate": (
        "reported cost exceeds completion-priced by ~reasoning_tokens*output_rate -> "
        "reasoning billed SEPARATELY on this provider; D1 must add a provider-specific "
        "reasoning line"
    ),
}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        print("error: pass a <model_id> your probe marked reasoning-capable", file=sys.stderr)
        return 2
    model = sys.argv[1]
    provider_slug = sys.argv[2] if len(sys.argv) > 2 else None

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("error: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Reason step by step, then answer: a bat and ball cost $1.10; "
                       "the bat costs $1.00 more than the ball. How much is the ball?",
        }],
        "reasoning": {"effort": "high"},
        "usage": {"include": True},  # ask OpenRouter to report the billed cost
    }
    if provider_slug:
        # Documented OpenRouter provider-routing shape: pin the upstream and
        # forbid silent fallback, so the captured billing is attributable to a
        # known route and the capture is reproducible.
        payload["provider"] = {"order": [provider_slug], "allow_fallbacks": False}
    else:
        print(
            "warning: no provider pinned -- billing is provider-specific, so pass a "
            "[provider_slug] for a reproducible capture",
            file=sys.stderr,
        )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    resp = httpx.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()

    usage = data.get("usage", {})
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    prompt = usage.get("prompt_tokens", 0)
    reported_cost = usage.get("cost")  # OpenRouter's actual charge (USD) -- the authority
    provider_served = data.get("provider")  # route that actually served the call

    pricing = registry_pricing(model)
    input_rate, output_rate = pricing if pricing is not None else (None, None)
    completion_priced = None
    if pricing is not None:
        completion_priced = prompt / 1_000_000 * input_rate + completion / 1_000_000 * output_rate

    billing, detail = classify_billing(
        reasoning_tokens=reasoning,
        completion_tokens=completion,
        reported_cost=reported_cost,
        completion_priced_cost=completion_priced,
        output_rate=output_rate,
    )

    fixture = {
        "_placeholder": False,
        "model": model,
        "provider_requested": provider_slug,
        "provider_served": provider_served,
        "usage": usage,
        "cost": reported_cost,
        "price_evidence": {
            "registry_input_per_mtok": input_rate,
            "registry_output_per_mtok": output_rate,
            "completion_priced_cost": completion_priced,
            "surcharge": detail["surcharge"],
            "expected_separate_surcharge": detail["expected_separate_surcharge"],
            "noise": detail["noise"],
        },
        "billing_relationship": billing,
        "_conclusion": CONCLUSIONS[billing],
        "_note": (
            "billing_relationship comes from classify_billing() in "
            "scripts/capture_reasoning_usage_fixture.py -- the single source the D1 gate "
            "also imports. It is derived from OpenRouter's billed `cost` vs the "
            "registry-priced prompt+completion cost, never from token containment. "
            "The decision is provider-specific: provider_requested/provider_served record "
            "the route this evidence is valid for."
        ),
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=2), encoding="utf-8")

    print(f"model={model}  provider_requested={provider_slug}  provider_served={provider_served}")
    print(f"reasoning_tokens={reasoning}  completion_tokens={completion}")
    print(f"reported cost={reported_cost}  completion_priced={completion_priced}")
    print(f"surcharge={detail['surcharge']}  expected_separate={detail['expected_separate_surcharge']}  noise={detail['noise']}")
    print(f"billing_relationship={billing}")
    print(CONCLUSIONS[billing])
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
