"""Maintainer CLI: run the reasoning-capability probe sweep (v1.3.0 A2/A4 run).

PAID. Requires OPENROUTER_API_KEY and an EXPLICIT --max-probe-usd ceiling. Refuses
to run without a guaranteed ceiling (the authorization's spend limit is enforced
twice: --max-probe-usd is a required arg, and run_probe_sweep re-checks the
worst-case cost against it). Writes the capability sidecar; resumable (fresh rows
are skipped), bounded concurrency, per-model degradation.

Every probe call caps output at PROBE_MAX_TOKENS, so the worst-case per-call cost is
(pinned endpoint output price x PROBE_MAX_TOKENS + prompt). Bound that cost EITHER
way -- exactly one is required:

  --resolve-endpoint-prices  (PREFERRED) prices every call from its OWN pinned
      endpoint via the public, keyless, FREE /endpoints API before any paid call.
      Refuses if any pinned endpoint's price cannot be resolved, since a model with
      no sound bound would otherwise spend against an under-counted ceiling.
  --max-cost-per-call        one uniform maintainer-asserted bound. Sound but loose:
      it must cover the priciest model in the sweep, so a single expensive outlier
      inflates the required ceiling for every other call (on the current registry
      that is roughly an order of magnitude of unusable headroom).

Never derive a bound from the registry's model-wide price: --provider-tag pins a
specific endpoint whose price can differ, so a registry-derived bound could
UNDER-count and breach the ceiling.

Usage:
  OPENROUTER_API_KEY=... uv run python scripts/probe_reasoning_capabilities.py \
      --max-probe-usd 15.00 --resolve-endpoint-prices [--concurrency 4]

Provider routing (correction #6): each model is pinned to an exact endpoint tag.
By default the tag is the model id's provider prefix (e.g. openai/gpt-x -> openai);
pass --provider-tag to pin one tag for every model, or refine per-model routing by
extending resolve_provider_tag(). --resolve-endpoint-prices validates those pins as
a side effect: an unknown tag fails price resolution and names the available tags.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# scripts/ is not on sys.path as the repo root; add it so `backend` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import model_registry, reasoning_capability, reasoning_probe  # noqa: E402


def resolve_provider_tag(model_entry, override=None):
    """Exact endpoint tag to pin for this model (correction #6). Default: the
    provider prefix of the model id. `override` pins one tag for all models."""
    if override:
        return override
    mid = model_entry.get("id", "")
    return mid.split("/", 1)[0] if "/" in mid else mid


async def _run(args) -> int:
    registry = model_registry.load_model_registry()
    models = [m for m in registry.get("models", []) if m.get("id")]
    existing = reasoning_capability.load_capabilities()
    api_key = os.environ["OPENROUTER_API_KEY"]

    def provider_of(model_entry):
        return resolve_provider_tag(model_entry, args.provider_tag)

    per_model_bounds = None
    if args.resolve_endpoint_prices:
        # Only the models this sweep will actually call. /endpoints is public and
        # free, so nothing here spends against the ceiling it is computing.
        needing = reasoning_probe.models_needing_probe(models, existing, provider_of)
        per_model_bounds = reasoning_probe.resolve_probe_call_bounds(needing, provider_of)
        worst_case = reasoning_probe.estimate_max_probe_cost_per_model(
            [m["id"] for m in needing], reasoning_probe.CANDIDATE_LEVELS, per_model_bounds
        )
        print(
            f"resolved per-endpoint prices for {len(needing)} model(s) to probe; "
            f"worst-case sweep ${worst_case:.4f} vs authorized ${args.max_probe_usd:.4f}"
        )

    merged = await reasoning_probe.run_probe_sweep(
        models,
        provider_of,
        api_key,
        max_probe_usd=args.max_probe_usd,
        max_cost_per_call_usd=args.max_cost_per_call,
        per_model_bounds=per_model_bounds,
        existing=existing,
        concurrency=args.concurrency,
    )
    reasoning_capability.save_capabilities(merged.values())
    probed = sum(1 for r in merged.values() if r.get("probed"))
    print(f"probed/kept {probed} of {len(models)} models -> {reasoning_capability.SIDECAR_PATH}")
    # Make skipped rows visible: a wrong endpoint tag silently drops a model, so name
    # what was NOT probed rather than reporting only a reassuring count.
    skipped = [m["id"] for m in models if not merged.get(m["id"], {}).get("probed")]
    if skipped:
        print(f"NOT probed ({len(skipped)}): {', '.join(skipped)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the reasoning-capability probe sweep (PAID).")
    # Required: no ceiling -> no run. This is the CLI-level spend guard.
    parser.add_argument("--max-probe-usd", type=float, required=True,
                        help="Hard maximum authorized spend for this sweep (USD).")
    parser.add_argument("--max-cost-per-call", type=float, default=None,
                        help="ONE uniform worst-case USD cost for every probe call. Set to the "
                             "PINNED endpoint's output price x the probe's max output tokens "
                             "(+ prompt); the ceiling = calls x this. Sound but loose -- it must "
                             "cover the priciest model in the sweep. Prefer "
                             "--resolve-endpoint-prices. Exactly one of the two is required.")
    parser.add_argument("--resolve-endpoint-prices", action="store_true",
                        help="Price each call from its OWN pinned endpoint via the public, "
                             "keyless, free /endpoints API before any paid call (PREFERRED). "
                             "Refuses if any pinned endpoint's price cannot be resolved.")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Parallel probe calls (>= 1).")
    parser.add_argument("--provider-tag", default=None,
                        help="Pin one endpoint tag for every model (default: per-model id prefix).")
    args = parser.parse_args(argv)

    if args.max_probe_usd <= 0:
        print("error: --max-probe-usd must be positive", file=sys.stderr)
        return 2
    # Exactly one bounding method: two would be ambiguous about which ceiling the
    # run actually enforced, and zero leaves the paid sweep unbounded.
    if args.resolve_endpoint_prices and args.max_cost_per_call is not None:
        print("error: pass either --resolve-endpoint-prices or --max-cost-per-call, not both",
              file=sys.stderr)
        return 2
    if not args.resolve_endpoint_prices and args.max_cost_per_call is None:
        print("error: pass --resolve-endpoint-prices (preferred) or --max-cost-per-call "
              "to bound each probe call", file=sys.stderr)
        return 2
    if args.max_cost_per_call is not None and args.max_cost_per_call <= 0:
        print("error: --max-cost-per-call must be positive", file=sys.stderr)
        return 2
    if args.concurrency < 1:
        print("error: --concurrency must be >= 1", file=sys.stderr)
        return 2
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("error: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    try:
        return asyncio.run(_run(args))
    except reasoning_probe.CeilingError as exc:
        print(f"refused (spend ceiling): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
