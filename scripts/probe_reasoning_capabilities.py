"""Maintainer CLI: run the reasoning-capability probe sweep (v1.3.0 A2/A4 run).

PAID. Requires OPENROUTER_API_KEY and an EXPLICIT --max-probe-usd ceiling. Refuses
to run without a guaranteed ceiling (the authorization's spend limit is enforced
twice: --max-probe-usd is a required arg, and run_probe_sweep re-checks the
worst-case cost against it). Writes the capability sidecar; resumable (fresh rows
are skipped), bounded concurrency, per-model degradation.

Every probe call caps output at PROBE_MAX_TOKENS, so the worst-case per-call cost is
(pinned endpoint output price x PROBE_MAX_TOKENS + prompt). The maintainer asserts
that via the REQUIRED --max-cost-per-call, and the ceiling = calls x that bound. We
do not derive it from the registry's model-wide price, because --provider-tag pins a
specific endpoint whose price can differ (auto-resolving the exact endpoint price via
the /endpoints API is a documented follow-up).

Usage:
  OPENROUTER_API_KEY=... uv run python scripts/probe_reasoning_capabilities.py \
      --max-probe-usd 5.00 --max-cost-per-call 0.05 [--concurrency 4]

Provider routing (correction #6): each model is pinned to an exact endpoint tag.
By default the tag is the model id's provider prefix (e.g. openai/gpt-x -> openai);
pass --provider-tag to pin one tag for every model, or refine per-model routing by
extending resolve_provider_tag(). Endpoint-tag resolution via the /endpoints API
(as in the D1 capture) is the follow-up hardening.
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

    merged = await reasoning_probe.run_probe_sweep(
        models,
        lambda m: resolve_provider_tag(m, args.provider_tag),
        api_key,
        max_probe_usd=args.max_probe_usd,
        max_cost_per_call_usd=args.max_cost_per_call,
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
    parser.add_argument("--max-cost-per-call", type=float, required=True,
                        help="Worst-case USD cost of ONE probe call. Set to the PINNED "
                             "endpoint's output price x the probe's max output tokens (+ prompt); "
                             "the ceiling = calls x this. Required so the bound reflects the "
                             "endpoint you route to, not the registry's model-wide price.")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Parallel probe calls (>= 1).")
    parser.add_argument("--provider-tag", default=None,
                        help="Pin one endpoint tag for every model (default: per-model id prefix).")
    args = parser.parse_args(argv)

    if args.max_probe_usd <= 0:
        print("error: --max-probe-usd must be positive", file=sys.stderr)
        return 2
    if args.max_cost_per_call <= 0:
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
