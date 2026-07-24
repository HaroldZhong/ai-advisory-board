"""Active reasoning-capability probe (v1.3.0 A2).

Empirically determines a model's reasoning-control surface through OpenRouter and
produces a capability record for the A1 sidecar. Per-model, via a PINNED endpoint
tag (correction #6), using OpenRouter's NORMALIZED `reasoning.effort` interface
(correction #1): send an effort object, observe whether the model actually reasoned
and whether the level *differentiates* effort.

The PAID run is a maintainer CLI gated by the authorized spend ceiling; this
module's probe/classify core is transport-injectable (`transport=` / the module
`_probe_transport`, mirroring `openrouter_client`) so tests never hit the network.
No paid call happens on import or in tests.

Classification (brainstorm §2.4 steps 3–4):
  * reasoning signal, in reliability order: usage.reasoning_tokens > 0 (strongest);
    else message.reasoning/reasoning_details present; else no observed reasoning.
    (We do NOT infer reasoning from a billed-vs-visible token gap: without a real
    tokenizer, comparing billed tokens to visible words misreads verbose ordinary
    output as hidden reasoning -- a false positive the honesty guard must avoid.)
  * baseline (no effort) reasoning => native_default_on (for onoff models).
  * signal rising low->med->high => `levels` (varies_effort); flat & non-zero =>
    `onoff`; no signal anywhere => `none`. Honesty guard: a "supported" surface is
    only ever written from an OBSERVED signal, never from metadata.
"""
from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

import httpx

from . import config
from .reasoning_capability import model_fingerprint, unknown_record


class CeilingError(RuntimeError):
    """Refuse to probe: the authorized spend ceiling is unset or cannot be
    guaranteed for the requested sweep. This is the code embodiment of the
    PAID PROBE AUTHORIZATION -- no paid call happens past this guard."""

# Deliberately probe the three widely-supported efforts. minimal/xhigh are omitted
# because a provider that rejects an effort fails the whole model probe (all-or-
# nothing per model), and B2's snap maps an unprobed minimal/xhigh to the nearest
# verified level -- a safe degradation. classify_capability records exactly the
# levels probed, so the sidecar never claims an unverified effort.
CANDIDATE_LEVELS = ("low", "medium", "high")
PROBE_PROMPT = (
    "Reason step by step, then answer: a bat and ball cost $1.10; the bat costs "
    "$1.00 more than the ball. How much is the ball?"
)
# Hard per-call output bound so a paid probe call's cost is finite and the spend
# ceiling is enforceable (not just an assumption). High enough that the effort
# levels still differentiate by reasoning-token count for `levels` models.
PROBE_MAX_TOKENS = 8000
# Generous fixed size of PROBE_PROMPT for the per-call cost bound (tokens).
PROBE_PROMPT_TOKENS_EST = 64
# Same <think>/<thinking> markup the runtime (backend.openrouter.extract_reasoning)
# treats as reasoning for tags-mode models -- kept in sync so the probe classifies
# a tags-mode model the way runtime will actually read it.
_THINK_TAG_RE = re.compile(r"<(think|thinking)>([\s\S]*?)</\1>")

_probe_transport = None  # httpx.MockTransport injection point for tests; None = real net.


def reasoning_signal(response: Dict[str, Any], extraction_mode: Optional[str] = None) -> int:
    """Observed reasoning tokens for one response, in reliability order. Returns 0
    when no reasoning is observed (the honest "did not reason" reading). For
    tags-mode models (extraction_mode == "tags"), a visible <think>/<thinking>
    block counts as reasoning, mirroring runtime extraction."""
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    rt = details.get("reasoning_tokens")
    if isinstance(rt, (int, float)) and not isinstance(rt, bool):
        # The authoritative counter. An explicit 0 is a definitive "did not reason"
        # and must short-circuit -- never fall through to the weaker text heuristic.
        return int(rt) if rt > 0 else 0

    message = ((response.get("choices") or [{}])[0] or {}).get("message") or {}
    if message.get("reasoning") or message.get("reasoning_details"):
        # Text present but no token count -> reasoning happened, magnitude unknown.
        return 1

    # Tags-mode models return reasoning as visible <think> blocks; count them (gated
    # to tags mode exactly like the runtime, so a field-mode answer that merely
    # mentions the markup is not misread as reasoning).
    if extraction_mode == "tags" and _THINK_TAG_RE.search(message.get("content") or ""):
        return 1

    # No reliable reasoning signal. We deliberately do NOT infer reasoning from a
    # billed-vs-visible token gap: visible content would have to be tokenized to
    # compare, and any word/char estimate misclassifies verbose ordinary output as
    # hidden reasoning. The honest reading is "did not observe reasoning".
    return 0


def classify_capability(
    model_id: str,
    fingerprint: Optional[str],
    provider_tag: str,
    baseline_signal: int,
    level_signals: Dict[str, int],
    *,
    levels: Iterable[str] = CANDIDATE_LEVELS,
    probed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a capability record from the probe signals. Pure + fully testable.
    `levels` is the set of effort levels ACTUALLY probed (default the full ladder);
    classification and the recorded `levels` reflect only those, so a caller running
    a cheaper custom sweep is never credited with unprobed levels."""
    rec = unknown_record(model_id, fingerprint)
    rec["provider_pinned"] = provider_tag
    rec["probed"] = True
    rec["probed_at"] = probed_at or datetime.now(timezone.utc).isoformat()

    probed_levels = list(levels)
    ordered = [level_signals.get(level, 0) for level in probed_levels]
    reasoned = baseline_signal > 0 or any(s > 0 for s in ordered)

    # Honesty guard: only observed reasoning yields a "supported" surface.
    rec["supports_reasoning"] = bool(reasoned)
    rec["native_default_on"] = baseline_signal > 0
    # `plain` is the no-effort (plain) request's behavior -> baseline only, NOT any
    # effort-only signal (an effort-only model does not reason on a plain request).
    rec["plain"] = "reasoned" if baseline_signal > 0 else "none"

    if not reasoned:
        rec["control_surface"] = "none"
        return rec

    non_decreasing = all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1))
    differentiates = len(ordered) >= 2 and non_decreasing and ordered[0] < ordered[-1]
    rec["varies_effort"] = bool(differentiates)

    if differentiates:
        rec["control_surface"] = "levels"
        # Only advertise levels that ACTUALLY produced reasoning: a zero-signal
        # effort was proven ineffective, so recording it would let snap/selection
        # route users to a no-reasoning setting instead of the nearest verified one.
        rec["levels"] = [lvl for lvl, sig in zip(probed_levels, ordered) if sig > 0]
    else:
        # Reasons but the level does not move effort -> only on/off is real.
        rec["control_surface"] = "onoff"
    return rec


async def _post(model_id, provider_tag, api_key, effort, transport):
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        # Pin the exact endpoint tag; billing + support are provider-specific (#6).
        "provider": {"order": [provider_tag], "allow_fallbacks": False},
        # Bound completion (incl. reasoning) tokens so per-call cost is finite and
        # the sweep's spend ceiling is enforceable rather than assumed.
        "max_tokens": PROBE_MAX_TOKENS,
    }
    if effort is not None:
        # OpenRouter's normalized reasoning interface (correction #1).
        payload["reasoning"] = {"effort": effort}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Metadata": "enabled",
    }
    async with httpx.AsyncClient(timeout=120.0, transport=transport) as client:
        # Respect the app's configured OpenRouter endpoint (relay/proxy) instead of
        # a hard-coded openrouter.ai URL (backend.config owns the base URL).
        resp = await client.post(config.OPENROUTER_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


async def probe_model(
    model_entry: Dict[str, Any],
    provider_tag: str,
    api_key: str,
    *,
    levels: Iterable[str] = CANDIDATE_LEVELS,
    transport=None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Probe one model end-to-end and return its capability record. One baseline
    call (no effort) + one call per candidate level, through the injected transport.

    PAID when transport is None (real network) -- callers gate that on the ceiling.
    """
    model_id = model_entry["id"]
    tx = transport if transport is not None else _probe_transport
    levels = tuple(levels)
    extraction = model_entry.get("reasoning_extraction")

    baseline = reasoning_signal(await _post(model_id, provider_tag, api_key, None, tx), extraction)
    level_signals: Dict[str, int] = {}
    for level in levels:
        level_signals[level] = reasoning_signal(
            await _post(model_id, provider_tag, api_key, level, tx), extraction
        )

    probed_at = (now or datetime.now(timezone.utc)).isoformat()
    return classify_capability(
        model_id, model_fingerprint(model_entry), provider_tag,
        baseline, level_signals, levels=levels, probed_at=probed_at,
    )


# --- A4: populate & cache the matrix (resumable, ceiling-guarded) -------------

def models_needing_probe(
    registry_models: Iterable[Dict[str, Any]],
    existing: Dict[str, Dict[str, Any]],
    provider_of: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> List[Dict[str, Any]]:
    """Resumable set: skip models already probed whose stored fingerprint still
    matches the registry entry (fresh); return the rest (unprobed or stale). When
    `provider_of` is given, a fresh row whose `provider_pinned` differs from the
    now-requested endpoint tag is ALSO re-probed -- probe results are
    provider-specific, so a --provider-tag change must invalidate the old row."""
    needing: List[Dict[str, Any]] = []
    for m in registry_models:
        mid = m.get("id")
        if not mid:
            continue
        rec = existing.get(mid)
        fresh = bool(rec and rec.get("probed") and rec.get("fingerprint") == model_fingerprint(m))
        if fresh and provider_of is not None and rec.get("provider_pinned") != provider_of(m):
            fresh = False  # a different endpoint was requested -> re-probe
        if fresh:
            continue
        needing.append(m)
    return needing


def estimate_max_probe_cost(num_models: int, levels: Iterable[str], max_cost_per_call_usd: float) -> float:
    """Upper bound on sweep spend: num_models x (1 baseline + one call per level) x
    the maintainer-asserted worst-case per-call cost. Because every probe call caps
    output at PROBE_MAX_TOKENS, the maintainer sets max_cost_per_call_usd to (the
    PINNED endpoint's output price x PROBE_MAX_TOKENS + prompt) and this is a real
    upper bound. We deliberately do NOT derive it from the registry's model-wide
    `pricing`: --provider-tag pins a specific endpoint whose price can differ from
    the curated model-wide value, so a registry-derived bound could under-count and
    break the ceiling (exact per-endpoint price auto-resolution via /endpoints is a
    documented follow-up)."""
    calls_per_model = 1 + len(tuple(levels))
    return num_models * calls_per_model * max_cost_per_call_usd


async def run_probe_sweep(
    registry_models: List[Dict[str, Any]],
    provider_of: Callable[[Dict[str, Any]], str],
    api_key: str,
    *,
    max_probe_usd: Optional[float],
    max_cost_per_call_usd: Optional[float],
    existing: Optional[Dict[str, Dict[str, Any]]] = None,
    levels: Iterable[str] = CANDIDATE_LEVELS,
    transport=None,
    concurrency: int = 4,
    now: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    """Probe every model that needs it, resumably, and return the merged records.

    SPEND-CEILING GUARD (PAID PROBE AUTHORIZATION): before any paid call, refuse if
    the authorized ceiling or the per-call bound is unset/non-positive/non-finite,
    or if the worst-case sweep cost (num calls x the asserted per-call bound) exceeds
    the ceiling. One model erroring degrades to a skipped row, but a SYSTEMIC failure
    (every call fails) aborts loudly rather than publishing an all-skipped matrix.
    Bounded concurrency; the caller persists via reasoning_capability.save_capabilities.
    """
    levels = tuple(levels)
    merged = dict(existing or {})
    needing = models_needing_probe(registry_models, merged, provider_of)

    def _positive_finite(value):
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value > 0)

    if max_probe_usd is None:
        raise CeilingError(
            "no authorized probe spend ceiling (<MAX_PROBE_USD> is unset) -- refusing to probe"
        )
    # `not (x > 0)` alone would miss NaN slipping through the ceiling comparison.
    if not _positive_finite(max_probe_usd):
        raise CeilingError(
            f"authorized probe ceiling {max_probe_usd!r} must be a positive, finite number -- refusing to probe"
        )
    if not _positive_finite(max_cost_per_call_usd):
        raise CeilingError(
            f"per-call cost bound {max_cost_per_call_usd!r} must be a positive, finite number "
            f"(the pinned endpoint's output price x {PROBE_MAX_TOKENS} max tokens) -- refusing to probe"
        )
    if concurrency < 1:
        # asyncio.Semaphore(0) starts locked -> every task blocks forever.
        raise ValueError(f"concurrency must be >= 1 (got {concurrency})")

    worst_case = estimate_max_probe_cost(len(needing), levels, max_cost_per_call_usd)
    if worst_case > max_probe_usd:
        raise CeilingError(
            f"worst-case probe spend ${worst_case:.4f} ({len(needing)} models x "
            f"{1 + len(levels)} calls x ${max_cost_per_call_usd}/call) exceeds the authorized "
            f"ceiling ${max_probe_usd:.4f} -- refusing to probe"
        )

    sem = asyncio.Semaphore(concurrency)

    async def _one(model_entry: Dict[str, Any]):
        async with sem:
            try:
                rec = await probe_model(
                    model_entry, provider_of(model_entry), api_key,
                    levels=levels, transport=transport, now=now,
                )
                return model_entry["id"], rec
            except Exception:
                # Degrade per-model: a single failure must not abort the matrix.
                return model_entry["id"], None

    by_id = {m["id"]: m for m in needing}
    results = await asyncio.gather(*[_one(m) for m in needing])
    succeeded = 0
    for mid, rec in results:
        if rec is not None:
            merged[mid] = rec
            succeeded += 1
        elif mid in merged:
            # A REQUIRED re-probe (stale fingerprint or changed provider) failed, so
            # the old record is no longer trustworthy -- runtime freshness only checks
            # the fingerprint, so a kept stale-provider row would be used silently.
            # Replace it with unknown rather than leaving the invalid record in place.
            merged[mid] = unknown_record(mid, model_fingerprint(by_id[mid]))
    if needing and succeeded == 0:
        # Every probe failed -> systemic (bad key / exhausted quota / wrong endpoint
        # tags for all). Don't let the CLI save an all-skipped sidecar and exit 0.
        raise RuntimeError(
            f"probe attempted {len(needing)} models but every call failed -- likely a systemic "
            f"error (invalid key, exhausted quota, or wrong endpoint tags); sidecar not updated"
        )
    return merged
