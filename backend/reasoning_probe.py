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
from typing import Any, Dict, Iterable, List, Optional

import httpx

from . import config
from .endpoint_pricing import (
    EndpointPricingError,
    fetch_endpoints,
    normalize_provider_name,
    price_endpoint,
    resolve_served_endpoint_tag,
)
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
_endpoints_get = None    # /endpoints fetcher injection point for tests; None = real net.


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
        # Bound completion (incl. reasoning) tokens so per-call cost is finite and
        # the sweep's spend ceiling is enforceable rather than assumed.
        "max_tokens": PROBE_MAX_TOKENS,
    }
    if provider_tag:
        # Pinning is OPT-IN. The sweep probes UNPINNED so the capability is learned
        # on the endpoint OpenRouter actually routes to -- the same endpoint real
        # traffic gets -- rather than one nobody chose. A guessed pin is worse than
        # none: `allow_fallbacks: False` turns a wrong tag into no route at all.
        payload["provider"] = {"order": [provider_tag], "allow_fallbacks": False}
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


def _differs(observed: Optional[str], expected: Optional[str]) -> bool:
    """True only when BOTH names are known and they name different providers. An
    absent name is 'unknown', not 'different' -- metadata can be missing per call and
    that must not be misread as the router moving."""
    if not observed or not expected:
        return False
    return normalize_provider_name(observed) != normalize_provider_name(expected)


def served_provider_name(response: Dict[str, Any]) -> Optional[str]:
    """DISPLAY NAME of the endpoint that actually served this response, from the
    router metadata the probe opts into (`X-OpenRouter-Metadata: enabled`).

    The metadata exposes provider / model / selected only -- there is NO endpoint tag
    in it -- so this is a name like "Google Vertex", not a slug. Callers must map it
    back to an exact tag (endpoint_pricing.resolve_served_endpoint_tag) and accept
    that the mapping can be ambiguous."""
    metadata = response.get("openrouter_metadata") or {}
    available = (metadata.get("endpoints") or {}).get("available") or []
    for entry in available:
        if entry.get("selected"):
            return entry.get("provider")
    return None


async def probe_model(
    model_entry: Dict[str, Any],
    api_key: str,
    *,
    levels: Iterable[str] = CANDIDATE_LEVELS,
    transport=None,
    now: Optional[datetime] = None,
    get=None,
) -> Dict[str, Any]:
    """Probe one model end-to-end and return its capability record. One baseline
    call (no effort) + one call per candidate level, through the injected transport.

    Probes UNPINNED and records the endpoint that ACTUALLY SERVED, rather than
    pinning a tag chosen up front. The recorded `provider_pinned` becomes the pin
    runtime applies to real traffic, so it must name an endpoint someone actually
    routes to -- not a guess. (The old default, the model id's provider prefix, named
    a non-existent endpoint for 17 of 33 registry models.)

    The exact tag is resolved from the served provider's DISPLAY NAME via the free
    /endpoints data. When that name maps to several tags that price differently, no
    exact endpoint can be determined and the record carries NO pin: runtime then
    routes normally, exactly as it does today, instead of being forced somewhere
    nobody chose.

    PAID when transport is None (real network) -- callers gate that on the ceiling.
    """
    model_id = model_entry["id"]
    tx = transport if transport is not None else _probe_transport
    levels = tuple(levels)
    extraction = model_entry.get("reasoning_extraction")

    # 1) Baseline goes out UNPINNED so OpenRouter picks the endpoint it would pick
    #    for real traffic.
    baseline_response = await _post(model_id, None, api_key, None, tx)
    baseline = reasoning_signal(baseline_response, extraction)
    served = served_provider_name(baseline_response)
    if not served:
        # No baseline provider -> no anchor to attribute ANY of this sweep to. Without
        # it nothing can be pinned, and the mixed-endpoint check below is blind too
        # (an unknown name is deliberately not "different"), so a `probed: true`
        # record here would assert a capability for whichever endpoint happens to
        # serve later. Refuse instead; a later sweep can retry once router metadata
        # is available.
        return unknown_record(model_id, model_fingerprint(model_entry))

    # 2) Resolve that endpoint EXACTLY and pin the remaining calls to it. Every level
    #    must be measured on the SAME endpoint as the one we are about to record: a
    #    capability observed on provider B but recorded against provider A would make
    #    runtime send A a reasoning shape that was never verified there. This pin is
    #    an OBSERVED tag (it just served), not a guess -- the failure mode that made
    #    the old up-front pin unusable does not apply.
    try:
        getter = get if get is not None else _endpoints_get
        kwargs = {} if getter is None else {"get": getter}
        # resolve_served_endpoint_tag is SYNCHRONOUS (httpx.get, 30s timeout). Called
        # directly it would block the event loop for the whole sweep, serialising
        # every concurrent probe behind one slow /endpoints response.
        provider_tag = await asyncio.to_thread(
            resolve_served_endpoint_tag, model_id, served, **kwargs
        )
    except Exception:
        # Best-effort enrichment only. The paid calls have already succeeded, so no
        # lookup failure -- HTTP, malformed JSON, unexpected shape -- may discard an
        # observed capability. Degrade to no pin.
        provider_tag = None

    level_signals: Dict[str, int] = {}
    mixed_endpoints = False
    for level in levels:
        response = await _post(model_id, provider_tag, api_key, level, tx)
        if provider_tag is None and _differs(served_provider_name(response), served):
            # Unpinnable AND the router moved between calls: these signals cannot be
            # attributed to one endpoint.
            mixed_endpoints = True
        level_signals[level] = reasoning_signal(response, extraction)

    if mixed_endpoints:
        # Publishing a surface here would claim a capability for endpoints that were
        # never measured together. Record UNKNOWN and leave it un-probed so a later
        # sweep can retry, rather than fabricating an attribution.
        return unknown_record(model_id, model_fingerprint(model_entry))

    probed_at = (now or datetime.now(timezone.utc)).isoformat()
    return classify_capability(
        model_id, model_fingerprint(model_entry), provider_tag,
        baseline, level_signals, levels=levels, probed_at=probed_at,
    )


# --- A4: populate & cache the matrix (resumable, ceiling-guarded) -------------

def models_needing_probe(
    registry_models: Iterable[Dict[str, Any]],
    existing: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resumable set: skip models already probed whose stored fingerprint still
    matches the registry entry (fresh); return the rest (unprobed or stale).

    There is no provider-change invalidation any more: `provider_pinned` is now an
    OBSERVATION of the endpoint that served the probe, not a tag requested up front,
    so there is no requested value to compare it against. Fingerprint changes (and
    the 30-day staleness warning) remain the invalidation signals."""
    needing: List[Dict[str, Any]] = []
    for m in registry_models:
        mid = m.get("id")
        if not mid:
            continue
        rec = existing.get(mid)
        fresh = bool(rec and rec.get("probed") and rec.get("fingerprint") == model_fingerprint(m))
        if fresh:
            continue
        needing.append(m)
    return needing


def estimate_max_probe_cost(num_models: int, levels: Iterable[str], max_cost_per_call_usd: float) -> float:
    """Upper bound on sweep spend using ONE uniform per-call bound: num_models x
    (1 baseline + one call per level) x the maintainer-asserted worst-case per-call
    cost. Because every probe call caps output at PROBE_MAX_TOKENS, the maintainer
    sets max_cost_per_call_usd to (the PINNED endpoint's output price x
    PROBE_MAX_TOKENS + prompt) and this is a real upper bound.

    This bound is SOUND but LOOSE: one uniform figure must cover the priciest model
    in the sweep, so a single expensive outlier inflates the ceiling for every other
    call. Prefer `estimate_max_probe_cost_per_model` with bounds resolved by
    `resolve_probe_call_bounds`, which prices each call from its OWN pinned endpoint.

    Never derive the uniform bound from the registry's model-wide `pricing`:
    --provider-tag pins a specific endpoint whose price can differ from the curated
    model-wide value, so a registry-derived bound could UNDER-count and breach the
    ceiling. Resolved endpoint prices (below) are the sound alternative."""
    calls_per_model = 1 + len(tuple(levels))
    return num_models * calls_per_model * max_cost_per_call_usd


def probe_call_bound_usd(pricing: Dict[str, Any]) -> float:
    """Worst-case USD cost of ONE probe call on the endpoint `pricing` describes.

    Every probe call caps output at PROBE_MAX_TOKENS, so the cap bounds the billable
    output. Rates are USD PER TOKEN, as `/endpoints` publishes them.

    A non-zero `internal_reasoning` rate is an EXPLICIT SEPARATE charge for reasoning
    tokens -- that is the D1 billing rule (see classify_billing: an explicit
    internal_reasoning price short-circuits straight to "separate"). The probe
    deliberately provokes reasoning, so on such an endpoint the completion-only
    figure is NOT an upper bound and the ceiling could be breached.

    The worst case charges the capped tokens at BOTH rates: every token billed as
    completion and surcharged as reasoning. That is deliberately conservative -- it
    stays a true upper bound whether the provider counts reasoning inside the
    completion cap or bills it on top.

    When the endpoint publishes NO internal_reasoning rate, the surcharge is assumed
    to equal the completion rate rather than zero. An absent field is not evidence
    that reasoning is free: classify_billing detects separate reasoning billing on
    endpoints publishing no such rate, by observing a surcharge of
    (reasoning_tokens x completion_rate) ON TOP of the completion-priced cost.

    Also counted, because endpoints bill more than the two token rates:
      * FLAT per-request fees (`per_request_usd`). Non-zero on search-native models:
        perplexity/sonar publishes 0.005/search and searches on every completion, so
        omitting it under-counts those calls by more than half.
      * The prompt is priced at the WORSE of the prompt and cache-WRITE rates, since
        a provider that auto-caches bills the prompt as a cache write.

    Refuses when the endpoint publishes a non-zero rate this module cannot map to a
    probe call: silently dropping an unknown charge fails OPEN, and a ceiling is only
    a guarantee if every published charge is either counted or explicitly ruled out.

    ASSUMPTION (documented, not enforceable here): the provider honours `max_tokens`.
    _post sends it on every call; a provider that ignored it could bill unbounded
    output, which no client-side arithmetic can bound."""
    unaccounted = pricing.get("unaccounted_nonzero_price_keys") or []
    if unaccounted:
        raise EndpointPricingError(
            f"endpoint publishes non-zero charges this bound does not model "
            f"({', '.join(unaccounted)}); counting only the known components would "
            f"under-count the ceiling -- refusing to price this call"
        )
    completion_rate = float(pricing["completion_per_token"])
    published_reasoning = pricing.get("internal_reasoning_per_token")
    if published_reasoning not in (None, "") and float(published_reasoning) > 0:
        # Explicit rate -> use exactly what this endpoint says it charges.
        reasoning_rate = float(published_reasoning)
    else:
        # NOT published is not evidence that it is free. classify_billing detects
        # separate reasoning billing on endpoints that publish NO internal_reasoning
        # rate, by observing a surcharge of reasoning_tokens x completion_rate on top
        # of the completion-priced cost. Assuming zero there would make the ceiling a
        # guess about the favourable regime; assume that documented surcharge instead.
        reasoning_rate = completion_rate
    prompt_rate = max(
        float(pricing["prompt_per_token"]),
        float(pricing.get("input_cache_write_per_token") or 0.0),
    )
    return (
        (completion_rate + reasoning_rate) * PROBE_MAX_TOKENS
        + prompt_rate * PROBE_PROMPT_TOKENS_EST
        + float(pricing.get("per_request_usd") or 0.0)
    )


def resolve_probe_call_bounds(
    model_entries: List[Dict[str, Any]],
    *,
    get=None,
) -> Dict[str, float]:
    """Per-model worst-case cost of ONE probe call, from the public (free, keyless)
    /endpoints API.

    The sweep probes UNPINNED, so ANY of a model's endpoints may serve the call. The
    bound is therefore the WORST across all of them -- the only figure that stays an
    upper bound whichever one OpenRouter picks. It is still far tighter than one
    uniform bound across the whole registry, which the priciest single model would
    otherwise dictate for every call.

    Refuses loudly rather than guessing: a model with no priceable endpoint has NO
    sound bound, and silently dropping it from the ceiling math would under-count the
    authorized spend. The error names every unresolved model and why.

    OpenRouter-only: `/endpoints`, endpoint tags and per-endpoint pricing are
    OpenRouter concepts. Against a generic openai-compatible relay there is nothing
    to resolve, and pricing the wrong service would produce a bound that does not
    describe what is actually billed -- so refuse and let the caller assert a
    uniform bound instead."""
    if not config.provider_is_openrouter():
        raise CeilingError(
            "per-endpoint price resolution needs OpenRouter (LLM_PROVIDER_KIND="
            f"{config.PROVIDER_KIND!r}): /endpoints, endpoint tags and per-endpoint "
            "pricing do not exist on a generic openai-compatible relay, and prices "
            "fetched elsewhere would not describe what this provider bills. Assert a "
            "uniform per-call bound instead -- refusing to probe"
        )
    kwargs = {} if get is None else {"get": get}
    bounds: Dict[str, float] = {}
    unresolved: List[str] = []
    for entry in model_entries:
        model_id = entry.get("id")
        try:
            url, endpoints = fetch_endpoints(model_id, **kwargs)
            if not endpoints:
                unresolved.append(f"{model_id}: publishes no endpoints")
                continue
            # EVERY endpoint must price, not just the priciest that happens to parse:
            # an endpoint we cannot price is one that could serve the call for an
            # unknown amount, so the max over the rest would not bound it. Validate
            # each bound BEFORE max(): float("NaN") parses fine and yields a NaN
            # bound, and every comparison against NaN is False, so max() would
            # silently keep an earlier finite value and the final check would pass a
            # model with one unbounded routable endpoint.
            per_endpoint = []
            for endpoint in endpoints:
                candidate = probe_call_bound_usd(price_endpoint(url, endpoint))
                if not (isinstance(candidate, float) and math.isfinite(candidate) and candidate > 0):
                    raise EndpointPricingError(
                        f"endpoint {endpoint.get('tag')!r} priced to a non-positive/non-finite "
                        f"bound {candidate!r}; it could still serve this call, so no maximum "
                        f"over the others would bound it"
                    )
                per_endpoint.append(candidate)
            bound = max(per_endpoint)
        except (EndpointPricingError, httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            unresolved.append(f"{model_id}: {exc}")
            continue
        if not (isinstance(bound, float) and math.isfinite(bound) and bound > 0):
            unresolved.append(f"{model_id}: resolved a non-positive/non-finite bound {bound!r}")
            continue
        bounds[model_id] = bound
    if unresolved:
        raise CeilingError(
            f"could not resolve a per-call price bound for {len(unresolved)} model(s), so the "
            f"sweep ceiling cannot be guaranteed -- refusing to probe:\n  "
            + "\n  ".join(unresolved)
        )
    return bounds


def estimate_max_probe_cost_per_model(
    model_ids: Iterable[str], levels: Iterable[str], per_model_bounds: Dict[str, float]
) -> float:
    """Upper bound on sweep spend when each model is priced by its own endpoint:
    sum over models of (1 baseline + one call per level) x that model's per-call
    bound. Raises KeyError for a model with no bound -- the caller must not be able
    to silently omit a model from the ceiling it is about to spend against."""
    calls_per_model = 1 + len(tuple(levels))
    return sum(per_model_bounds[mid] for mid in model_ids) * calls_per_model


async def run_probe_sweep(
    registry_models: List[Dict[str, Any]],
    api_key: str,
    *,
    max_probe_usd: Optional[float],
    max_cost_per_call_usd: Optional[float] = None,
    per_model_bounds: Optional[Dict[str, float]] = None,
    existing: Optional[Dict[str, Dict[str, Any]]] = None,
    levels: Iterable[str] = CANDIDATE_LEVELS,
    transport=None,
    concurrency: int = 4,
    now: Optional[datetime] = None,
    get=None,
) -> Dict[str, Dict[str, Any]]:
    """Probe every model that needs it, resumably, and return the merged records.

    SPEND-CEILING GUARD (PAID PROBE AUTHORIZATION): before any paid call, refuse if
    the authorized ceiling is unset/non-positive/non-finite, if no sound per-call
    bound is available, or if the worst-case sweep cost exceeds the ceiling.

    Two ways to bound a call, exactly one required:
      * `per_model_bounds` (PREFERRED) -- each model priced by its OWN pinned
        endpoint (see resolve_probe_call_bounds). Every model about to be probed
        must have a positive finite bound; a missing one refuses rather than
        omitting that model's spend from the ceiling.
      * `max_cost_per_call_usd` -- one uniform maintainer-asserted bound. Sound but
        loose, since it must cover the priciest model in the sweep.

    One model erroring degrades to a skipped row, but a SYSTEMIC failure (every call
    fails) aborts loudly rather than publishing an all-skipped matrix. Bounded
    concurrency; the caller persists via reasoning_capability.save_capabilities.
    """
    levels = tuple(levels)
    merged = dict(existing or {})
    needing = models_needing_probe(registry_models, merged)

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
    if per_model_bounds is None and not _positive_finite(max_cost_per_call_usd):
        raise CeilingError(
            f"per-call cost bound {max_cost_per_call_usd!r} must be a positive, finite number "
            f"(the pinned endpoint's output price x {PROBE_MAX_TOKENS} max tokens), or pass "
            f"per_model_bounds -- refusing to probe"
        )
    if concurrency < 1:
        # asyncio.Semaphore(0) starts locked -> every task blocks forever.
        raise ValueError(f"concurrency must be >= 1 (got {concurrency})")

    if per_model_bounds is not None:
        # A model with no sound bound must NOT be quietly dropped from the ceiling
        # math -- it would still be probed, spending against an under-counted total.
        missing = [
            m["id"] for m in needing
            if not _positive_finite(per_model_bounds.get(m["id"]))
        ]
        if missing:
            raise CeilingError(
                f"no positive, finite per-call bound for {len(missing)} model(s) about to be "
                f"probed ({', '.join(sorted(missing))}) -- refusing to probe"
            )
        worst_case = estimate_max_probe_cost_per_model(
            [m["id"] for m in needing], levels, per_model_bounds
        )
        basis = f"{len(needing)} models x {1 + len(levels)} calls, each priced by its own endpoint"
    else:
        worst_case = estimate_max_probe_cost(len(needing), levels, max_cost_per_call_usd)
        basis = (
            f"{len(needing)} models x {1 + len(levels)} calls x ${max_cost_per_call_usd}/call"
        )
    if worst_case > max_probe_usd:
        raise CeilingError(
            f"worst-case probe spend ${worst_case:.4f} ({basis}) exceeds the authorized "
            f"ceiling ${max_probe_usd:.4f} -- refusing to probe"
        )

    sem = asyncio.Semaphore(concurrency)

    async def _one(model_entry: Dict[str, Any]):
        async with sem:
            try:
                rec = await probe_model(
                    model_entry, api_key,
                    levels=levels, transport=transport, now=now, get=get,
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
            # Only a record that actually carries a probed capability counts. An
            # `unknown` row means the calls happened but could not be attributed --
            # if the router-metadata opt-in is not honoured for the account EVERY
            # model returns one, and counting those as successes would let the
            # systemic guard pass and publish an all-unknown sidecar with exit 0.
            if rec.get("probed"):
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
            f"probe attempted {len(needing)} models and none produced a usable capability -- "
            f"likely a systemic error (invalid key, exhausted quota, or router metadata not "
            f"enabled for this account, which leaves every result unattributable); "
            f"sidecar not updated"
        )
    return merged
