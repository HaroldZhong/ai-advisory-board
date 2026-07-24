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
    else message.reasoning/reasoning_details present; else billed-vs-visible gap.
  * baseline (no effort) reasoning => native_default_on (for onoff models).
  * signal rising low->med->high => `levels` (varies_effort); flat & non-zero =>
    `onoff`; no signal anywhere => `none`. Honesty guard: a "supported" surface is
    only ever written from an OBSERVED signal, never from metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import httpx

from .reasoning_capability import model_fingerprint, unknown_record

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
CANDIDATE_LEVELS = ("low", "medium", "high")
PROBE_PROMPT = (
    "Reason step by step, then answer: a bat and ball cost $1.10; the bat costs "
    "$1.00 more than the ball. How much is the ball?"
)
# A billed-vs-visible gap only counts as hidden reasoning past this many tokens,
# so ordinary output-length noise is not mistaken for reasoning.
_GAP_THRESHOLD = 50

_probe_transport = None  # httpx.MockTransport injection point for tests; None = real net.


def reasoning_signal(response: Dict[str, Any]) -> int:
    """Observed reasoning tokens for one response, in reliability order. Returns 0
    when no reasoning is observed (the honest "did not reason" reading)."""
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    rt = details.get("reasoning_tokens")
    if isinstance(rt, (int, float)) and not isinstance(rt, bool) and rt > 0:
        return int(rt)

    message = ((response.get("choices") or [{}])[0] or {}).get("message") or {}
    if message.get("reasoning") or message.get("reasoning_details"):
        # Text present but no token count -> reasoning happened, magnitude unknown.
        return 1

    # Billed-vs-visible gap (Horvat): completion tokens far beyond visible content.
    completion = usage.get("completion_tokens")
    content = message.get("content") or ""
    visible = len(content.split())
    if isinstance(completion, (int, float)) and (completion - visible) > _GAP_THRESHOLD:
        return int(completion - visible)
    return 0


def classify_capability(
    model_id: str,
    fingerprint: Optional[str],
    provider_tag: str,
    baseline_signal: int,
    level_signals: Dict[str, int],
    *,
    probed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a capability record from the probe signals. Pure + fully testable."""
    rec = unknown_record(model_id, fingerprint)
    rec["provider_pinned"] = provider_tag
    rec["probed"] = True
    rec["probed_at"] = probed_at or datetime.now(timezone.utc).isoformat()

    ordered = [level_signals.get(level, 0) for level in CANDIDATE_LEVELS]
    reasoned = baseline_signal > 0 or any(s > 0 for s in ordered)

    # Honesty guard: only observed reasoning yields a "supported" surface.
    rec["supports_reasoning"] = bool(reasoned)
    rec["native_default_on"] = baseline_signal > 0
    rec["plain"] = "reasoned" if reasoned else "none"

    if not reasoned:
        rec["control_surface"] = "none"
        return rec

    non_decreasing = all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1))
    differentiates = non_decreasing and ordered[0] < ordered[-1]
    rec["varies_effort"] = bool(differentiates)

    if differentiates:
        rec["control_surface"] = "levels"
        rec["levels"] = list(CANDIDATE_LEVELS)
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
        resp = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
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

    baseline = reasoning_signal(await _post(model_id, provider_tag, api_key, None, tx))
    level_signals: Dict[str, int] = {}
    for level in levels:
        level_signals[level] = reasoning_signal(
            await _post(model_id, provider_tag, api_key, level, tx)
        )

    probed_at = (now or datetime.now(timezone.utc)).isoformat()
    return classify_capability(
        model_id, model_fingerprint(model_entry), provider_tag,
        baseline, level_signals, probed_at=probed_at,
    )
