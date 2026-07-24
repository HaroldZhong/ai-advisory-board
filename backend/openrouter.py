"""OpenRouter API client for making LLM requests."""

import asyncio
import httpx
from typing import AsyncIterator, List, Dict, Any, Optional, Tuple
from .config import OPENROUTER_API_URL, get_openrouter_api_key, provider_is_openrouter
from .logger import logger


def connect_timeout_for(total_timeout: float) -> float:
    """Connect deadline strictly below the wall-clock timeout, so a blocked
    network raises ConnectTimeout (kind=network) before asyncio.wait_for
    cancels the request (kind=timeout). Some callers pass timeout=10.0."""
    return min(10.0, total_timeout / 2)


def classify_openrouter_error(exc: Exception) -> str:
    """Map an exception from an OpenRouter call to a coarse failure kind."""
    # NetworkError covers Connect/Read/Write/CloseError; ProxyError is a
    # separate TransportError subclass (failed tunnel = network problem too).
    if isinstance(exc, (httpx.NetworkError, httpx.ProxyError, httpx.ConnectTimeout)):
        return "network"
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, asyncio.TimeoutError)):
        return "timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 401:
            return "auth"
        # 403 is NOT auth: OpenRouter documents it as moderation/guardrail
        # blocks (input flagged), so it must not point operators at key fixes.
        if code == 403:
            return "other"
        if code == 402:
            return "quota"
        if code == 408:  # OpenRouter documents 408 as request timeout
            return "timeout"
        return "other"
    return "other"


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    zdr_enabled: bool = False,
    thinking_effort: Optional[str] = None,
    include_error_kind: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        zdr_enabled: Restrict routing to OpenRouter ZDR endpoints
        thinking_effort: Optional OpenRouter reasoning effort for supported models
        include_error_kind: Return {"error": True, "error_kind": kind} instead of None on failure

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed

    Raises:
        ValueError: zdr_enabled is True but the configured provider isn't
            OpenRouter. ZDR routing is an OpenRouter-specific `provider`
            field a generic OpenAI-compatible endpoint (relay, Ollama, LM
            Studio) has no notion of — silently dropping it would send
            content to that endpoint while callers still believe it was
            ZDR-routed. backend.main.prepare_turn rejects zdr_enabled turns
            before they reach here for normal chat/council traffic; this
            raise is the hard backstop for any other caller (e.g. the
            attachment upload/vision-extraction path) that doesn't go
            through that pre-flight.
    """
    # Raised BEFORE any network call is prepared — a caller error here must
    # never silently downgrade to an unprotected request.
    if zdr_enabled and not provider_is_openrouter():
        raise ValueError("ZDR routing requires OpenRouter")

    api_key = get_openrouter_api_key()
    if not api_key:
        logger.error("OpenRouter API key not configured")
        if include_error_kind:
            return {"error": True, "error_kind": "auth"}
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }
    reasoning_payload, endpoint_pin = resolve_model_reasoning(
        model, thinking_effort, zdr_enabled=zdr_enabled
    )
    provider = {}
    if zdr_enabled:
        provider["zdr"] = True  # HARD: never weakened
    if endpoint_pin and provider_is_openrouter():
        # Capabilities are provider-specific: route to the endpoint the shape was
        # learned on (probe pins the same way), so a probed reasoning object can't be
        # mis-applied on a different endpoint. `provider.order`/`allow_fallbacks` are
        # OpenRouter-only -- an openai-compatible relay rejects `provider` (400), and
        # the probe's endpoint tags don't apply there, so the pin is skipped off-OR.
        #
        # ZDR turns never reach here: resolve_model_reasoning returns no pin (and no
        # endpoint-specific shape) for them -- see its docstring.
        provider["order"] = [endpoint_pin]
        provider["allow_fallbacks"] = False
    if provider:
        payload["provider"] = provider
    if reasoning_payload:
        payload["reasoning"] = reasoning_payload

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=connect_timeout_for(timeout))
        ) as client:
            try:
                response = await asyncio.wait_for(
                    client.post(
                        OPENROUTER_API_URL,
                        headers=headers,
                        json=payload
                    ),
                    timeout=timeout,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Some OpenAI-compatible endpoints reject the `reasoning`
                # field outright (400) instead of ignoring it. Retry once
                # without it rather than failing the whole turn.
                if (
                    e.response.status_code == 400
                    and "reasoning" in payload
                    and not provider_is_openrouter()
                ):
                    logger.warning(
                        "OpenAI-compatible endpoint rejected reasoning field for model=%s; retrying without it",
                        model,
                    )
                    payload.pop("reasoning")
                    response = await asyncio.wait_for(
                        client.post(
                            OPENROUTER_API_URL,
                            headers=headers,
                            json=payload
                        ),
                        timeout=timeout,
                    )
                    response.raise_for_status()
                else:
                    raise

            data = response.json()
            message = data['choices'][0]['message']

            content = message.get('content', '')
            reasoning = ""
            
            # Extract reasoning if model supports it
            content, reasoning = extract_reasoning(content, message, model)

            # Extract usage if available
            usage = data.get('usage', {})

            return {
                'content': content,
                'reasoning_details': reasoning if reasoning else None,
                'usage': usage
            }

    except Exception as e:
        kind = classify_openrouter_error(e)
        logger.error(
            "OpenRouter call failed model=%s kind=%s error=%s",
            model, kind, e,
        )
        if include_error_kind:
            return {"error": True, "error_kind": kind}
        return None


def _lookup_registry_model(model: str) -> Optional[Dict[str, Any]]:
    """Look up a model's curated registry entry by id, or None if not found."""
    from .config import CURATED_MODELS

    return next((candidate for candidate in CURATED_MODELS if candidate["id"] == model), None)


_reasoning_capabilities_cache = None


def _reasoning_capability_records() -> Dict[str, Any]:
    """Cached load of the probe-backed capability sidecar. Read-only at runtime --
    the probe is a separate maintainer process, so a running app loads the sidecar
    once (a fresh probe is picked up on restart). Tests reset via
    reset_reasoning_capability_cache()."""
    global _reasoning_capabilities_cache
    if _reasoning_capabilities_cache is None:
        from .reasoning_capability import load_capabilities
        _reasoning_capabilities_cache = load_capabilities()
    return _reasoning_capabilities_cache


def reset_reasoning_capability_cache() -> None:
    """Drop the cached sidecar (test seam; also lets a long-running process re-read
    after a probe if a caller chooses to)."""
    global _reasoning_capabilities_cache
    _reasoning_capabilities_cache = None


def model_supports_reasoning(
    model: str,
    capability_records: Optional[Dict[str, Any]] = None,
    *,
    zdr_enabled: bool = False,
) -> bool:
    """Whether this model accepts reasoning controls, from the SINGLE capability
    authority (v1.3.0 correction #2). The probe-backed sidecar is authoritative once
    a model is probed; the registry's `supports_reasoning` is a DEPRECATED fallback
    used ONLY while the sidecar has no verified record for the model (it is removed
    outright once the probe covers every model). `capability_records` is injectable
    for tests; runtime uses the cached sidecar load.

    `zdr_enabled` bypasses the probe row for the same reason resolve_model_reasoning
    does: the row describes whichever endpoint ORDINARY routing selected and says
    nothing about the ZDR endpoint this request will reach. Consulting it would let a
    `none` row silently suppress reasoning on a distinct ZDR route, and a positive row
    enable it on an unverified one."""
    from .reasoning_capability import get_capability

    registry_model = _lookup_registry_model(model)
    # Probe rows are OpenRouter-endpoint-specific: only consult them when routing to
    # OpenRouter. Off-OpenRouter (a relay) the probed capability doesn't apply, so use
    # the deprecated registry fallback (correction #2). A ZDR turn is the same case:
    # the probed endpoint is not the endpoint this request reaches.
    if provider_is_openrouter() and not zdr_enabled:
        records = capability_records if capability_records is not None else _reasoning_capability_records()
        cap = get_capability(records, model, registry_model)
        if cap.get("control_surface") != "unknown":
            # Probed -> the sidecar decides (a "none" surface correctly reports False).
            return bool(cap.get("supports_reasoning"))
    # Un-probed / off-OpenRouter -> DEPRECATED registry fallback (correction #2).
    return registry_model is not None and registry_model.get("supports_reasoning") is True


def resolve_model_reasoning(
    model: str,
    thinking_effort: Optional[str],
    capability_records: Optional[Dict[str, Any]] = None,
    *,
    zdr_enabled: bool = False,
) -> tuple:
    """Return `(reasoning_object, endpoint_pin)` for this model from the single
    capability authority (correction #2 + the B4 wire boundary).

    A PROBED model uses the B2/B4 per-surface translation -- `levels` snaps to the
    nearest supported effort, `onoff` -> {"enabled": ...}, `budget` ->
    {"max_tokens": ...}, `none`/Auto -> omit -- and, when a shape IS sent, returns
    the record's `provider_pinned` endpoint tag as `endpoint_pin`: capabilities are
    provider-specific (the probe pins `provider.order` and re-probes on a provider
    change), so the caller MUST route to that endpoint or the shape can be silently
    dropped/mis-applied on a different one. An UN-PROBED model falls back to a plain
    {"effort": ...} object iff the DEPRECATED registry supports_reasoning says so
    (today's behavior, until the probe runs), with no pin.
    `reasoning_object` is None to send no reasoning object.

    ZDR (`zdr_enabled`) BYPASSES the probed record entirely -- neither the pin nor
    the endpoint-specific shape is used. The probe observes whichever endpoint
    ORDINARY (non-ZDR) routing selects, and /endpoints publishes no per-endpoint ZDR
    field, so a probed row says nothing about the endpoint a ZDR turn will reach.
    Pinning it would intersect with `zdr: true` under `allow_fallbacks: false` and
    yield NO ROUTE; sending its native shape ({"enabled":...}, {"max_tokens":...}) to
    a DIFFERENT, unverified ZDR endpoint would be exactly the mis-application the pin
    exists to prevent. Falls back to the normalized `reasoning.effort` interface,
    which is provider-agnostic (preflight rule 1)."""
    from .reasoning_capability import get_capability
    from .reasoning_control import resolve_reasoning_payload

    registry_model = _lookup_registry_model(model)
    # Probe rows are OpenRouter-endpoint-specific: only apply them when routing to
    # OpenRouter. Off-OpenRouter (a relay) a probed row must not suppress/reshape
    # reasoning -- fall through to the deprecated registry fallback below.
    # A ZDR turn is treated the same way: the probed endpoint is not the endpoint
    # this request will reach, so neither its shape nor its pin may be applied.
    if provider_is_openrouter() and not zdr_enabled:
        records = capability_records if capability_records is not None else _reasoning_capability_records()
        cap = get_capability(records, model, registry_model)
        if cap.get("control_surface") != "unknown":
            reasoning = resolve_reasoning_payload(cap, thinking_effort)
            # Only send a shape the runtime can also PARSE. A probed record can flip a
            # model to reasoning-supported, but extract_reasoning() still needs a known
            # extraction mode (registry `reasoning_extraction`); without one we'd burn
            # reasoning tokens while dropping the returned reasoning TEXT. Omit until the
            # extraction mode is known (a probe that records it -> the follow-up).
            if reasoning is not None and (registry_model or {}).get("reasoning_extraction") not in ("field", "tags"):
                return None, None
            # Pin only when we actually send a probed shape (nothing to mis-route otherwise).
            pin = cap.get("provider_pinned") if reasoning is not None else None
            return reasoning, pin
    # Un-probed OR off-OpenRouter -> deprecated fallback: a plain effort object iff the
    # registry (via the single authority) says the model supports reasoning. No pin.
    if thinking_effort and model_supports_reasoning(
        model, capability_records=capability_records, zdr_enabled=zdr_enabled
    ):
        return {"effort": thinking_effort}, None
    return None, None


def reasoning_tokens_from_usage(usage: Optional[Dict[str, Any]]) -> Optional[int]:
    """The reasoning tokens a member actually spent this turn, or None when none
    were reported (v1.3.0 B5 honesty, §3.3/§3.5). This is the ONLY reasoning signal
    we report post-turn -- NEVER the presence of reasoning text, which can be
    non-empty while `reasoning_tokens` is 0. So text-present-but-tokens-0 reads as
    'reasoning not available' (None), the exact honesty trap."""
    details = (usage or {}).get("completion_tokens_details") or {}
    rt = details.get("reasoning_tokens")
    if isinstance(rt, (int, float)) and not isinstance(rt, bool) and rt > 0:
        return int(rt)
    return None


def extract_reasoning(content: str, message: Dict[str, Any], model: str) -> tuple[str, str]:
    """
    Extract reasoning from the response based on model capabilities.
    
    Args:
        content: The response content string
        message: The full message object from API
        model: The model identifier
        
    Returns:
        Tuple of (clean_content, extracted_reasoning)
    """
    import re

    # 1. Capability Check
    # If model is not in our registry, do not extract reasoning
    registry_model = _lookup_registry_model(model)
    extraction_mode = registry_model.get("reasoning_extraction") if registry_model else None
    if extraction_mode not in ("field", "tags"):
        return content, ""

    reasoning = ""

    # 2. Field Extraction (Precedence 1)
    if extraction_mode == "field":
        if message.get("reasoning"):
            reasoning = message["reasoning"]
        elif message.get("reasoning_details"):
            # reasoning_details may be a string, a dict, or (OpenRouter's
            # documented non-streaming shape) a list of reasoning blocks
            # carrying `text` or `summary` — same normalization as
            # reasoning_stream.ReasoningStreamState.
            rd = message["reasoning_details"]
            if isinstance(rd, str):
                reasoning = rd
            else:
                blocks = rd if isinstance(rd, list) else [rd]
                parts = []
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    if isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    elif isinstance(block.get("summary"), str):
                        parts.append(block["summary"])
                reasoning = "\n\n".join(parts)
                if not reasoning and isinstance(rd, dict):
                    # provider-specific dict without text/summary: keep legacy behavior
                    reasoning = str(rd)
    
    # 3. Tag Parsing (Precedence 2)
    # Deliberately gated to tags-mode models: field-mode answers that merely
    # MENTION <think>/<thinking> markup (e.g. explaining tags) must not have
    # that text stripped as if it were hidden reasoning.
    if not reasoning and extraction_mode == "tags":
        # Non-greedy regex to find <think> or <thinking> blocks
        # Matches: <think>...</think> OR <thinking>...</thinking>
        pattern = r"<(think|thinking)>([\s\S]*?)</\1>"
        
        matches = re.findall(pattern, content)
        if matches:
            # Extract all reasoning blocks
            reasoning_blocks = [m[1].strip() for m in matches]
            reasoning = "\n\n".join(reasoning_blocks)
            
            # Remove tags from content
            content = re.sub(pattern, "", content).strip()
            
    # 4. Truncation
    if reasoning and len(reasoning) > 2000:
        reasoning = reasoning[:2000] + "\n...(reasoning truncated)"
        
    return content, reasoning


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    zdr_enabled: bool = False,
    thinking_effort: Optional[str] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model
        zdr_enabled: Restrict routing to OpenRouter ZDR endpoints
        thinking_effort: Optional OpenRouter reasoning effort for supported models

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    query_kwargs = {"zdr_enabled": zdr_enabled}
    if thinking_effort is not None:
        query_kwargs["thinking_effort"] = thinking_effort

    tasks = [query_model(model, messages, **query_kwargs) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


async def query_models_as_completed(
    models: List[str],
    messages: List[Dict[str, str]],
    zdr_enabled: bool = False,
    thinking_effort: Optional[str] = None,
    include_error_kind: bool = False,
) -> AsyncIterator[Tuple[str, Optional[Dict[str, Any]]]]:
    """
    Query multiple models concurrently, yielding (model, response) as each
    finishes rather than waiting for the slowest (used for progressive Stage 1).

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model
        zdr_enabled: Restrict routing to OpenRouter ZDR endpoints
        thinking_effort: Optional OpenRouter reasoning effort for supported models
        include_error_kind: Preserve coarse failure kinds instead of yielding bare None

    Yields:
        (model, response) tuples in completion order; response is None on failure
        unless include_error_kind is true.
    """
    query_kwargs = {"zdr_enabled": zdr_enabled}
    if thinking_effort is not None:
        query_kwargs["thinking_effort"] = thinking_effort
    if include_error_kind:
        query_kwargs["include_error_kind"] = True

    async def _labeled(model: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        return model, await query_model(model, messages, **query_kwargs)

    # Wrap each call in an explicit Task (not a bare coroutine) so it can be
    # cancelled if this generator is closed early (browser closes/navigates
    # mid-Stage-1) instead of running to its full HTTP timeout for nothing.
    tasks = [asyncio.ensure_future(_labeled(model)) for model in models]
    try:
        for task in asyncio.as_completed(tasks):
            yield await task
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
