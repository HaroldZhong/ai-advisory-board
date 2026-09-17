"""Discover the configured provider's models; curated entries are policy overlays."""

import asyncio
import hashlib
import math
import time
from typing import Dict, List, Any, Optional

import httpx
from . import config
from .logger import logger
from .config import OPENROUTER_BASE_URL, provider_is_openrouter

OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_ZDR_ENDPOINTS_URL = f"{OPENROUTER_BASE_URL}/endpoints/zdr"
OPENROUTER_KEY_URL = f"{OPENROUTER_BASE_URL}/key"
CACHE_TTL_SECONDS = 3600
REFRESH_COOLDOWN_SECONDS = 30
MAX_CATALOG_BYTES = 8 * 1024 * 1024
MAX_CATALOG_MODELS = 10000
OPENROUTER_NETWORK_HINT = (
    "Could not reach openrouter.ai. If you are behind a firewall "
    "or in a region where openrouter.ai is blocked, set HTTPS_PROXY "
    "or OPENROUTER_BASE_URL."
)
_probe_transport = None
_catalog_transport = None  # Test seam, never a separate provider configuration.
_cache: Dict[str, Any] = {}


def _cache_state():
    global _cache
    # Credentials affect account-scoped catalogs. Never expose this digest to clients.
    scope = (OPENROUTER_MODELS_URL, provider_is_openrouter(),
             hashlib.sha256((config.get_openrouter_api_key() or "").encode()).digest())
    if _cache.get("scope") != scope:
        _cache = {"scope": scope, "models": None, "last_fetched": 0,
                  "last_attempt": 0, "task": None, "error": None}
    return _cache


async def _fetch_catalog_rows(url, id_field):
    """Only send credentials to the configured URL. Never follow provider links."""
    key = config.get_openrouter_api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    rows, seen, cursor = [], set(), None
    expected_total = None
    total_bytes = 0
    deadline = time.monotonic() + 20
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False,
                                     transport=_catalog_transport) as client:
            while True:
                if time.monotonic() > deadline:
                    raise ValueError("catalog deadline")
                async with client.stream("GET", url, headers=headers,
                                         params={"after": cursor} if cursor else None) as response:
                    response.raise_for_status()
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        total_bytes += len(chunk)
                        if total_bytes > MAX_CATALOG_BYTES or time.monotonic() > deadline:
                            raise ValueError("oversized catalog")
                    import json
                    payload = json.loads(body)
                if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                    raise ValueError("invalid catalog shape")
                page = payload["data"]
                for row in page:
                    if not isinstance(row, dict) or not isinstance(row.get(id_field), str) or not row[id_field].strip():
                        raise ValueError("invalid catalog row")
                    identity = row[id_field]
                    if identity != identity.strip() or len(identity) > 512 or any(ord(c) < 32 for c in identity):
                        raise ValueError("invalid catalog identity")
                    if id_field == "id" and identity in seen:
                        raise ValueError("duplicate model id")
                    seen.add(identity)
                    rows.append(row)
                    if len(rows) > MAX_CATALOG_MODELS:
                        raise ValueError("oversized catalog")
                # Honor same-endpoint cursors; never follow provider-supplied URLs.
                links = payload.get("links", {})
                if not isinstance(links, dict) or links.get("next") is not None:
                    raise ValueError("unsupported pagination links")
                if any(payload.get(k) for k in ("next", "next_page", "next_cursor")):
                    raise ValueError("unsupported pagination")
                for field in ("total", "total_count"):
                    if field in payload:
                        total = payload[field]
                        if type(total) is not int or total < 0 or (expected_total is not None and total != expected_total):
                            raise ValueError("invalid catalog total")
                        expected_total = total
                if expected_total is not None and len(rows) > expected_total:
                    raise ValueError("incomplete catalog")
                if payload.get("has_more", False) is False:
                    if expected_total is not None and expected_total != len(rows):
                        raise ValueError("incomplete catalog")
                    return rows
                if payload.get("has_more") is not True or not page:
                    raise ValueError("invalid pagination")
                next_cursor = payload.get("last_id")
                if next_cursor != page[-1][id_field] or next_cursor == cursor:
                    raise ValueError("invalid pagination cursor")
                cursor = next_cursor
    except Exception:
        # No upstream response body, headers, URL query or credential in logs/UI.
        logger.warning("[Models] Provider catalog fetch failed; retaining prior data")
        return None


async def fetch_openrouter_models():
    """Discover via the existing configured OpenAI-compatible /models endpoint."""
    return await _fetch_catalog_rows(OPENROUTER_MODELS_URL, "id")


async def fetch_openrouter_zdr_model_ids():
    rows = await _fetch_catalog_rows(OPENROUTER_ZDR_ENDPOINTS_URL, "model_id")
    return {row["model_id"] for row in rows} if rows is not None else None


def _rate(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (ValueError, OverflowError):
        return None


def parse_openrouter_model(raw, supports_zdr=None):
    raw_pricing = raw.get("pricing") if isinstance(raw.get("pricing"), dict) else {}
    # OpenAI-compatible /models has no standard currency or price units.
    rates = [_rate(raw_pricing.get(k)) if provider_is_openrouter() else None
             for k in ("prompt", "completion")]
    pricing = {k: rate * 1_000_000 if rate is not None and math.isfinite(rate * 1_000_000) else None
               for k, rate in zip(("input", "output"), rates)}
    architecture = raw.get("architecture") if isinstance(raw.get("architecture"), dict) else {}
    parameters = raw.get("supported_parameters")
    parameters = parameters if isinstance(parameters, list) and all(isinstance(p, str) for p in parameters) else None
    modalities = architecture.get("output_modalities")
    text_output = "text" in modalities if isinstance(modalities, list) and modalities else None
    context = raw.get("context_length")
    context = context if type(context) is int and context > 0 else None
    return {
        "id": raw["id"], "name": raw.get("name") if isinstance(raw.get("name"), str) else raw["id"],
        "context_length": context, "pricing": pricing,
        "pricing_source": "provider" if all(r is not None for r in pricing.values()) else "unknown",
        "supports_zdr": supports_zdr, "architecture": architecture,
        "supported_parameters": parameters, "text_output": text_output,
        "top_provider": raw.get("top_provider") if isinstance(raw.get("top_provider"), dict) else {},
    }


async def _refresh_catalog(state):
    state["last_attempt"] = time.time()
    raw = await fetch_openrouter_models()
    if raw is None:
        state["error"] = "Refresh failed. Keeping the previous catalog or recommended fallback."
        return
    zdr = await fetch_openrouter_zdr_model_ids() if provider_is_openrouter() else set()
    parsed = {row["id"]: parse_openrouter_model(row, None if zdr is None else row["id"] in zdr) for row in raw}
    # A configuration/key change or clear_cache during the await invalidates this generation.
    if _cache_state() is not state:
        return
    state.update(models=parsed, last_fetched=time.time(), error=(
        "Model list updated, but ZDR availability could not be verified. "
        "ZDR selections remain unavailable until a successful refresh."
        if zdr is None else None
    ))


async def get_openrouter_models_cached(force=False):
    state = _cache_state()
    age = time.time() - state["last_fetched"]
    if state["task"] is not None and not state["task"].done():
        await asyncio.shield(state["task"])
    elif (force or state["error"] or state["models"] is None or age >= CACHE_TTL_SECONDS) and time.time() - state["last_attempt"] >= REFRESH_COOLDOWN_SECONDS:
        # ponytail: one configured provider per process; no cross-provider scheduler needed.
        state["task"] = asyncio.create_task(_refresh_catalog(state))
        await asyncio.shield(state["task"])
    return state["models"] if _cache_state() is state else None


def catalog_status():
    state = _cache_state()
    return {"last_fetched": state["last_fetched"] or None,
            "stale": bool(state["error"]) or time.time() - state["last_fetched"] >= CACHE_TTL_SECONDS,
            "error": state["error"], "refresh_interval_seconds": CACHE_TTL_SECONDS,
            "refresh_cooldown_seconds": REFRESH_COOLDOWN_SECONDS,
            "next_refresh_at": state["last_attempt"] + REFRESH_COOLDOWN_SECONDS if state["last_attempt"] else None,
            "provider_kind": config.PROVIDER_KIND}


def _merge_models(curated_models, live):
    curated_by_id = {m["id"]: m for m in curated_models}
    result = []
    for model_id in dict.fromkeys([*curated_by_id, *(live or {})]):
        curated = curated_by_id.get(model_id, {})
        model = {**curated, "id": model_id, "name": curated.get("name", model_id),
                 "type": curated.get("type", "both"), "capabilities": curated.get("capabilities", []),
                 "default_council": curated.get("default_council", False), "recommended": bool(curated),
                 "available": live is None or model_id in live, "pricing_source": "curated"}
        row = (live or {}).get(model_id)
        if row:
            model.update(row)
            if model.get("pricing_source") == "unknown" and provider_is_openrouter() and curated.get("pricing"):
                model.update(pricing=curated["pricing"], pricing_source="curated")
            parameters = row.get("supported_parameters")
            if provider_is_openrouter() and parameters is not None:
                model["supports_reasoning"] = "reasoning" in parameters or "include_reasoning" in parameters
                if model["supports_reasoning"] and not model.get("reasoning_extraction"):
                    model["reasoning_extraction"] = "field"
            inputs = row.get("architecture", {}).get("input_modalities")
            if row.get("text_output") is False or (isinstance(inputs, list) and inputs and "text" not in inputs):
                model["type"] = "other"
        if not provider_is_openrouter():
            model.update(supports_zdr=False, pricing={"input": None, "output": None}, pricing_source="unknown")
        model.setdefault("pricing", {"input": None, "output": None})
        result.append(model)
    return result


async def get_enriched_models(curated_models, force=False):
    live = await get_openrouter_models_cached(force=True) if force else await get_openrouter_models_cached()
    return _merge_models(curated_models, live)


def get_model_metadata(model_id):
    """Synchronous shared lookup; never performs network I/O on a generation path."""
    curated = [m for m in config.CURATED_MODELS if m["id"] == model_id]
    live = _cache_state()["models"]
    row = None if live is None else ({model_id: live[model_id]} if model_id in live else {})
    return next(iter(_merge_models(curated, row)), None)


def clear_cache():
    global _cache
    _cache = {}


async def check_connectivity(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Two-stage probe: (1) unauthenticated GET /models proves network
    reachability only (it serves 200 without credentials); (2) if a key is
    configured, authenticated GET /key validates it (401/403 = bad key).
    Credit exhaustion is NOT detectable here; it surfaces at chat time.

    `api_key`, if given, is validated instead of the configured key — lets
    the first-run UI check a just-typed key before it's saved.
    """
    from .openrouter import classify_openrouter_error
    from . import config

    result = {"reachable": False, "key_valid": None, "error_kind": None, "detail": ""}

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=8.0), transport=_probe_transport
    ) as client:
        # Stage 1: reachability. ANY HTTP response proves the endpoint is
        # reachable — some relays gate /models behind auth (401/403), which
        # still means the network path works. Only transport errors fail here.
        try:
            await client.get(OPENROUTER_MODELS_URL)
            result["reachable"] = True
        except Exception as e:
            kind = classify_openrouter_error(e)
            details = {
                "network": OPENROUTER_NETWORK_HINT,
                "timeout": "openrouter.ai did not respond in time. Check your network or proxy.",
            }
            logger.warning("[OpenRouter] Reachability probe failed kind=%s: %s", kind, e)
            result["error_kind"] = kind
            result["detail"] = details.get(kind, f"Unexpected error: {e}")
            return result

        # Stage 2: key validity (only meaningful once reachable)
        api_key = api_key or config.get_openrouter_api_key()
        if api_key is None:
            result["detail"] = "Network OK. No API key configured yet."
            return result

        try:
            response = await client.get(
                OPENROUTER_KEY_URL,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            result["key_valid"] = True
            result["detail"] = "ok"
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                result["key_valid"] = False
                result["error_kind"] = "auth"
                result["detail"] = "Reached openrouter.ai but the API key was rejected. Check your key."
            elif e.response.status_code in (404, 405, 501):
                # OpenAI-compatible relays (OPENROUTER_BASE_URL) often don't
                # implement OpenRouter's /key endpoint. Reachability is already
                # proven; key validity just can't be checked here — not an error.
                result["detail"] = (
                    "Network OK. This endpoint does not support key validation; "
                    "key status unknown."
                )
            else:
                result["error_kind"] = "other"
                result["detail"] = f"Key check returned HTTP {e.response.status_code}."
        except Exception as e:
            result["error_kind"] = classify_openrouter_error(e)
            result["detail"] = f"Key check failed: {e}"

    return result
