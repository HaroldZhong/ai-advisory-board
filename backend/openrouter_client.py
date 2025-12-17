"""OpenRouter API client for fetching live model data."""

import httpx
import time
from typing import Dict, List, Any, Optional
from .logger import logger

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_SECONDS = 3600  # 1 hour cache

# In-memory cache
_cache: Dict[str, Any] = {
    "models": None,
    "last_fetched": 0
}


async def fetch_openrouter_models() -> Optional[List[Dict[str, Any]]]:
    """
    Fetch all available models from OpenRouter API.
    
    Returns:
        List of model dicts or None if fetch fails
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(OPENROUTER_MODELS_URL)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            logger.info(f"[OpenRouter] Fetched {len(models)} models from API")
            return models
    except httpx.TimeoutException:
        logger.warning("[OpenRouter] API request timed out")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"[OpenRouter] API returned status {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"[OpenRouter] Error fetching models: {e}")
        return None


def parse_openrouter_model(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a raw OpenRouter model into our format.
    
    Args:
        raw: Raw model data from OpenRouter API
        
    Returns:
        Parsed model dict with standardized fields
    """
    pricing = raw.get("pricing", {})
    
    # OpenRouter returns price per token as string, convert to per-million
    prompt_price = float(pricing.get("prompt", 0)) * 1_000_000
    completion_price = float(pricing.get("completion", 0)) * 1_000_000
    
    return {
        "id": raw.get("id", ""),
        "name": raw.get("name", raw.get("id", "Unknown")),
        "context_length": raw.get("context_length", 0),
        "pricing": {
            "input": round(prompt_price, 4),
            "output": round(completion_price, 4)
        },
        "architecture": raw.get("architecture", {}),
        "top_provider": raw.get("top_provider", {})
    }


async def get_openrouter_models_cached() -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Get OpenRouter models with caching.
    
    Returns:
        Dict mapping model ID to parsed model data, or None if unavailable
    """
    global _cache
    
    now = time.time()
    cache_age = now - _cache["last_fetched"]
    
    # Return cached data if still valid
    if _cache["models"] is not None and cache_age < CACHE_TTL_SECONDS:
        logger.debug(f"[OpenRouter] Using cached data ({int(cache_age)}s old)")
        return _cache["models"]
    
    # Fetch fresh data
    raw_models = await fetch_openrouter_models()
    
    if raw_models is None:
        # If fetch failed but we have cached data, use it even if stale
        if _cache["models"] is not None:
            logger.warning("[OpenRouter] Using stale cache after fetch failure")
            return _cache["models"]
        return None
    
    # Parse and cache
    parsed = {}
    for raw in raw_models:
        model_id = raw.get("id", "")
        if model_id:
            parsed[model_id] = parse_openrouter_model(raw)
    
    _cache["models"] = parsed
    _cache["last_fetched"] = now
    logger.info(f"[OpenRouter] Cached {len(parsed)} models")
    
    return parsed


async def get_enriched_models(curated_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge curated model list with live OpenRouter data.
    
    The curated list provides:
    - id: Which models to include
    - capabilities: Custom capability tags
    - type: chairman/council/both
    
    OpenRouter provides:
    - name: Display name
    - pricing: Current pricing
    - context_length: Token limit
    
    Args:
        curated_models: List of curated model configs
        
    Returns:
        Enriched model list with live data merged in
    """
    openrouter_data = await get_openrouter_models_cached()
    
    enriched = []
    for curated in curated_models:
        model_id = curated.get("id", "")
        
        # Start with curated data
        enriched_model = {
            "id": model_id,
            "capabilities": curated.get("capabilities", []),
            "type": curated.get("type", "council")
        }
        
        # Merge live data if available
        if openrouter_data and model_id in openrouter_data:
            live = openrouter_data[model_id]
            enriched_model["name"] = live.get("name", curated.get("name", model_id))
            enriched_model["pricing"] = live.get("pricing", curated.get("pricing", {"input": 0, "output": 0}))
            enriched_model["context_length"] = live.get("context_length", 0)
            enriched_model["available"] = True
        else:
            # Fallback to curated data
            enriched_model["name"] = curated.get("name", model_id)
            enriched_model["pricing"] = curated.get("pricing", {"input": 0, "output": 0})
            enriched_model["context_length"] = curated.get("context_length", 0)
            enriched_model["available"] = openrouter_data is None  # Unknown if API failed
            
            if openrouter_data is not None:
                logger.warning(f"[OpenRouter] Model {model_id} not found in API response")
        
        enriched.append(enriched_model)
    
    return enriched


def clear_cache():
    """Clear the model cache (useful for testing)."""
    global _cache
    _cache = {"models": None, "last_fetched": 0}
    logger.info("[OpenRouter] Cache cleared")
