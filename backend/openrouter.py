"""OpenRouter API client for making LLM requests."""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_URL, get_openrouter_api_key
from .logger import logger


def classify_openrouter_error(exc: Exception) -> str:
    """Map an exception from an OpenRouter call to a coarse failure kind."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "network"
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, asyncio.TimeoutError)):
        return "timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return "auth"
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
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        zdr_enabled: Restrict routing to OpenRouter ZDR endpoints
        thinking_effort: Optional OpenRouter reasoning effort for supported models

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    api_key = get_openrouter_api_key()
    if not api_key:
        logger.error("OpenRouter API key not configured")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }
    if zdr_enabled:
        payload["provider"] = {"zdr": True}
    if thinking_effort and model_supports_reasoning(model):
        payload["reasoning"] = {"effort": thinking_effort}

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0)
        ) as client:
            response = await asyncio.wait_for(
                client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload
                ),
                timeout=timeout,
            )
            response.raise_for_status()

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
        logger.error(
            "OpenRouter call failed model=%s kind=%s error=%s",
            model, classify_openrouter_error(e), e,
        )
        return None


def model_supports_reasoning(model: str) -> bool:
    """Return whether the curated registry says a model accepts reasoning controls."""
    from .config import CURATED_MODELS

    registry_model = next((candidate for candidate in CURATED_MODELS if candidate["id"] == model), None)
    return registry_model is not None and registry_model.get("supports_reasoning") is True


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
    from .config import REASONING_MODELS
    import re
    
    # 1. Capability Check
    # If model is not in our registry, do not extract reasoning
    if model not in REASONING_MODELS:
        return content, ""
        
    capabilities = REASONING_MODELS[model]
    reasoning = ""
    
    # 2. Field Extraction (Precedence 1)
    if capabilities.get("use_field"):
        if message.get("reasoning"):
            reasoning = message["reasoning"]
        elif message.get("reasoning_details"):
            # reasoning_details might be a dict or string depending on provider
            rd = message["reasoning_details"]
            if isinstance(rd, str):
                reasoning = rd
            elif isinstance(rd, dict):
                # Try to extract text from dict structure if possible
                # This is provider specific, but common pattern is 'text' or 'content'
                reasoning = str(rd) 
    
    # 3. Tag Parsing (Precedence 2)
    # Only if no reasoning found yet OR explicit parse_tags is requested
    if not reasoning and capabilities.get("parse_tags"):
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
