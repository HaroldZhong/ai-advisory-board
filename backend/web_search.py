"""
Stage 0 web grounding via Parallel Search MCP or Perplexity (OpenRouter).
"""

from typing import Dict, Any, Optional
import os
from . import config
from .openrouter import query_model
from .logger import logger

# Model mapping for search depth
SEARCH_MODELS = {
    "fast": "perplexity/sonar",          # $1/M in+out — quick web grounding
    "deep": "perplexity/sonar-pro",      # $3/M in, $15/M out — thorough research
}


def resolve_search_provider(depth: str = "fast", zdr_enabled: bool = False) -> str:
    """Preserve configured OpenRouter/deep/ZDR choices; keyless fast otherwise.

    Explicit providers never silently fall back after an error. A saved .env
    setting and a process override use the normal config.load_dotenv precedence.
    """
    provider = os.getenv("AAB_WEB_SEARCH_PROVIDER", "auto").strip().lower()
    if provider not in ("auto", "parallel", "perplexity", "disabled"):
        raise ValueError("AAB_WEB_SEARCH_PROVIDER must be auto, parallel, perplexity or disabled")
    if provider != "auto":
        return provider
    if (zdr_enabled or depth == "deep"
            or (config.provider_is_openrouter() and config.has_openrouter_api_key())):
        return "perplexity"
    return "parallel"


async def web_search_stage0(
    query: str,
    depth: str = "fast",
    timeout: float = 30.0,
    zdr_enabled: bool = False,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform web search grounding without changing explicit provider/privacy choices.

    Args:
        query: The user's question.
        depth: "fast" (Parallel or sonar) or "deep" (sonar-pro).
        timeout: Request timeout.
        zdr_enabled: Restrict routing to OpenRouter ZDR endpoints.

    Returns:
        Dict with keys: context (str), citations (list), model (str), usage (dict).
        If search fails, context will be empty string.
    """
    model = SEARCH_MODELS.get(depth, SEARCH_MODELS["fast"])

    try:
        provider = resolve_search_provider(depth, zdr_enabled)
        if provider == "disabled":
            return {"context": "", "citations": [], "model": "", "usage": {}}
        if provider == "parallel":
            # Never send ZDR data to a service outside OpenRouter's ZDR routing.
            if zdr_enabled or depth != "fast":
                raise ValueError("Parallel supports fast search without ZDR; choose Perplexity for deep/ZDR search")
            from .parallel_search import search
            return await search(query, timeout=timeout, conversation_id=conversation_id)
    except Exception as exc:
        logger.warning("[WEBSEARCH] Search unavailable: %s", exc)
        return {"context": "", "citations": [], "model": "", "usage": {}}

    system_prompt = (
        "You are a research assistant. Search the web for the most current and relevant information "
        "to answer the user's question. Return a concise summary of your findings with inline citations "
        "in [Source Title](URL) format. Focus on facts, data, and recent developments. "
        "If no relevant web results are found, respond with exactly: NO_WEB_RESULTS"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        logger.info("[WEBSEARCH] Stage 0: Querying %s for query=%r...", model, query[:80])
        response = await query_model(
            model,
            messages,
            timeout=timeout,
            zdr_enabled=zdr_enabled,
        )

        if not response or not response.get("content"):
            logger.warning("[WEBSEARCH] No response from %s", model)
            return {"context": "", "citations": [], "model": model, "usage": {}}

        content = response["content"].strip()

        if "NO_WEB_RESULTS" in content:
            logger.info("[WEBSEARCH] No relevant web results found.")
            return {"context": "", "citations": [], "model": model, "usage": response.get("usage", {})}

        # Extract any URLs as citations
        import re
        urls = re.findall(r'https?://[^\s\)\]]+', content)
        citations = list(set(urls))[:10]  # Dedupe, cap at 10

        logger.info("[WEBSEARCH] Got %d chars of web context with %d citations", len(content), len(citations))

        return {
            "context": content,
            "citations": citations,
            "model": model,
            "usage": response.get("usage", {}),
        }

    except Exception as e:
        logger.error("[WEBSEARCH] Stage 0 failed: %s", e, exc_info=True)
        return {"context": "", "citations": [], "model": model, "usage": {}}
