"""
Stage 0: Web Search via Perplexity (OpenRouter).

Uses Perplexity's sonar/sonar-pro models as a pre-Council research step.
These models have native web search built-in — no external search API needed.
"""

from typing import Dict, Any, Optional
from .openrouter import query_model
from .logger import logger

# Model mapping for search depth
SEARCH_MODELS = {
    "fast": "perplexity/sonar",          # $1/M in+out — quick web grounding
    "deep": "perplexity/sonar-pro",      # $3/M in, $15/M out — thorough research
}


async def web_search_stage0(
    query: str,
    depth: str = "fast",
    timeout: float = 30.0,
    zdr_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Perform web search grounding via Perplexity models on OpenRouter.

    Args:
        query: The user's question.
        depth: "fast" (sonar) or "deep" (sonar-pro).
        timeout: Request timeout.
        zdr_enabled: Restrict routing to OpenRouter ZDR endpoints.

    Returns:
        Dict with keys: context (str), citations (list), model (str), usage (dict).
        If search fails, context will be empty string.
    """
    model = SEARCH_MODELS.get(depth, SEARCH_MODELS["fast"])

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
