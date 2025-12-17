import time
import uuid
from typing import Dict, Any
from ..types import ToolResult, ToolError
from ..registry import ToolRegistry

@ToolRegistry.register(
    name="web.search",
    description="Search the web for current information. Returns a list of candidate pages (title, URL, snippet).",
    args_schema={
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "The search query"},
            "num": {"type": "integer", "default": 5, "maximum": 10}
        },
        "required": ["q"]
    },
    examples=["Find AAPL stock price", "Latest AI news"]
)
async def web_search(args: Dict[str, Any], run_id: str, call_id: str) -> ToolResult:
    """
    Mock implementation of web search.
    """
    query = args.get("q", "")
    num = args.get("num", 5)
    
    # Simulate latency
    # time.sleep(0.1) 
    
    # Mock results
    results = []
    for i in range(num):
        results.append({
            "source_id": f"s{i+1}",
            "title": f"Result {i+1} for {query}",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"This is a mocked snippet for result {i+1} related to {query}..."
        })

    return ToolResult(
        id=call_id,
        run_id=run_id,
        ok=True,
        data={"results": results},
        meta={"latency_ms": 100, "cached": False}
    )

@ToolRegistry.register(
    name="web.fetch",
    description="Fetch the full content of a specific URL. Use this to get details from a search result.",
    args_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "format": "uri"}
        },
        "required": ["url"]
    },
    examples=["Read usage policy from openai.com"]
)
async def web_fetch(args: Dict[str, Any], run_id: str, call_id: str) -> ToolResult:
    """
    Mock implementation of web fetch with safety checks.
    """
    url = args.get("url", "")
    
    # Safety Check: URLs must be http/https
    if not (url.startswith("http://") or url.startswith("https://")):
         return ToolResult(
            id=call_id,
            run_id=run_id,
            ok=False,
            error=ToolError(type="validation_error", message="URL must start with http:// or https://"),
            meta={"latency_ms": 0}
        )
    
    # Simulate fetch
    # time.sleep(0.2)
    
    # Mock content
    content = f"Simulated content for {url}. " * 50 # 1000+ chars
    
    # Basic sanitization (mock)
    if "<script>" in content:
        content = content.replace("<script>", "[SCRIPT REMOVED]")
        
    return ToolResult(
        id=call_id,
        run_id=run_id,
        ok=True,
        data={"content": content, "url": url},
        meta={"latency_ms": 200, "size_bytes": len(content)}
    )
