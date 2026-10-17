"""Anonymous Parallel Search MCP over Streamable HTTP, using the existing HTTP client."""

import asyncio
import json
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import httpx

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "AI-Advisory-Board/0.1.0 (+https://github.com/HaroldZhong/ai-advisory-board)"
PROTOCOL_VERSION = "2025-03-26"
MAX_RESPONSE_BYTES = 1_000_000
_PROCESS_SESSION = uuid.uuid4().hex


def _result(reply: Any, request_id: int) -> dict:
    if not isinstance(reply, dict) or reply.get("jsonrpc") != "2.0" or reply.get("id") != request_id:
        raise ValueError("Invalid Parallel MCP response")
    if "error" in reply:
        raise ValueError("Parallel MCP returned a protocol error")
    if not isinstance(reply.get("result"), dict):
        raise ValueError("Missing Parallel MCP result")
    return reply["result"]


async def _rpc(client: httpx.AsyncClient, request_id: Optional[int], method: str,
               params: Optional[dict] = None) -> dict:
    payload = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        payload["id"] = request_id
    if params is not None:
        payload["params"] = params
    async with client.stream("POST", MCP_URL, json=payload) as response:
        response.raise_for_status()
        if response.headers.get("mcp-session-id"):
            client.headers["Mcp-Session-Id"] = response.headers["mcp-session-id"]
        if request_id is None:
            return {}
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            # Streamable HTTP may return JSON or SSE. Select our matching reply,
            # ignoring notifications and comments; multiline data follows SSE rules.
            # Return at the reply even if the server keeps the stream open.
            data = []
            size = 0
            async for line in response.aiter_lines():
                size += len(line.encode("utf-8")) + 1
                if size > MAX_RESPONSE_BYTES:
                    raise ValueError("Parallel MCP response exceeds size limit")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                elif not line and data:
                    reply = json.loads("\n".join(data))
                    data = []
                    if isinstance(reply, dict) and reply.get("id") == request_id:
                        return _result(reply, request_id)
            raise ValueError("Missing Parallel MCP stream reply")
        size = 0
        chunks = []
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise ValueError("Parallel MCP response exceeds size limit")
            chunks.append(chunk)
        return _result(json.loads(b"".join(chunks)), request_id)


async def _search(query: str, timeout: float, conversation_id: Optional[str]) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/event-stream"}
    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=False) as client:
        initialized = await _rpc(client, 1, "initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "AI-Advisory-Board", "version": "0.1.0"},
        })
        if initialized.get("protocolVersion") != PROTOCOL_VERSION:
            raise ValueError("Unsupported Parallel MCP protocol version")
        client.headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        await _rpc(client, None, "notifications/initialized")
        tools = await _rpc(client, 2, "tools/list")
        if not any(tool.get("name") == "web_search" for tool in tools.get("tools", [])):
            raise ValueError("Parallel MCP does not expose web_search")
        session_id = (uuid.uuid5(uuid.NAMESPACE_URL, "aab:" + conversation_id).hex
                      if conversation_id else _PROCESS_SESSION)
        result = await _rpc(client, 3, "tools/call", {
            "name": "web_search",
            "arguments": {"objective": query, "search_queries": [query], "session_id": session_id},
        })
        if result.get("isError"):
            raise ValueError("Parallel MCP search failed")
        content = result.get("structuredContent")
        if content is None:
            parts = [part["text"] for part in result.get("content", []) if part.get("type") == "text"]
            content = json.loads("\n".join(parts))
        if not isinstance(content, dict) or not isinstance(content.get("results"), list):
            raise ValueError("Invalid Parallel search results")
        contexts, citations = [], []
        for item in content["results"][:10]:
            url = item.get("url", "")
            parsed = urlsplit(url)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname
                    or parsed.username or parsed.password or any(char.isspace() for char in url)):
                continue
            excerpts = item.get("excerpts", [])
            if not isinstance(excerpts, list) or not all(isinstance(part, str) for part in excerpts):
                raise ValueError("Invalid Parallel search excerpts")
            if url in citations or not any(part.strip() for part in excerpts):
                continue
            title = str(item.get("title") or url).replace("\n", " ")[:300]
            contexts.append(f"{title}\nURL: {url}\n" + "\n".join(excerpts)[:4000])
            citations.append(url)
        return {"context": "\n\n".join(contexts), "citations": citations,
                "model": "parallel/fast", "usage": {}}


async def search(query: str, timeout: float = 30.0,
                 conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """Bound the complete handshake/discovery/search, and propagate cancellation."""
    return await asyncio.wait_for(_search(query, timeout, conversation_id), timeout=timeout)
