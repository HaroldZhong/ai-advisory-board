"""Production resolver/HTTP caller and turn consumption; model replies are scripted."""
import asyncio
import importlib
import json
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def search_modules(monkeypatch, tmp_path):
    monkeypatch.setenv("AAB_DATA_DIR", str(tmp_path / "app"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("AAB_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    from backend import config, parallel_search, web_search
    monkeypatch.setattr(config, "provider_is_openrouter", lambda: True)
    return web_search, parallel_search


def mock_mcp(monkeypatch, parallel, *, sse=False, text_content=False, failure=None, rows=None):
    requests = []
    rows = rows if rows is not None else [
        {"title": "Python release", "url": "https://www.python.org/downloads/", "excerpts": ["Python release details."]},
        {"title": "Unsafe", "url": "javascript:alert(1)", "excerpts": ["ignore"]},
        {"title": "Duplicate", "url": "https://www.python.org/downloads/", "excerpts": ["duplicate"]},
    ]

    async def handle(request):
        requests.append(request)
        assert str(request.url) == "https://search.parallel.ai/mcp"
        assert request.headers["user-agent"] == parallel.USER_AGENT
        assert "authorization" not in request.headers
        assert "text/event-stream" in request.headers["accept"]
        payload = json.loads(request.content)
        method = payload["method"]
        if method != "initialize":
            assert request.headers["mcp-session-id"] == "test-session"
            assert request.headers["mcp-protocol-version"] == parallel.PROTOCOL_VERSION
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": parallel.PROTOCOL_VERSION, "capabilities": {"tools": {}}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search"}]}
        else:
            assert method == "tools/call" and payload["params"]["name"] == "web_search"
            assert payload["params"]["arguments"]["objective"]
            assert payload["params"]["arguments"]["search_queries"]
            assert len(payload["params"]["arguments"]["session_id"]) == 32
            if failure == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            if failure == "http":
                return httpx.Response(429, headers={"Retry-After": "60"})
            if failure == "malformed":
                return httpx.Response(200, text="not JSON")
            if failure == "protocol":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "error": {"code": -32603}})
            if failure == "tool":
                result = {"isError": True, "content": [{"type": "text", "text": "quota"}]}
            elif failure == "id":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": 999, "result": {}})
            else:
                content = {"results": rows}
                result = ({"content": [{"type": "text", "text": json.dumps(content)}]}
                          if text_content else {"structuredContent": content})
        body = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        headers = {"mcp-session-id": "test-session"}
        if sse:
            return httpx.Response(200, headers={**headers, "content-type": "text/event-stream"},
                                  text=": keepalive\r\n\r\nevent: message\r\ndata: " + json.dumps(body) + "\r\n\r\n")
        return httpx.Response(200, headers=headers, json=body)

    original = httpx.AsyncClient

    def factory(**kwargs):
        assert kwargs["follow_redirects"] is False
        return original(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(parallel, "httpx", SimpleNamespace(AsyncClient=factory))
    return requests


@pytest.mark.parametrize("key,openrouter,depth,zdr,choice,expected", [
    ("", True, "fast", False, None, "parallel"),
    ("saved-key", True, "fast", False, None, "perplexity"),
    ("local-key", False, "fast", False, None, "parallel"),
    ("", True, "deep", False, None, "perplexity"),
    ("", True, "fast", True, None, "perplexity"),
    ("saved-key", True, "fast", False, "parallel", "parallel"),
    ("", True, "fast", False, "perplexity", "perplexity"),
    ("saved-key", True, "fast", False, "disabled", "disabled"),
])
def test_resolver_preserves_choices(search_modules, monkeypatch, key, openrouter, depth, zdr, choice, expected):
    web, _parallel = search_modules
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    monkeypatch.setattr(web.config, "provider_is_openrouter", lambda: openrouter)
    if choice:
        monkeypatch.setenv("AAB_WEB_SEARCH_PROVIDER", choice)
    assert web.resolve_search_provider(depth, zdr) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("sse,text_content", [(False, False), (True, False), (False, True)])
async def test_anonymous_caller_contract(search_modules, monkeypatch, sse, text_content):
    web, parallel = search_modules
    requests = mock_mcp(monkeypatch, parallel, sse=sse, text_content=text_content)
    result = await web.web_search_stage0("latest Python release details", conversation_id="conversation")
    assert result["model"] == "parallel/fast" and result["usage"] == {}
    assert result["citations"] == ["https://www.python.org/downloads/"]
    assert "Python release details." in result["context"] and "javascript:" not in result["context"]
    assert [json.loads(r.content)["method"] for r in requests] == [
        "initialize", "notifications/initialized", "tools/list", "tools/call"]
    first = json.loads(requests[-1].content)["params"]["arguments"]["session_id"]
    await web.web_search_stage0("follow up Python details", conversation_id="conversation")
    assert json.loads(requests[-1].content)["params"]["arguments"]["session_id"] == first


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "http", "malformed", "protocol", "tool", "id"])
async def test_failed_parallel_never_falls_back(search_modules, monkeypatch, failure):
    web, parallel = search_modules
    monkeypatch.setenv("AAB_WEB_SEARCH_PROVIDER", "parallel")
    monkeypatch.setenv("OPENROUTER_API_KEY", "explicit-model-key")
    mock_mcp(monkeypatch, parallel, failure=failure)
    async def forbidden(*a, **k):
        pytest.fail("Explicit provider failure must not invoke Perplexity")
    monkeypatch.setattr(web, "query_model", forbidden)
    result = await web.web_search_stage0("Python release details")
    assert result["context"] == "" and result["citations"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("choice,depth,zdr", [("disabled", "fast", False), ("parallel", "deep", False),
                                               ("parallel", "fast", True), ("typo", "fast", False)])
async def test_permissions_and_unsupported_modes_make_no_requests(search_modules, monkeypatch, choice, depth, zdr):
    web, parallel = search_modules
    requests = mock_mcp(monkeypatch, parallel)
    monkeypatch.setenv("AAB_WEB_SEARCH_PROVIDER", choice)
    result = await web.web_search_stage0("private question", depth=depth, zdr_enabled=zdr)
    assert not result["context"] and not requests


@pytest.mark.asyncio
async def test_perplexity_error_does_not_fall_back(search_modules, monkeypatch):
    web, parallel = search_modules
    monkeypatch.setenv("OPENROUTER_API_KEY", "saved-key")
    requests = mock_mcp(monkeypatch, parallel)
    calls = []
    async def fail(*args, **kwargs):
        calls.append(args)
        return None
    monkeypatch.setattr(web, "query_model", fail)
    result = await web.web_search_stage0("question")
    assert calls and result["model"] == "perplexity/sonar" and not requests


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_total_timeout_and_cancellation_close_transport(search_modules, monkeypatch, cancel):
    web, parallel = search_modules
    started, closed = asyncio.Event(), asyncio.Event()
    async def handle(request):
        started.set()
        try:
            await asyncio.Future()
        finally:
            closed.set()
    original = httpx.AsyncClient
    monkeypatch.setattr(parallel, "httpx", SimpleNamespace(
        AsyncClient=lambda **kw: original(**kw, transport=httpx.MockTransport(handle))))
    task = asyncio.create_task(web.web_search_stage0("question", timeout=1 if cancel else 0.01))
    await started.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert (await task)["context"] == ""
    assert closed.is_set()


def test_estimates_follow_provider(search_modules, monkeypatch):
    _web, _parallel = search_modules
    from backend.budget_router import estimate_web_search_cost
    assert estimate_web_search_cost() == 0
    assert estimate_web_search_cost("deep") > 0
    monkeypatch.setenv("OPENROUTER_API_KEY", "saved-key")
    assert estimate_web_search_cost() > 0
    monkeypatch.setenv("AAB_WEB_SEARCH_PROVIDER", "parallel")
    assert estimate_web_search_cost() == 0
    monkeypatch.setenv("AAB_WEB_SEARCH_PROVIDER", "disabled")
    assert estimate_web_search_cost() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "council"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_turn_entrypoint_consumes_web_results(search_modules, monkeypatch, tmp_path, mode, enabled):
    """Real send/turn/resolver/MCP path, with deterministic model responses."""
    from test_pipeline_unification import _setup_chat_fakes, _setup_council_fakes
    web, parallel = search_modules
    requests = mock_mcp(monkeypatch, parallel)
    main = importlib.import_module("backend.main")
    council = importlib.import_module("backend.council")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("web-turn")
    captured = []
    if mode == "chat":
        _setup_chat_fakes(monkeypatch, main)
        # Keep actual context rendering and result consumption in chat_with_chairman.
        monkeypatch.setattr(main, "chat_with_chairman", council.chat_with_chairman)
        async def model(model, messages, **kwargs):
            captured.append(messages[0]["content"])
            return {"content": "Scripted answer", "usage": {}}
        monkeypatch.setattr(council, "query_model", model)
    else:
        _setup_council_fakes(monkeypatch, main, stage1_calls=captured)
    result = await main.send_message("web-turn", main.SendMessageRequest(
        content="latest Python release details", mode=mode, web_search_enabled=enabled))
    assert result["type"] == mode
    assert bool(requests) == enabled
    assert any("Python release details." in text for text in captured) == enabled
    assert main.storage.get_conversation("web-turn")["messages"][-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_sse_returns_reply_before_stream_closes(search_modules):
    _web, parallel = search_modules
    class ReplyStream(httpx.AsyncByteStream):
        closed = False
        async def __aiter__(self):
            yield b': ping\n\ndata: {"jsonrpc":"2.0","method":"notice"}\n\n'
            yield b'data: {"jsonrpc":"2.0",\ndata: "id":1,"result":{"ok":true}}\n\n'
            await asyncio.Future()
        async def aclose(self):
            self.closed = True
    stream = ReplyStream()
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )) as client:
        result = await asyncio.wait_for(parallel._rpc(client, 1, "test"), timeout=1)
    assert result == {"ok": True} and stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_url", ["file:///etc/passwd", "https://user:pass@example.com", "https:///missing", "https://a b.test/", "data:text/html,x"])
async def test_non_web_and_invalid_source_urls_are_ignored(search_modules, monkeypatch, bad_url):
    web, parallel = search_modules
    mock_mcp(monkeypatch, parallel, rows=[{"url": bad_url, "excerpts": ["not a usable source"]}])
    result = await web.web_search_stage0("question")
    assert result["context"] == "" and result["citations"] == []


@pytest.mark.asyncio
async def test_oversized_reply_is_rejected(search_modules, monkeypatch):
    web, parallel = search_modules
    monkeypatch.setattr(parallel, "MAX_RESPONSE_BYTES", 50)
    mock_mcp(monkeypatch, parallel)
    result = await web.web_search_stage0("question")
    assert not result["context"]


def test_saved_configuration_and_environment_override(tmp_path):
    """Read real saved dotenv settings in fresh processes, isolated from ambient keys."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    root = tmp_path / "settings"
    root.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in (
        "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "LLM_PROVIDER_KIND", "AAB_WEB_SEARCH_PROVIDER")}
    env["AAB_DATA_DIR"] = str(root)
    command = [sys.executable, "-c", "from backend.web_search import resolve_search_provider; print(resolve_search_provider())"]
    repo = Path(__file__).resolve().parents[1]
    (root / ".env").write_text("OPENROUTER_API_KEY=saved-model-key\n", encoding="utf-8")
    assert subprocess.check_output(command, cwd=repo, env=env, text=True).strip() == "perplexity"
    (root / ".env").write_text("AAB_WEB_SEARCH_PROVIDER=disabled\n", encoding="utf-8")
    assert subprocess.check_output(command, cwd=repo, env=env, text=True).strip() == "disabled"
    env["AAB_WEB_SEARCH_PROVIDER"] = "parallel"
    assert subprocess.check_output(command, cwd=repo, env=env, text=True).strip() == "parallel"
