"""Distinguish network / auth / quota / timeout failures (audit §4.2, §4.3)."""
import asyncio
import httpx
import pytest
from backend.openrouter import classify_openrouter_error, connect_timeout_for


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError("boom", request=req, response=resp)


@pytest.mark.parametrize("exc,expected", [
    (httpx.ConnectError("dns fail"), "network"),
    (httpx.ConnectTimeout("connect timeout"), "network"),
    (httpx.ReadError("connection reset"), "network"),
    (httpx.WriteError("broken pipe"), "network"),
    (httpx.ProxyError("tunnel failed"), "network"),
    (httpx.ReadTimeout("read timeout"), "timeout"),
    (asyncio.TimeoutError(), "timeout"),
    (_status_error(401), "auth"),
    (_status_error(403), "other"),  # moderation/guardrail per OpenRouter docs, not credentials
    (_status_error(402), "quota"),
    (_status_error(408), "timeout"),
    (_status_error(500), "other"),
    (ValueError("weird"), "other"),
])
def test_classify_openrouter_error(exc, expected):
    assert classify_openrouter_error(exc) == expected


def test_connect_deadline_stays_below_wall_clock():
    assert connect_timeout_for(120.0) == 10.0
    assert connect_timeout_for(10.0) == 5.0  # ties would race asyncio.wait_for
