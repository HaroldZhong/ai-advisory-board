import asyncio
import importlib
import time

import pytest


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_query_model_enforces_wall_clock_timeout(monkeypatch):
    openrouter = import_module_with_api_key(monkeypatch, "backend.openrouter")

    class HangingAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            await asyncio.sleep(1)
            raise AssertionError("query_model did not cancel the hanging request")

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", HangingAsyncClient)

    started = time.monotonic()
    result = await openrouter.query_model(
        "anthropic/claude-opus-4.7",
        [{"role": "user", "content": "hi"}],
        timeout=0.01,
    )

    assert result is None
    assert time.monotonic() - started < 0.5
