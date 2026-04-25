import importlib
import sys

import pytest
from starlette.requests import Request


def unload_backend_modules(*module_names):
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def import_without_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    unload_backend_modules(
        "backend.config",
        "backend.openrouter",
        "backend.file_processing",
        "backend.openrouter_pdf",
        "backend.main",
    )
    return importlib.import_module(module_name)


def make_request(host="127.0.0.1"):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/config/setup",
            "headers": [],
            "client": (host, 4321),
        }
    )


def test_config_imports_without_api_key(monkeypatch):
    config = import_without_api_key(monkeypatch, "backend.config")

    assert config.get_openrouter_api_key() is None
    assert config.has_openrouter_api_key() is False


@pytest.mark.asyncio
async def test_setup_config_updates_runtime_key_and_env_file(monkeypatch, tmp_path):
    main = import_without_api_key(monkeypatch, "backend.main")
    env_path = tmp_path / ".env"

    monkeypatch.setattr(main.config, "ENV_PATH", env_path)

    result = await main.setup_config({"api_key": " new-key "}, request=make_request())

    assert result == {"success": True}
    assert main.config.get_openrouter_api_key() == "new-key"
    assert main.config.OPENROUTER_API_KEY == "new-key"
    assert env_path.read_text() == "OPENROUTER_API_KEY=new-key\n"


@pytest.mark.asyncio
async def test_setup_config_rejects_non_local_clients(monkeypatch, tmp_path):
    main = import_without_api_key(monkeypatch, "backend.main")
    monkeypatch.setattr(main.config, "ENV_PATH", tmp_path / ".env")

    with pytest.raises(main.HTTPException) as exc:
        await main.setup_config({"api_key": "new-key"}, request=make_request("203.0.113.5"))

    assert exc.value.status_code == 403
    assert main.config.get_openrouter_api_key() is None


@pytest.mark.asyncio
async def test_openrouter_reads_api_key_at_request_time(monkeypatch):
    openrouter = import_without_api_key(monkeypatch, "backend.openrouter")
    captured_headers = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            captured_headers.append(kwargs["headers"])
            return FakeResponse()

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", FakeAsyncClient)

    assert await openrouter.query_model("model-a", [{"role": "user", "content": "hi"}]) is None

    monkeypatch.setenv("OPENROUTER_API_KEY", "runtime-key")
    result = await openrouter.query_model("model-a", [{"role": "user", "content": "hi"}])

    assert result["content"] == "ok"
    assert captured_headers == [{"Authorization": "Bearer runtime-key", "Content-Type": "application/json"}]


@pytest.mark.asyncio
async def test_openrouter_pdf_reads_api_key_at_request_time(monkeypatch):
    openrouter_pdf = import_without_api_key(monkeypatch, "backend.openrouter_pdf")
    captured_headers = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "Extracted PDF text"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            captured_headers.append(kwargs["headers"])
            return FakeResponse()

    missing_key_result = await openrouter_pdf.extract_pdf_with_openrouter(
        b"%PDF-1.4", "test.pdf"
    )

    assert missing_key_result["status"] == "failed"
    assert captured_headers == []

    monkeypatch.setenv("OPENROUTER_API_KEY", "runtime-pdf-key")
    monkeypatch.setattr(openrouter_pdf.httpx, "AsyncClient", FakeAsyncClient)

    result = await openrouter_pdf.extract_pdf_with_openrouter(b"%PDF-1.4", "test.pdf")

    assert result["status"] == "success"
    assert result["text"] == "Extracted PDF text"
    assert captured_headers[0]["Authorization"] == "Bearer runtime-pdf-key"
