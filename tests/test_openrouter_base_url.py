"""OPENROUTER_BASE_URL env var must reroute all OpenRouter URLs (audit §4.3)."""
import importlib


def test_default_urls_unchanged(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    from backend import config, openrouter_client
    importlib.reload(config)
    importlib.reload(openrouter_client)
    assert config.OPENROUTER_API_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert openrouter_client.OPENROUTER_MODELS_URL == "https://openrouter.ai/api/v1/models"


def test_override_reroutes_all_urls(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://relay.example.com/api/v1/")
    from backend import config, openrouter_client
    importlib.reload(config)
    importlib.reload(openrouter_client)
    assert config.OPENROUTER_API_URL == "https://relay.example.com/api/v1/chat/completions"
    assert openrouter_client.OPENROUTER_MODELS_URL == "https://relay.example.com/api/v1/models"
    assert openrouter_client.OPENROUTER_ZDR_ENDPOINTS_URL == "https://relay.example.com/api/v1/endpoints/zdr"
    # restore for other tests in the session
    monkeypatch.delenv("OPENROUTER_BASE_URL")
    importlib.reload(config)
    importlib.reload(openrouter_client)
