"""OPENROUTER_BASE_URL env var must reroute all OpenRouter URLs (audit §4.3)."""
import importlib


def _reload_url_modules(monkeypatch):
    """Reload config + URL consumers with dotenv disabled, so a developer's
    local .env (which config re-reads on reload) cannot leak into the test."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    from backend import config, endpoint_pricing, openrouter_client, openrouter_pdf
    importlib.reload(config)
    importlib.reload(openrouter_client)
    importlib.reload(openrouter_pdf)
    importlib.reload(endpoint_pricing)
    return config, openrouter_client, openrouter_pdf, endpoint_pricing


def _restore(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    _reload_url_modules(monkeypatch)


DEFAULT_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
RELAY_ENDPOINTS_URL = "https://relay.example.com/api/v1/models/{model}/endpoints"


def test_default_urls_unchanged(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    config, openrouter_client, openrouter_pdf, endpoint_pricing = _reload_url_modules(monkeypatch)
    assert config.OPENROUTER_API_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert openrouter_client.OPENROUTER_MODELS_URL == "https://openrouter.ai/api/v1/models"
    assert openrouter_pdf.OPENROUTER_URL == "https://openrouter.ai/api/v1/chat/completions"
    # The prices that BOUND paid probe spend must come from the same service that
    # bills the probe calls.
    assert endpoint_pricing.OPENROUTER_ENDPOINTS_URL == DEFAULT_ENDPOINTS_URL


def test_override_reroutes_all_urls(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://relay.example.com/api/v1/")
    config, openrouter_client, openrouter_pdf, endpoint_pricing = _reload_url_modules(monkeypatch)
    assert config.OPENROUTER_API_URL == "https://relay.example.com/api/v1/chat/completions"
    assert openrouter_client.OPENROUTER_MODELS_URL == "https://relay.example.com/api/v1/models"
    assert openrouter_client.OPENROUTER_ZDR_ENDPOINTS_URL == "https://relay.example.com/api/v1/endpoints/zdr"
    assert openrouter_pdf.OPENROUTER_URL == "https://relay.example.com/api/v1/chat/completions"
    # Pricing a relay's calls against openrouter.ai would bound the wrong service.
    assert endpoint_pricing.OPENROUTER_ENDPOINTS_URL == RELAY_ENDPOINTS_URL
    # restore for other tests in the session
    _restore(monkeypatch)


def test_blank_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "   ")
    config, openrouter_client, _openrouter_pdf, endpoint_pricing = _reload_url_modules(monkeypatch)
    assert config.OPENROUTER_API_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert openrouter_client.OPENROUTER_MODELS_URL == "https://openrouter.ai/api/v1/models"
    assert endpoint_pricing.OPENROUTER_ENDPOINTS_URL == DEFAULT_ENDPOINTS_URL
    # restore for other tests in the session
    _restore(monkeypatch)
