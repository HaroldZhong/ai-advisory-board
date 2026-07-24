import importlib

import pytest


def _cli():
    return importlib.import_module("scripts.probe_reasoning_capabilities")


def test_cli_requires_a_spend_ceiling():
    """No --max-probe-usd -> argparse refuses (the CLI-level spend guard). A probe
    can never run without an explicit ceiling."""
    cli = _cli()
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_rejects_nonpositive_ceiling(monkeypatch):
    cli = _cli()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert cli.main(["--max-probe-usd", "0"]) == 2


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_cli_rejects_nonpositive_concurrency(monkeypatch, bad):
    """concurrency < 1 would create Semaphore(0) and hang; reject before any call."""
    cli = _cli()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert cli.main(["--max-probe-usd", "5", "--concurrency", bad]) == 2


def test_cli_refuses_without_api_key(monkeypatch):
    cli = _cli()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert cli.main(["--max-probe-usd", "5"]) == 1


def test_resolve_provider_tag_defaults_to_id_prefix():
    cli = _cli()
    assert cli.resolve_provider_tag({"id": "openai/gpt-x"}) == "openai"
    assert cli.resolve_provider_tag({"id": "anthropic/claude"}, None) == "anthropic"
    # explicit override pins one tag for all models
    assert cli.resolve_provider_tag({"id": "openai/gpt-x"}, "azure") == "azure"
