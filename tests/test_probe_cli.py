import importlib

import pytest


def _cli():
    return importlib.import_module("scripts.probe_reasoning_capabilities")


@pytest.mark.parametrize("argv", [[], ["--max-cost-per-call", "0.05"], ["--resolve-endpoint-prices"]])
def test_cli_always_requires_a_total_ceiling(argv):
    """--max-probe-usd is a required argparse arg -- no bounding method substitutes
    for an explicit authorized total, so a probe can never run without one."""
    cli = _cli()
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_cli_requires_a_per_call_bounding_method(monkeypatch):
    """A ceiling alone is not enough: without a per-call bound there is nothing to
    multiply into a worst case, so the sweep would be effectively unbounded."""
    cli = _cli()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert cli.main(["--max-probe-usd", "5"]) == 2


def test_cli_rejects_both_bounding_methods_at_once(monkeypatch):
    """Ambiguous which ceiling the run actually enforced -- refuse rather than
    silently letting one win."""
    cli = _cli()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert cli.main([
        "--max-probe-usd", "5", "--max-cost-per-call", "0.05", "--resolve-endpoint-prices",
    ]) == 2


def test_cli_rejects_nonpositive_ceiling(monkeypatch):
    cli = _cli()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert cli.main(["--max-probe-usd", "0", "--max-cost-per-call", "0.05"]) == 2


def test_cli_rejects_nonpositive_per_call_bound(monkeypatch):
    cli = _cli()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert cli.main(["--max-probe-usd", "5", "--max-cost-per-call", "0"]) == 2


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_cli_rejects_nonpositive_concurrency(monkeypatch, bad):
    """concurrency < 1 would create Semaphore(0) and hang; reject before any call."""
    cli = _cli()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert cli.main(["--max-probe-usd", "5", "--max-cost-per-call", "0.05", "--concurrency", bad]) == 2


def test_cli_refuses_without_api_key(monkeypatch):
    cli = _cli()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert cli.main(["--max-probe-usd", "5", "--max-cost-per-call", "0.05"]) == 1


def test_resolve_provider_tag_defaults_to_id_prefix():
    cli = _cli()
    assert cli.resolve_provider_tag({"id": "openai/gpt-x"}) == "openai"
    assert cli.resolve_provider_tag({"id": "anthropic/claude"}, None) == "anthropic"
    # explicit override pins one tag for all models
    assert cli.resolve_provider_tag({"id": "openai/gpt-x"}, "azure") == "azure"
