import importlib
import json

import httpx
import pytest


def _probe():
    return importlib.import_module("backend.reasoning_probe")


# --- reasoning_signal: reliability-ordered detection -------------------------

def test_signal_prefers_reasoning_tokens():
    probe = _probe()
    resp = {"usage": {"completion_tokens": 30,
                      "completion_tokens_details": {"reasoning_tokens": 42}}}
    assert probe.reasoning_signal(resp) == 42


def test_signal_falls_back_to_reasoning_text():
    probe = _probe()
    resp = {"choices": [{"message": {"content": "ans", "reasoning": "because..."}}],
            "usage": {"completion_tokens": 5}}
    # no token count, but reasoning text present -> observed (>=1)
    assert probe.reasoning_signal(resp) >= 1


def test_signal_detects_billed_vs_visible_gap():
    probe = _probe()
    resp = {"choices": [{"message": {"content": "five"}}],  # 1 visible token
            "usage": {"completion_tokens": 400}}            # huge hidden gap
    assert probe.reasoning_signal(resp) > 0


def test_signal_zero_when_no_reasoning():
    probe = _probe()
    resp = {"choices": [{"message": {"content": "the ball is five cents"}}],
            "usage": {"completion_tokens": 6}}
    assert probe.reasoning_signal(resp) == 0


# --- classify_capability: surface from signals ------------------------------

def test_classify_levels_when_signal_rises():
    probe = _probe()
    rec = probe.classify_capability("m/1", "fp", "openai", 0,
                                    {"low": 10, "medium": 20, "high": 30})
    assert rec["control_surface"] == "levels"
    assert rec["varies_effort"] is True
    assert rec["supports_reasoning"] is True
    assert rec["levels"] == ["low", "medium", "high"]


def test_classify_onoff_when_flat_nonzero():
    probe = _probe()
    rec = probe.classify_capability("m/1", "fp", "openai", 0,
                                    {"low": 20, "medium": 20, "high": 20})
    assert rec["control_surface"] == "onoff"
    assert rec["varies_effort"] is False
    assert rec["supports_reasoning"] is True


def test_classify_none_when_no_signal_anywhere():
    probe = _probe()
    rec = probe.classify_capability("m/1", "fp", "openai", 0,
                                    {"low": 0, "medium": 0, "high": 0})
    # honesty guard: no observed reasoning -> none, NOT supported
    assert rec["control_surface"] == "none"
    assert rec["supports_reasoning"] is False
    assert rec["varies_effort"] is None


def test_classify_records_native_default_on_from_baseline():
    probe = _probe()
    on = probe.classify_capability("m/1", "fp", "openai", 15, {"low": 15, "medium": 15, "high": 15})
    off = probe.classify_capability("m/2", "fp", "openai", 0, {"low": 10, "medium": 20, "high": 30})
    assert on["native_default_on"] is True   # reasoned with no effort sent
    assert off["native_default_on"] is False
    assert on["provider_pinned"] == "openai"


# --- probe_model: end-to-end through an injected MockTransport ---------------

def _mock_transport(signals_by_effort):
    """Return httpx.MockTransport that scripts reasoning_tokens by the request's
    reasoning.effort (None = the baseline call)."""
    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        # pin is enforced client-side; assert the probe sent it
        assert body["provider"] == {"order": ["openai"], "allow_fallbacks": False}
        effort = (body.get("reasoning") or {}).get("effort")
        rt = signals_by_effort.get(effort, 0)
        payload = {
            "choices": [{"message": {"content": "The ball costs $0.05."}}],
            "usage": {"completion_tokens": 20,
                      "completion_tokens_details": {"reasoning_tokens": rt}},
        }
        return httpx.Response(200, json=payload)
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_probe_model_classifies_levels_via_mock_transport():
    probe = _probe()
    entry = {"id": "openai/gpt-x", "supports_reasoning": True, "reasoning_extraction": "field"}
    tx = _mock_transport({None: 0, "low": 10, "medium": 25, "high": 40})
    rec = await probe.probe_model(entry, "openai", "test-key", transport=tx)
    assert rec["model_id"] == "openai/gpt-x"
    assert rec["control_surface"] == "levels"
    assert rec["probed"] is True
    assert rec["provider_pinned"] == "openai"
    assert rec["probed_at"]


@pytest.mark.asyncio
async def test_probe_model_classifies_none_for_nonreasoning_model():
    probe = _probe()
    entry = {"id": "x/plain"}
    tx = _mock_transport({None: 0, "low": 0, "medium": 0, "high": 0})
    rec = await probe.probe_model(entry, "openai", "test-key", transport=tx)
    assert rec["control_surface"] == "none"
    assert rec["supports_reasoning"] is False


@pytest.mark.asyncio
async def test_probe_model_never_hits_network_without_transport(monkeypatch):
    """Sanity: the module ships with no real transport, so nothing probes on import
    and a paid run requires an explicit, injected/real transport + key (ceiling-
    gated). Here we confirm the injected mock is honored, not the network."""
    probe = _probe()
    assert probe._probe_transport is None
    entry = {"id": "x/y"}
    tx = _mock_transport({None: 5, "low": 5, "medium": 5, "high": 5})
    rec = await probe.probe_model(entry, "openai", "k", transport=tx)
    assert rec["native_default_on"] is True and rec["control_surface"] == "onoff"
