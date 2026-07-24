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


def test_signal_ignores_billed_vs_visible_gap():
    probe = _probe()
    # Honesty guard: a large billed-vs-visible token gap, with NO reasoning_tokens
    # and NO message.reasoning, is NOT treated as reasoning. Comparing billed tokens
    # to visible words has no reliable unit, so verbose ordinary output must not be
    # misread as hidden reasoning.
    terse = {"choices": [{"message": {"content": "five"}}], "usage": {"completion_tokens": 400}}
    verbose = {"choices": [{"message": {"content": " ".join(["word"] * 200)}}],
               "usage": {"completion_tokens": 260}}  # 260 tokens vs 200 words -> old false positive
    assert probe.reasoning_signal(terse) == 0
    assert probe.reasoning_signal(verbose) == 0


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


# --- A4: sweep orchestration (resumable, ceiling-guarded) --------------------

def _entry(mid, extraction="field"):
    return {"id": mid, "supports_reasoning": True, "reasoning_extraction": extraction}


def test_models_needing_probe_skips_fresh_includes_stale_and_unprobed():
    probe = _probe()
    a, b, c = _entry("a"), _entry("b"), _entry("c")
    existing = {
        "a": {"model_id": "a", "probed": True, "fingerprint": probe.model_fingerprint(a)},   # fresh
        "b": {"model_id": "b", "probed": True, "fingerprint": "STALE"},                        # stale
        # c absent -> unprobed
    }
    needing = {m["id"] for m in probe.models_needing_probe([a, b, c], existing)}
    assert needing == {"b", "c"}


def test_estimate_max_probe_cost():
    probe = _probe()
    # 2 models x (1 baseline + 3 levels) x $0.01 = $0.08
    assert probe.estimate_max_probe_cost(2, ("low", "medium", "high"), 0.01) == pytest.approx(0.08)


@pytest.mark.asyncio
async def test_sweep_refuses_without_a_ceiling():
    probe = _probe()
    with pytest.raises(probe.CeilingError, match="unset"):
        await probe.run_probe_sweep(
            [_entry("a")], lambda m: "openai", "k",
            max_probe_usd=None, max_cost_per_call_usd=0.01,
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_cost", [0, -0.01])
async def test_sweep_refuses_when_per_call_cost_not_positive(bad_cost):
    probe = _probe()
    # A non-positive per-call cost zeroes the worst-case estimate; without this
    # guard `0 > max_probe_usd` is False and the full paid sweep would fire.
    with pytest.raises(probe.CeilingError, match="max_cost_per_call_usd"):
        await probe.run_probe_sweep(
            [_entry("a"), _entry("b")], lambda m: "openai", "k",
            max_probe_usd=5.0, max_cost_per_call_usd=bad_cost,
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )


@pytest.mark.asyncio
async def test_probe_sends_bounded_max_tokens():
    probe = _probe()
    seen = []

    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        seen.append(body.get("max_tokens"))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 3}},
        })

    await probe.probe_model(_entry("m/x"), "openai", "k", transport=httpx.MockTransport(handler))
    # every probe call (baseline + each level) bounds output so per-call cost is finite
    assert seen and all(mt == probe.PROBE_MAX_TOKENS for mt in seen)


@pytest.mark.asyncio
async def test_sweep_refuses_when_worst_case_exceeds_ceiling():
    probe = _probe()
    with pytest.raises(probe.CeilingError, match="exceeds the authorized ceiling"):
        # 3 models x 4 calls x $0.10 = $1.20 worst case, ceiling $0.50
        await probe.run_probe_sweep(
            [_entry("a"), _entry("b"), _entry("c")], lambda m: "openai", "k",
            max_probe_usd=0.50, max_cost_per_call_usd=0.10,
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )


@pytest.mark.asyncio
async def test_sweep_probes_within_ceiling_and_merges():
    probe = _probe()
    tx = _mock_transport({None: 0, "low": 10, "medium": 20, "high": 30})
    merged = await probe.run_probe_sweep(
        [_entry("a"), _entry("b")], lambda m: "openai", "k",
        max_probe_usd=1.00, max_cost_per_call_usd=0.01, transport=tx,
    )
    assert merged["a"]["control_surface"] == "levels"
    assert merged["b"]["control_surface"] == "levels"


@pytest.mark.asyncio
async def test_sweep_is_resumable_skips_fresh_rows():
    probe = _probe()
    a, b = _entry("a"), _entry("b")
    existing = {"a": {"model_id": "a", "probed": True, "fingerprint": probe.model_fingerprint(a),
                      "control_surface": "onoff"}}
    tx = _mock_transport({None: 0, "low": 10, "medium": 20, "high": 30})
    merged = await probe.run_probe_sweep(
        [a, b], lambda m: "openai", "k",
        max_probe_usd=1.00, max_cost_per_call_usd=0.01, existing=existing, transport=tx,
    )
    # a is fresh -> untouched; b is newly probed
    assert merged["a"]["control_surface"] == "onoff"
    assert merged["b"]["control_surface"] == "levels"


@pytest.mark.asyncio
async def test_sweep_degrades_per_model_on_error():
    probe = _probe()

    def flaky_transport(request):
        import json as _json
        body = _json.loads(request.content.decode("utf-8"))
        if body["model"] == "bad":
            return httpx.Response(500, json={"error": "boom"})
        effort = (body.get("reasoning") or {}).get("effort")
        rt = {None: 0, "low": 10, "medium": 20, "high": 30}.get(effort, 0)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": rt}},
        })

    merged = await probe.run_probe_sweep(
        [_entry("bad"), _entry("good")], lambda m: "openai", "k",
        max_probe_usd=1.00, max_cost_per_call_usd=0.01,
        transport=httpx.MockTransport(flaky_transport),
    )
    # one failing model does not abort the sweep; the good one still lands
    assert "bad" not in merged
    assert merged["good"]["control_surface"] == "levels"
