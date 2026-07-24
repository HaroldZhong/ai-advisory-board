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


def test_signal_explicit_zero_reasoning_tokens_is_definitive():
    probe = _probe()
    # reasoning_tokens explicitly 0 is authoritative -> 0, even with a long visible
    # answer that a fall-through heuristic might otherwise misread.
    resp = {"choices": [{"message": {"content": " ".join(["w"] * 300)}}],
            "usage": {"completion_tokens": 400, "completion_tokens_details": {"reasoning_tokens": 0}}}
    assert probe.reasoning_signal(resp) == 0


def test_signal_counts_think_tags_for_tags_mode_only():
    probe = _probe()
    # tags-mode model: reasoning arrives as a visible <think> block, no reasoning_tokens
    resp = {"choices": [{"message": {"content": "<think>8-3=5, remove 3</think> The ball is $0.05."}}],
            "usage": {"completion_tokens": 20}}
    assert probe.reasoning_signal(resp, "tags") >= 1
    # gated exactly like the runtime: field-mode / unknown must NOT count the markup
    assert probe.reasoning_signal(resp, "field") == 0
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


def test_classify_honors_only_the_probed_levels():
    probe = _probe()
    # one-level custom sweep: a high signal must NOT be credited as low/medium too
    one = probe.classify_capability("m/1", "fp", "openai", 0, {"high": 30}, levels=("high",))
    assert one["supports_reasoning"] is True and one["control_surface"] == "onoff"
    # onoff surface claims no specific level ladder -> must NOT phantom-fill low/medium/high
    assert one.get("levels") is None
    # two-level sweep must differentiate on those two, not be downgraded by a
    # phantom unprobed 'high' treated as 0
    two = probe.classify_capability("m/1", "fp", "openai", 0, {"low": 10, "medium": 25}, levels=("low", "medium"))
    assert two["control_surface"] == "levels"
    assert two["levels"] == ["low", "medium"]


def test_classify_excludes_zero_signal_levels_from_the_ladder():
    probe = _probe()
    # low produced NO reasoning; only medium/high did -> low must not be advertised
    # as a supported level (else snap could route users to a no-reasoning setting).
    rec = probe.classify_capability("m/1", "fp", "openai", 0, {"low": 0, "medium": 20, "high": 30})
    assert rec["control_surface"] == "levels"
    assert rec["levels"] == ["medium", "high"]


def test_plain_reflects_baseline_only_not_effort_reasoning():
    probe = _probe()
    # baseline (no-effort) did NOT reason but an effort level did -> plain='none'
    effort_only = probe.classify_capability("m/1", "fp", "openai", 0, {"low": 0, "medium": 20, "high": 30})
    assert effort_only["plain"] == "none" and effort_only["supports_reasoning"] is True
    # baseline reasoned -> plain='reasoned'
    baseline_on = probe.classify_capability("m/2", "fp", "openai", 12, {"low": 12, "medium": 12, "high": 12})
    assert baseline_on["plain"] == "reasoned"


# --- probe_model: end-to-end through an injected MockTransport ---------------

def _mock_transport(signals_by_effort, served_provider=None):
    """Return httpx.MockTransport that scripts reasoning_tokens by the request's
    reasoning.effort (None = the baseline call). Optionally reports which provider
    served, the way router metadata does."""
    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        # The sweep probes UNPINNED so the capability is learned on the endpoint
        # OpenRouter actually routes to. A pin here would send `allow_fallbacks:
        # false` with a tag nobody verified -- which meant NO ROUTE for 17 of 33
        # registry models under the old id-prefix guess.
        assert "provider" not in body, f"probe must not pin a provider: {body.get('provider')!r}"
        assert body["max_tokens"] == 8000
        effort = (body.get("reasoning") or {}).get("effort")
        rt = signals_by_effort.get(effort, 0)
        payload = {
            "choices": [{"message": {"content": "The ball costs $0.05."}}],
            "usage": {"completion_tokens": 20,
                      "completion_tokens_details": {"reasoning_tokens": rt}},
        }
        if served_provider is not None:
            payload["openrouter_metadata"] = {
                "endpoints": {"available": [{"provider": served_provider,
                                             "model": "m/x", "selected": True}]}
            }
        return httpx.Response(200, json=payload)
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_probe_model_classifies_levels_via_mock_transport():
    probe = _probe()
    entry = {"id": "openai/gpt-x", "supports_reasoning": True, "reasoning_extraction": "field"}
    tx = _mock_transport({None: 0, "low": 10, "medium": 25, "high": 40},
                         served_provider="OpenAI")
    rec = await probe.probe_model(
        entry, "test-key", transport=tx,
        get=_endpoints_getter({"openai/gpt-x": [("openai", 0.000001, 0.000002)]}),
    )
    assert rec["model_id"] == "openai/gpt-x"
    assert rec["control_surface"] == "levels"
    assert rec["probed"] is True
    # The pin is RESOLVED FROM THE PROVIDER THAT SERVED (display name "OpenAI" ->
    # the model's single endpoint with that provider_name), never requested up front.
    assert rec["provider_pinned"] == "openai"
    assert rec["probed_at"]


@pytest.mark.asyncio
async def test_probe_model_classifies_none_for_nonreasoning_model():
    probe = _probe()
    entry = {"id": "x/plain"}
    tx = _mock_transport({None: 0, "low": 0, "medium": 0, "high": 0})
    rec = await probe.probe_model(entry, "test-key", transport=tx)
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
    rec = await probe.probe_model(entry, "k", transport=tx)
    assert rec["native_default_on"] is True and rec["control_surface"] == "onoff"


@pytest.mark.asyncio
async def test_probe_posts_to_configured_base_url(monkeypatch):
    probe = _probe()
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                         "usage": {"completion_tokens": 5,
                                                   "completion_tokens_details": {"reasoning_tokens": 3}}})

    monkeypatch.setattr(probe.config, "OPENROUTER_API_URL", "https://relay.example/api/v1/chat/completions")
    await probe.probe_model(_entry("m/x"), "k", transport=httpx.MockTransport(handler))
    # respects the configured base URL, not a hard-coded openrouter.ai
    assert seen and all(u == "https://relay.example/api/v1/chat/completions" for u in seen)


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
    # num_models x (1 baseline + 3 levels) x the asserted per-call bound
    assert probe.estimate_max_probe_cost(2, ("low", "medium", "high"), 0.01) == pytest.approx(0.08)


@pytest.mark.asyncio
async def test_sweep_refuses_without_a_ceiling():
    probe = _probe()
    with pytest.raises(probe.CeilingError, match="unset"):
        await probe.run_probe_sweep(
            [_entry("a")], "k", max_probe_usd=None, max_cost_per_call_usd=0.01,
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
async def test_sweep_refuses_nonpositive_or_nonfinite_ceiling(bad):
    probe = _probe()
    # 0/negative AND NaN/inf must all be rejected: `not (x > 0)` alone would let NaN
    # slip through the `worst_case > ceiling` comparison and fire the paid sweep.
    with pytest.raises(probe.CeilingError, match="positive, finite"):
        await probe.run_probe_sweep(
            [_entry("a")], "k", max_probe_usd=bad, max_cost_per_call_usd=0.01,
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
async def test_sweep_refuses_bad_per_call_bound(bad):
    probe = _probe()
    # The per-call bound must be a real positive, finite number: a 0/negative/NaN
    # bound would zero-or-poison the worst-case estimate and defeat the ceiling.
    with pytest.raises(probe.CeilingError, match="per-call cost bound"):
        await probe.run_probe_sweep(
            [_entry("a")], "k", max_probe_usd=100.0, max_cost_per_call_usd=bad,
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )


@pytest.mark.asyncio
async def test_sweep_rejects_zero_concurrency():
    probe = _probe()
    # Semaphore(0) starts locked -> the sweep would hang forever.
    with pytest.raises(ValueError, match="concurrency"):
        await probe.run_probe_sweep(
            [_entry("a")], "k", max_probe_usd=100.0, max_cost_per_call_usd=0.01, concurrency=0,
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )


@pytest.mark.asyncio
async def test_sweep_fails_loud_when_every_probe_fails():
    probe = _probe()
    # Systemic failure (e.g. invalid key) fails every call -> abort, don't publish
    # an all-skipped matrix and exit 0.
    def all_fail(request):
        return httpx.Response(401, json={"error": "invalid key"})
    with pytest.raises(RuntimeError, match="every call failed"):
        await probe.run_probe_sweep(
            [_entry("a"), _entry("b")], "k", max_probe_usd=100.0, max_cost_per_call_usd=0.01,
            transport=httpx.MockTransport(all_fail),
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

    await probe.probe_model(_entry("m/x"), "k", transport=httpx.MockTransport(handler))
    # every probe call (baseline + each level) bounds output so per-call cost is finite
    assert seen and all(mt == probe.PROBE_MAX_TOKENS for mt in seen)


@pytest.mark.asyncio
async def test_sweep_refuses_when_worst_case_exceeds_ceiling():
    probe = _probe()
    with pytest.raises(probe.CeilingError, match="exceeds the authorized ceiling"):
        # 3 models x 4 calls x $0.10 = $1.20 worst case, ceiling $0.50
        await probe.run_probe_sweep(
            [_entry("a"), _entry("b"), _entry("c")],
            "k", max_probe_usd=0.50, max_cost_per_call_usd=0.10,
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )


@pytest.mark.asyncio
async def test_sweep_probes_within_ceiling_and_merges():
    probe = _probe()
    tx = _mock_transport({None: 0, "low": 10, "medium": 20, "high": 30})
    merged = await probe.run_probe_sweep(
        [_entry("a"), _entry("b")], "k",
        max_probe_usd=1.00, max_cost_per_call_usd=0.01, transport=tx,
    )
    assert merged["a"]["control_surface"] == "levels"
    assert merged["b"]["control_surface"] == "levels"


@pytest.mark.asyncio
async def test_sweep_is_resumable_skips_fresh_rows():
    probe = _probe()
    a, b = _entry("a"), _entry("b")
    existing = {"a": {"model_id": "a", "probed": True, "fingerprint": probe.model_fingerprint(a),
                      "provider_pinned": "openai", "control_surface": "onoff"}}
    tx = _mock_transport({None: 0, "low": 10, "medium": 20, "high": 30})
    merged = await probe.run_probe_sweep(
        [a, b], "k",
        max_probe_usd=1.00, max_cost_per_call_usd=0.01, existing=existing, transport=tx,
    )
    # a is fresh (fingerprint + provider match) -> untouched; b is newly probed
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
        [_entry("bad"), _entry("good")], "k",
        max_probe_usd=1.00, max_cost_per_call_usd=0.01,
        transport=httpx.MockTransport(flaky_transport),
    )
    # one failing model does not abort the sweep; the good one still lands
    assert "bad" not in merged
    assert merged["good"]["control_surface"] == "levels"


@pytest.mark.asyncio
async def test_failed_reprobe_replaces_stale_row_with_unknown():
    probe = _probe()
    a, b = _entry("a"), _entry("b")
    # 'a' has a fresh fingerprint but was measured for a DIFFERENT provider -> the
    # A STALE FINGERPRINT forces the re-probe. (Provider-change invalidation is gone:
    # provider_pinned is now an observation of what served, not a requested tag, so
    # there is no requested value to compare a stored row against.) Make only 'a' fail.
    existing = {"a": {"model_id": "a", "probed": True, "fingerprint": "STALE",
                      "provider_pinned": "openai", "control_surface": "levels"}}

    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        if body["model"] == "a":
            return httpx.Response(500, json={"error": "boom"})
        rt = {None: 0, "low": 10, "medium": 20, "high": 30}.get((body.get("reasoning") or {}).get("effort"), 0)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                         "usage": {"completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": rt}}})

    merged = await probe.run_probe_sweep(
        [a, b], "k", max_probe_usd=100.0, max_cost_per_call_usd=0.01, existing=existing,
        transport=httpx.MockTransport(handler),
    )
    # the stale openai-measured row must NOT survive as authoritative -> unknown
    assert merged["a"]["control_surface"] == "unknown"
    assert not merged["a"].get("probed")
    assert merged["b"]["control_surface"] == "levels"


# --- per-model spend bounds (resolved from each model's OWN pinned endpoint) --
#
# A single uniform --max-cost-per-call must cover the priciest model in the sweep,
# so one expensive outlier inflates the required ceiling for every other call. These
# pin the tighter, still-sound alternative: price each call from its own endpoint.

def _endpoints_getter(pricing_by_model):
    """Fake the PUBLIC /endpoints API. `pricing_by_model` maps model id -> list of
    (tag, prompt_per_token, completion_per_token)."""
    def get(url, timeout=None):
        model = url.split("/models/", 1)[1].rsplit("/endpoints", 1)[0]
        endpoints = [
            {"tag": tag, "provider_name": tag, "name": tag, "model_id": model,
             "pricing": {"prompt": str(p), "completion": str(c)}}
            for tag, p, c in pricing_by_model.get(model, [])
        ]
        return httpx.Response(
            200, json={"data": {"endpoints": endpoints}},
            request=httpx.Request("GET", url),
        )
    return get


def test_probe_call_bound_prices_output_at_the_cap():
    probe = _probe()
    # A probe call cannot bill more than PROBE_MAX_TOKENS of output, so pricing the
    # output at the cap (plus the fixed prompt) bounds the route. With no published
    # internal_reasoning rate the surcharge is assumed equal to the completion rate,
    # so the capped output is charged twice.
    bound = probe.probe_call_bound_usd(
        {"prompt_per_token": 0.0000005, "completion_per_token": 0.000001}
    )
    expected = (
        2 * 0.000001 * probe.PROBE_MAX_TOKENS
        + 0.0000005 * probe.PROBE_PROMPT_TOKENS_EST
    )
    assert bound == pytest.approx(expected)


def test_resolve_probe_call_bounds_prices_each_model_by_its_own_endpoint():
    probe = _probe()
    get = _endpoints_getter({
        "cheap/model": [("cheap", 0.0000001, 0.0000002)],
        "pricey/model": [("pricey", 0.000002, 0.00018)],
    })
    bounds = probe.resolve_probe_call_bounds(
        [_entry("cheap/model"), _entry("pricey/model")], get=get,
    )
    # No published internal_reasoning rate -> assumed surcharge == completion rate.
    assert bounds["cheap/model"] == pytest.approx(
        2 * 0.0000002 * probe.PROBE_MAX_TOKENS + 0.0000001 * probe.PROBE_PROMPT_TOKENS_EST)
    assert bounds["pricey/model"] == pytest.approx(
        2 * 0.00018 * probe.PROBE_MAX_TOKENS + 0.000002 * probe.PROBE_PROMPT_TOKENS_EST)
    # The whole point: the models do NOT share one bound.
    assert bounds["pricey/model"] > bounds["cheap/model"] * 100


def test_resolve_probe_call_bounds_refuses_a_model_with_no_priceable_endpoint():
    probe = _probe()
    # Dropping an unpriceable model from the ceiling math would under-count the spend
    # it is still about to make, so this refuses loudly and names the model.
    get = _endpoints_getter({"a/model": []})
    with pytest.raises(probe.CeilingError, match="a/model"):
        probe.resolve_probe_call_bounds([_entry("a/model")], get=get)


def test_estimate_max_probe_cost_per_model_sums_each_models_own_bound():
    probe = _probe()
    # (1 baseline + 3 levels) calls against each model's own per-call bound.
    total = probe.estimate_max_probe_cost_per_model(
        ["a", "b"], ("low", "medium", "high"), {"a": 0.01, "b": 0.25},
    )
    assert total == pytest.approx((0.01 + 0.25) * 4)


def test_per_model_ceiling_is_far_tighter_than_a_uniform_bound():
    probe = _probe()
    # THE MOTIVATING DEFECT: with one expensive outlier, a uniform bound must be set
    # to the outlier's per-call cost and applied to EVERY call, so the ceiling the
    # maintainer must authorize balloons even though the sweep cannot spend it.
    bounds = {"cheap": 0.008, "outlier": 1.44}
    levels = ("low", "medium", "high")
    per_model = probe.estimate_max_probe_cost_per_model(bounds, levels, bounds)
    uniform = probe.estimate_max_probe_cost(len(bounds), levels, max(bounds.values()))
    assert per_model < uniform
    assert uniform > per_model * 1.5


@pytest.mark.asyncio
async def test_sweep_accepts_per_model_bounds_without_a_uniform_bound():
    probe = _probe()
    merged = await probe.run_probe_sweep(
        [_entry("a"), _entry("b")], "k",
        max_probe_usd=1.0,
        per_model_bounds={"a": 0.01, "b": 0.02},
        transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
    )
    assert merged["a"]["probed"] is True and merged["b"]["probed"] is True


@pytest.mark.asyncio
async def test_sweep_refuses_when_a_model_it_will_probe_has_no_bound():
    probe = _probe()
    # "b" would still be probed, spending against a ceiling computed without it.
    # match must not be satisfied by every CeilingError this call can raise, so
    # assert the specific guard AND the named model.
    with pytest.raises(probe.CeilingError) as excinfo:
        await probe.run_probe_sweep(
            [_entry("a"), _entry("b")], "k",
            max_probe_usd=1.0,
            per_model_bounds={"a": 0.01},
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )
    message = str(excinfo.value)
    assert "no positive, finite per-call bound" in message
    assert "(b)" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
async def test_sweep_refuses_a_nonpositive_or_nonfinite_per_model_bound(bad):
    probe = _probe()
    with pytest.raises(probe.CeilingError, match="per-call bound"):
        await probe.run_probe_sweep(
            [_entry("a")], "k",
            max_probe_usd=1.0,
            per_model_bounds={"a": bad},
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )


@pytest.mark.asyncio
async def test_sweep_refuses_when_per_model_worst_case_exceeds_the_ceiling():
    probe = _probe()
    # 2 models x 4 calls x $0.30 = $2.40 > $1.00 authorized.
    with pytest.raises(probe.CeilingError, match="exceeds the authorized"):
        await probe.run_probe_sweep(
            [_entry("a"), _entry("b")], "k",
            max_probe_usd=1.0,
            per_model_bounds={"a": 0.30, "b": 0.30},
            transport=_mock_transport({None: 0, "low": 1, "medium": 2, "high": 3}),
        )


def test_bound_includes_the_separate_reasoning_surcharge():
    probe = _probe()
    # A non-zero internal_reasoning rate is an EXPLICIT separate charge (the D1
    # billing rule). The probe deliberately provokes reasoning, so a completion-only
    # figure would NOT be an upper bound and the ceiling could be breached.
    without = probe.probe_call_bound_usd(
        {"prompt_per_token": 0.0000005, "completion_per_token": 0.000001}
    )
    with_reasoning = probe.probe_call_bound_usd(
        {"prompt_per_token": 0.0000005, "completion_per_token": 0.000001,
         "internal_reasoning_per_token": 0.000004}
    )
    assert with_reasoning > without
    assert with_reasoning == pytest.approx(
        (0.000001 + 0.000004) * probe.PROBE_MAX_TOKENS
        + 0.0000005 * probe.PROBE_PROMPT_TOKENS_EST
    )


@pytest.mark.parametrize("rate", [None, 0, "0", ""])
def test_bound_unchanged_when_reasoning_is_billed_inside_completion(rate):
    probe = _probe()
    # No explicit surcharge -> reasoning is inside the completion price; adding
    # anything here would inflate every ceiling for no reason.
    base = {"prompt_per_token": 0.0000005, "completion_per_token": 0.000001}
    assert probe.probe_call_bound_usd({**base, "internal_reasoning_per_token": rate}) == \
        pytest.approx(probe.probe_call_bound_usd(base))


def test_resolve_probe_call_bounds_refuses_off_openrouter(monkeypatch):
    probe = _probe()
    # /endpoints, endpoint tags and per-endpoint pricing are OpenRouter concepts.
    # Pricing a different service would yield a bound that doesn't describe what is
    # actually billed, so refuse rather than "resolve" a meaningless number.
    monkeypatch.setattr(probe.config, "provider_is_openrouter", lambda: False)
    monkeypatch.setattr(probe.config, "PROVIDER_KIND", "openai-compatible", raising=False)
    with pytest.raises(probe.CeilingError, match="OpenRouter"):
        probe.resolve_probe_call_bounds(
            [_entry("a/model")],
            get=_endpoints_getter({"a/model": [("a", 0.0000001, 0.0000002)]}),
        )


def test_bound_includes_flat_per_request_fees():
    probe = _probe()
    # Search-native models bill a flat fee per call: perplexity/sonar publishes
    # web_search 0.005 and searches on EVERY completion, so omitting it under-counts
    # such a call by more than half.
    base = {"prompt_per_token": 0.000001, "completion_per_token": 0.000001}
    assert probe.probe_call_bound_usd({**base, "per_request_usd": 0.005}) == pytest.approx(
        probe.probe_call_bound_usd(base) + 0.005)


def test_bound_prices_prompt_at_the_worse_of_prompt_and_cache_write():
    probe = _probe()
    # A provider that auto-caches bills the prompt as a cache WRITE.
    cheap_prompt = {"prompt_per_token": 0.000001, "completion_per_token": 0.000002,
                    "input_cache_write_per_token": 0.00001}
    assert probe.probe_call_bound_usd(cheap_prompt) == pytest.approx(
        2 * 0.000002 * probe.PROBE_MAX_TOKENS + 0.00001 * probe.PROBE_PROMPT_TOKENS_EST)


def test_bound_refuses_unaccounted_nonzero_charges():
    probe = _probe()
    # Fail CLOSED: a published charge this bound cannot model must refuse, never be
    # silently dropped from the ceiling.
    from backend.endpoint_pricing import EndpointPricingError
    with pytest.raises(EndpointPricingError, match="mystery_fee"):
        probe.probe_call_bound_usd({
            "prompt_per_token": 0.000001, "completion_per_token": 0.000001,
            "unaccounted_nonzero_price_keys": ["mystery_fee"],
        })


def test_resolver_flags_unknown_nonzero_charges_and_ignores_known_ones():
    from backend.endpoint_pricing import fetch_endpoint_pricing
    url = "https://openrouter.ai/api/v1/models/x/y/endpoints"

    def getter(pricing):
        def get(_url, timeout=None):
            return httpx.Response(200, json={"data": {"endpoints": [
                {"tag": "t", "provider_name": "T", "pricing": pricing}]}},
                request=httpx.Request("GET", url))
        return get

    known = fetch_endpoint_pricing("x/y", "t", get=getter({
        "prompt": "0.000001", "completion": "0.000002", "web_search": "0.005",
        "image": "0.1", "discount": "0.5",  # inapplicable to a text probe / not a charge
    }))
    assert known["unaccounted_nonzero_price_keys"] == []
    assert known["per_request_usd"] == pytest.approx(0.005)

    surprise = fetch_endpoint_pricing("x/y", "t", get=getter({
        "prompt": "0.000001", "completion": "0.000002", "brand_new_fee": "0.01",
    }))
    assert surprise["unaccounted_nonzero_price_keys"] == ["brand_new_fee"]
    # A zero/absent unknown rate cannot bill, so it must NOT block pricing.
    zeroed = fetch_endpoint_pricing("x/y", "t", get=getter({
        "prompt": "0.000001", "completion": "0.000002", "brand_new_fee": "0",
    }))
    assert zeroed["unaccounted_nonzero_price_keys"] == []


def test_bound_assumes_a_reasoning_surcharge_when_none_is_published():
    probe = _probe()
    # An ABSENT internal_reasoning rate is not evidence reasoning is free:
    # classify_billing detects separate billing on endpoints publishing no such rate,
    # as a surcharge of reasoning_tokens x completion_rate on TOP of completion. The
    # ceiling must not assume the favourable regime.
    bound = probe.probe_call_bound_usd(
        {"prompt_per_token": 0.0000005, "completion_per_token": 0.000001})
    assert bound == pytest.approx(
        (0.000001 + 0.000001) * probe.PROBE_MAX_TOKENS
        + 0.0000005 * probe.PROBE_PROMPT_TOKENS_EST)


def test_published_reasoning_rate_overrides_the_assumed_surcharge():
    probe = _probe()
    # An explicit rate is real evidence -- use exactly it, above or below completion.
    cheap = probe.probe_call_bound_usd({
        "prompt_per_token": 0.0000005, "completion_per_token": 0.000001,
        "internal_reasoning_per_token": 0.0000001})
    assert cheap == pytest.approx(
        (0.000001 + 0.0000001) * probe.PROBE_MAX_TOKENS
        + 0.0000005 * probe.PROBE_PROMPT_TOKENS_EST)
    assumed = probe.probe_call_bound_usd(
        {"prompt_per_token": 0.0000005, "completion_per_token": 0.000001})
    assert cheap < assumed


def test_cache_write_ttl_variants_are_accounted_not_refused():
    """Anthropic publishes input_cache_write_1h alongside input_cache_write. A new TTL
    variant must be priced (at the worst rate in the family), not trip the
    fail-closed guard -- otherwise every Anthropic model becomes unpriceable."""
    from backend.endpoint_pricing import fetch_endpoint_pricing
    url = "https://openrouter.ai/api/v1/models/a/b/endpoints"

    def get(_url, timeout=None):
        return httpx.Response(200, json={"data": {"endpoints": [{
            "tag": "anthropic", "provider_name": "Anthropic",
            "pricing": {"prompt": "0.000003", "completion": "0.000015",
                        "input_cache_write": "0.00000375",
                        "input_cache_write_1h": "0.000006",
                        "input_cache_read": "0.0000003"},
        }]}}, request=httpx.Request("GET", url))

    price = fetch_endpoint_pricing("a/b", "anthropic", get=get)
    assert price["unaccounted_nonzero_price_keys"] == []
    # worst of the cache-write family, not just the plain key
    assert price["input_cache_write_per_token"] == pytest.approx(0.000006)


# --- recording the endpoint that ACTUALLY SERVED -----------------------------

def _named_endpoints_getter(by_model):
    """/endpoints where each entry is (tag, provider_name)."""
    def get(url, timeout=None):
        model = url.split("/models/", 1)[1].rsplit("/endpoints", 1)[0]
        return httpx.Response(200, json={"data": {"endpoints": [
            {"tag": tag, "provider_name": name, "model_id": model,
             "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
            for tag, name in by_model.get(model, [])
        ]}}, request=httpx.Request("GET", url))
    return get


def test_served_provider_name_reads_the_selected_endpoint():
    probe = _probe()
    assert probe.served_provider_name({"openrouter_metadata": {"endpoints": {"available": [
        {"provider": "Together", "selected": False},
        {"provider": "Google Vertex", "selected": True},
    ]}}}) == "Google Vertex"
    # no metadata (header not honoured / older route) -> unknown, not a guess
    assert probe.served_provider_name({"choices": []}) is None


def test_served_name_resolves_to_an_exact_tag_when_unambiguous():
    from backend.endpoint_pricing import resolve_served_endpoint_tag
    get = _named_endpoints_getter({"x/y": [("google-vertex", "Google Vertex"),
                                           ("google-ai-studio", "Google AI Studio")]})
    # display name -> the ONE endpoint published under it
    assert resolve_served_endpoint_tag("x/y", "Google Vertex", get=get) == "google-vertex"
    # normalisation: metadata spelling never matches a slug character-for-character
    assert resolve_served_endpoint_tag("x/y", "google vertex", get=get) == "google-vertex"


def test_served_name_yields_no_pin_when_it_maps_to_several_tags():
    from backend.endpoint_pricing import resolve_served_endpoint_tag
    # Router metadata reports a provider NAME, never a tag, and one name can cover
    # variants that price differently. Recording either would force real traffic onto
    # an endpoint nobody chose, so this must decline rather than guess.
    get = _named_endpoints_getter({"x/y": [
        ("google-vertex", "Google Vertex"),
        ("google-vertex/europe", "Google Vertex"),
        ("google-vertex/global", "Google Vertex"),
    ]})
    assert resolve_served_endpoint_tag("x/y", "Google Vertex", get=get) is None


def test_served_name_yields_no_pin_when_unknown_or_absent():
    from backend.endpoint_pricing import resolve_served_endpoint_tag
    get = _named_endpoints_getter({"x/y": [("openai", "OpenAI")]})
    assert resolve_served_endpoint_tag("x/y", "Nobody", get=get) is None
    assert resolve_served_endpoint_tag("x/y", None, get=get) is None


@pytest.mark.asyncio
async def test_probe_records_no_pin_when_the_served_provider_is_ambiguous():
    probe = _probe()
    # Capability is still recorded -- only the pin is withheld. resolve_model_reasoning
    # then returns no endpoint_pin and runtime routes normally, exactly as today.
    tx = _mock_transport({None: 0, "low": 10, "medium": 20, "high": 30},
                         served_provider="Google Vertex")
    rec = await probe.probe_model(
        _entry("g/m"), "k", transport=tx,
        get=_named_endpoints_getter({"g/m": [("google-vertex", "Google Vertex"),
                                             ("google-vertex/europe", "Google Vertex")]}),
    )
    assert rec["probed"] is True
    assert rec["control_surface"] == "levels"
    assert rec["provider_pinned"] is None


@pytest.mark.asyncio
async def test_probe_keeps_the_capability_when_tag_resolution_fails():
    probe = _probe()
    def exploding_get(url, timeout=None):
        raise httpx.ConnectError("endpoints unreachable")
    rec = await probe.probe_model(
        _entry("g/m"), "k",
        transport=_mock_transport({None: 0, "low": 10, "medium": 20, "high": 30},
                                  served_provider="OpenAI"),
        get=exploding_get,
    )
    # A failed pin lookup must not discard an observed capability.
    assert rec["probed"] is True and rec["control_surface"] == "levels"
    assert rec["provider_pinned"] is None


def test_bounds_cover_the_priciest_endpoint_since_the_probe_is_unpinned():
    probe = _probe()
    # Unpinned => any endpoint may serve => only the WORST is an upper bound.
    get = _endpoints_getter({"m/x": [("cheap", 0.0000001, 0.0000002),
                                     ("dear", 0.000002, 0.00001)]})
    bounds = probe.resolve_probe_call_bounds([_entry("m/x")], get=get)
    assert bounds["m/x"] == pytest.approx(
        2 * 0.00001 * probe.PROBE_MAX_TOKENS + 0.000002 * probe.PROBE_PROMPT_TOKENS_EST)
