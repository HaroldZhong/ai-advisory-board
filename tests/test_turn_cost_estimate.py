import importlib

import pytest


def import_modules(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    main = importlib.import_module("backend.main")
    br = importlib.import_module("backend.budget_router")
    config = importlib.import_module("backend.config")
    return main, br, config


def test_council_estimate_exceeds_chat_estimate(monkeypatch):
    """D3 (§5.1): the pre-send estimate accounts for council FAN-OUT, so a council
    turn is materially pricier than a single chat turn -- the honesty gap a
    chairman-only estimate (estimate_message_cost) would hide."""
    _main, br, config = import_modules(monkeypatch)
    council_models = config.COUNCIL_MODELS
    chairman = config.CHAIRMAN_MODEL

    chat = br.estimate_turn_cost("chat", council_models, chairman)
    council = br.estimate_turn_cost("council", council_models, chairman)

    assert chat > 0
    assert council > chat  # N members x (stage1 + stage2) + chairman stage3 dominates


def test_council_estimate_scales_with_member_count(monkeypatch):
    """More council members => higher estimate (the fan-out is real, not flat)."""
    _main, br, config = import_modules(monkeypatch)
    chairman = config.CHAIRMAN_MODEL
    two = br.estimate_turn_cost("council", config.COUNCIL_MODELS[:2], chairman)
    four = br.estimate_turn_cost("council", config.COUNCIL_MODELS[:4], chairman)
    assert four > two


def test_estimate_is_deterministic_and_rounded(monkeypatch):
    _main, br, config = import_modules(monkeypatch)
    a = br.estimate_turn_cost("council", config.COUNCIL_MODELS, config.CHAIRMAN_MODEL)
    b = br.estimate_turn_cost("council", config.COUNCIL_MODELS, config.CHAIRMAN_MODEL)
    assert a == b
    assert round(a, 6) == a


@pytest.mark.asyncio
async def test_estimate_endpoint_shape_and_never_blocks(monkeypatch, tmp_path):
    """The endpoint returns an approximate estimate and NEVER raises for budget
    reasons (soft seatbelt: warn, not block)."""
    main, _br, config = import_modules(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conv = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conv["id"]

    result = await main.estimate_turn_endpoint(conv_id, main.TurnEstimateRequest(mode="council"))
    assert result["approximate"] is True
    assert result["threshold"] == config.LARGE_TURN_ESTIMATE_USD
    assert result["predicted_cost"] > 0
    assert result["is_large"] == (result["predicted_cost"] >= config.LARGE_TURN_ESTIMATE_USD)


@pytest.mark.asyncio
async def test_estimate_endpoint_is_large_tracks_threshold(monkeypatch, tmp_path):
    """D3: is_large == (predicted >= threshold). A tiny threshold flags the turn as
    large; a huge one does not. (LARGE_TURN_ESTIMATE_USD is the tunable knob.)"""
    main, _br, config = import_modules(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conv = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conv["id"]

    monkeypatch.setattr(config, "LARGE_TURN_ESTIMATE_USD", 0.0001)
    large = await main.estimate_turn_endpoint(conv_id, main.TurnEstimateRequest(mode="council"))
    assert large["is_large"] is True

    monkeypatch.setattr(config, "LARGE_TURN_ESTIMATE_USD", 1000.0)
    small = await main.estimate_turn_endpoint(conv_id, main.TurnEstimateRequest(mode="council"))
    assert small["is_large"] is False


def test_chat_estimate_mirrors_run_planner_pricing(monkeypatch):
    """Codex #110 R4: a chat estimate reuses estimate_message_cost (the run planner's
    own pricing) with the resolved execution mode, so a research-mode chat turn is
    priced like the real turn -- not a fixed heuristic that under-estimates it."""
    _main, br, config = import_modules(monkeypatch)
    chairman = config.CHAIRMAN_MODEL
    rag = 16000
    standard = br.estimate_turn_cost("chat", None, chairman, rag_tokens=rag, execution_mode="standard")
    research = br.estimate_turn_cost("chat", None, chairman, rag_tokens=rag, execution_mode="research")
    assert research > standard  # research prices more base tokens
    assert research == br.estimate_message_cost("research", rag, chairman)  # equals the planner primitive


@pytest.mark.asyncio
async def test_estimate_endpoint_flags_large_research_chat_turn(monkeypatch, tmp_path):
    """Codex #110 R4: a research-mode chat turn with the default Opus chairman prices
    above the $0.15 threshold, so the pre-send confirm is NOT skipped for it."""
    main, _br, config = import_modules(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conv = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conv["id"]

    research_chat = await main.estimate_turn_endpoint(
        conv_id, main.TurnEstimateRequest(mode="chat", execution_mode="research")
    )
    assert research_chat["is_large"] is True


@pytest.mark.asyncio
async def test_estimate_endpoint_accounts_for_routing_overrides(monkeypatch, tmp_path):
    """D3 (Codex #110): the estimate reflects the per-send routing overrides that move
    cost -- a bigger RAG context (rag_preset='max' or execution_mode='research')
    predicts a higher cost than the auto baseline, so a large research/high-context
    turn is not silently under-estimated below the threshold."""
    main, _br, _config = import_modules(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conv = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conv["id"]

    baseline = await main.estimate_turn_endpoint(conv_id, main.TurnEstimateRequest(mode="council"))
    max_rag = await main.estimate_turn_endpoint(
        conv_id, main.TurnEstimateRequest(mode="council", rag_preset="max")
    )
    research = await main.estimate_turn_endpoint(
        conv_id, main.TurnEstimateRequest(mode="council", execution_mode="research")
    )

    assert max_rag["predicted_cost"] > baseline["predicted_cost"]
    assert research["predicted_cost"] > baseline["predicted_cost"]


@pytest.mark.asyncio
async def test_estimate_endpoint_promotes_auto_chat_by_task_signal(monkeypatch, tmp_path):
    """Codex #110 R5: on auto execution mode the estimate runs the SAME task-signal
    routing as the send path (create_run_plan on the pending content), so a chat turn
    whose content trips the research signal is priced as research (large) while a plain
    chat turn is not -- an expensive auto chat send no longer bypasses the confirm."""
    main, _br, _config = import_modules(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conv = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conv["id"]

    plain = await main.estimate_turn_endpoint(
        conv_id, main.TurnEstimateRequest(mode="chat", content="hi there")
    )
    research_kw = await main.estimate_turn_endpoint(
        conv_id,
        main.TurnEstimateRequest(mode="chat", content="research and analyze and compare these papers"),
    )
    assert plain["is_large"] is False
    assert research_kw["is_large"] is True


def test_council_estimate_includes_steward_chairman_call(monkeypatch):
    """Codex #110 R9: the council estimate includes the Stage 0b tool-steward call on
    the chairman (run before stage 1), added on top of the member + chairman stages."""
    _main, br, config = import_modules(monkeypatch)
    from backend.budget_router import _model_call_cost, _TURN_TOKEN_ESTIMATES

    council = br.estimate_turn_cost("council", config.COUNCIL_MODELS, config.CHAIRMAN_MODEL)
    s1 = _TURN_TOKEN_ESTIMATES["council_stage1"]
    s2 = _TURN_TOKEN_ESTIMATES["council_stage2"]
    s3 = _TURN_TOKEN_ESTIMATES["chairman_stage3"]
    st = _TURN_TOKEN_ESTIMATES["steward"]
    without_steward = sum(
        _model_call_cost(m, s1["input"], s1["output"]) + _model_call_cost(m, s2["input"], s2["output"])
        for m in config.COUNCIL_MODELS
    ) + _model_call_cost(config.CHAIRMAN_MODEL, s3["input"], s3["output"])
    steward_cost = _model_call_cost(config.CHAIRMAN_MODEL, st["input"], st["output"])
    assert steward_cost > 0
    assert council == round(without_steward + steward_cost, 6)


@pytest.mark.asyncio
async def test_estimate_endpoint_includes_web_search_when_enabled(monkeypatch, tmp_path):
    """Codex #110 R9: web search adds a Stage 0 Perplexity grounding call, so an
    enabled (especially deep) search predicts more than the same turn without it."""
    main, _br, _config = import_modules(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conv = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conv["id"]

    none = await main.estimate_turn_endpoint(conv_id, main.TurnEstimateRequest(mode="chat"))
    fast = await main.estimate_turn_endpoint(
        conv_id, main.TurnEstimateRequest(mode="chat", web_search_enabled=True, web_search_depth="fast")
    )
    deep = await main.estimate_turn_endpoint(
        conv_id, main.TurnEstimateRequest(mode="chat", web_search_enabled=True, web_search_depth="deep")
    )
    assert fast["predicted_cost"] > none["predicted_cost"]
    assert deep["predicted_cost"] > fast["predicted_cost"]


@pytest.mark.asyncio
async def test_estimate_endpoint_404_for_missing_conversation(monkeypatch, tmp_path):
    main, _br, _config = import_modules(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    with pytest.raises(main.HTTPException) as exc:
        await main.estimate_turn_endpoint("does-not-exist", main.TurnEstimateRequest(mode="council"))
    assert exc.value.status_code == 404
