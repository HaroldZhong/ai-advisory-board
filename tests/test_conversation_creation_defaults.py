import importlib

import pytest
from fastapi import HTTPException


def import_main(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module("backend.main")


@pytest.mark.asyncio
async def test_create_conversation_persists_preset_zdr_and_budget(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)
    private_preset = next(
        preset for preset in main.config.MODEL_PRESETS if preset["id"] == "private"
    )

    request = main.CreateConversationRequest(
        topic="Private research",
        preset_id="private",
        zdr_enabled=True,
        budget_usd=2,
        budget_allow_overage=False,
    )

    conversation = await main.create_conversation(request)

    assert conversation["metadata"]["preset_id"] == "private"
    assert conversation["metadata"]["zdr_enabled"] is True
    assert conversation["metadata"]["chairman_model"] == private_preset["chairman_model"]
    assert conversation["metadata"]["council_models"] == private_preset["council_models"]
    policy = main.storage.get_session_policy(conversation["id"])
    assert policy["budget_usd"] == 2
    assert policy["allow_overage"] is False


@pytest.mark.asyncio
async def test_private_preset_rejects_disabled_zdr(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(main.CreateConversationRequest(
            preset_id="private",
            zdr_enabled=False,
        ))

    assert exc.value.status_code == 400
    assert "requires ZDR" in exc.value.detail


@pytest.mark.asyncio
async def test_private_preset_rejects_non_zdr_model_overrides(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    for request in [
        main.CreateConversationRequest(
            preset_id="private",
            chairman_model="openai/gpt-5.5",
        ),
        main.CreateConversationRequest(
            preset_id="private",
            council_members=["openai/gpt-5.4", "x-ai/grok-4.1-fast", "deepseek/deepseek-v4-pro"],
        ),
    ]:
        with pytest.raises(HTTPException) as exc:
            await main.create_conversation(request)

        assert exc.value.status_code == 400
        assert "ZDR-capable models" in exc.value.detail


@pytest.mark.asyncio
async def test_create_conversation_rejects_utility_model_as_chairman(monkeypatch, tmp_path):
    """Codex P2: google/gemini-2.5-flash (UTILITY_MODEL) exists in the
    registry only for RAG-extraction cost accounting (audit §12, P5-T3) and
    must not be selectable as chairman just because it's a valid registry id."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(main.CreateConversationRequest(
            chairman_model="google/gemini-2.5-flash",
        ))

    assert exc.value.status_code == 400
    assert "google/gemini-2.5-flash" in exc.value.detail
    assert "utility" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_create_conversation_rejects_utility_model_as_council_member(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(main.CreateConversationRequest(
            council_members=["openai/gpt-4o-mini", "google/gemini-2.5-flash", "x-ai/grok-4.1-fast"],
        ))

    assert exc.value.status_code == 400
    assert "google/gemini-2.5-flash" in exc.value.detail
    assert "utility" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_create_conversation_rejects_search_model_as_chairman(monkeypatch, tmp_path):
    """v1.2.0: search-type models (e.g. perplexity/sonar) exist only for the
    dedicated Stage 0 web-search step and must not be selectable as chairman."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(main.CreateConversationRequest(
            chairman_model="perplexity/sonar",
        ))

    assert exc.value.status_code == 400
    assert "perplexity/sonar" in exc.value.detail
    assert "search" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_create_conversation_rejects_search_model_as_council_member(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(main.CreateConversationRequest(
            council_members=["openai/gpt-4o-mini", "perplexity/sonar", "x-ai/grok-4.1-fast"],
        ))

    assert exc.value.status_code == 400
    assert "perplexity/sonar" in exc.value.detail
    assert "search" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_create_conversation_accepts_normal_chairman_and_council_models(monkeypatch, tmp_path):
    """Control: the utility-type gate must not affect ordinary registry
    models still selectable as chairman/council."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    conversation = await main.create_conversation(main.CreateConversationRequest(
        chairman_model="openai/gpt-5.5",
        council_members=["openai/gpt-4o-mini", "x-ai/grok-4.1-fast", "deepseek/deepseek-v4-pro"],
    ))

    assert conversation["metadata"]["chairman_model"] == "openai/gpt-5.5"
    assert conversation["metadata"]["council_models"] == [
        "openai/gpt-4o-mini", "x-ai/grok-4.1-fast", "deepseek/deepseek-v4-pro",
    ]


@pytest.mark.asyncio
async def test_create_conversation_does_not_store_implicit_zdr_false(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    conversation = await main.create_conversation(main.CreateConversationRequest())

    assert "zdr_enabled" not in conversation["metadata"]
    assert main.storage.get_session_policy(conversation["id"])["allow_overage"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("chairman_override", ["", "   "])
async def test_create_conversation_applies_preset_when_overrides_are_blank(monkeypatch, tmp_path, chairman_override):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)
    balanced = next(preset for preset in main.config.MODEL_PRESETS if preset["id"] == "balanced")

    conversation = await main.create_conversation(main.CreateConversationRequest(
        preset_id="balanced",
        council_members=[],
        chairman_model=chairman_override,
    ))

    assert conversation["metadata"]["preset_id"] == "balanced"
    assert conversation["metadata"]["council_models"] == balanced["council_models"]
    assert conversation["metadata"]["chairman_model"] == balanced["chairman_model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("default_mode", ["chat", "council"])
async def test_create_conversation_persists_default_mode(monkeypatch, tmp_path, default_mode):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    conversation = await main.create_conversation(main.CreateConversationRequest(
        default_mode=default_mode,
    ))

    assert conversation["metadata"]["default_mode"] == default_mode


@pytest.mark.asyncio
async def test_create_conversation_rejects_invalid_default_mode(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(main.CreateConversationRequest(default_mode="bogus"))

    assert exc.value.status_code == 400
    assert "default_mode" in exc.value.detail


@pytest.mark.asyncio
async def test_create_conversation_omits_default_mode_when_absent(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    conversation = await main.create_conversation(main.CreateConversationRequest())

    assert "default_mode" not in conversation["metadata"]
