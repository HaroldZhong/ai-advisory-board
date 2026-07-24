import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException


def import_main(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module("backend.main")


@pytest.mark.asyncio
async def test_folder_endpoint_roundtrip_persists_updates(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))

    created = await main.create_folder(main.FolderCreate(name="Research", color="#4466ff"))
    listed = await main.get_folders()
    updated = await main.update_folder(
        created["id"],
        main.FolderUpdate(name="Research Notes"),
    )

    assert listed == [created]
    assert updated["id"] == created["id"]
    assert updated["name"] == "Research Notes"
    assert updated["color"] == "#4466ff"

    delete_result = await main.delete_folder(created["id"])

    assert delete_result == {"success": True}
    assert await main.get_folders() == []


@pytest.mark.asyncio
async def test_conversation_update_renames_and_moves_folder(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-organized"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    fake_rag = SimpleNamespace(update_conversation_folder=Mock())
    monkeypatch.setattr(main, "rag_system", fake_rag)

    main.storage.create_conversation(conversation_id)

    updated = await main.update_conversation(
        conversation_id,
        main.ConversationUpdate(title="Renamed chat", folder_id="folder-1"),
    )

    assert updated["title"] == "Renamed chat"
    assert updated["metadata"]["folder_id"] == "folder-1"
    fake_rag.update_conversation_folder.assert_called_once_with(conversation_id, "folder-1")

    listed = await main.list_conversations()
    assert listed[0]["id"] == conversation_id
    assert listed[0]["title"] == "Renamed chat"
    assert listed[0]["folder_id"] == "folder-1"


@pytest.mark.asyncio
async def test_conversation_update_removes_folder_route(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-root"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    fake_rag = SimpleNamespace(update_conversation_folder=Mock())
    monkeypatch.setattr(main, "rag_system", fake_rag)

    main.storage.create_conversation(conversation_id, {"folder_id": "folder-1"})

    updated = await main.update_conversation(
        conversation_id,
        main.ConversationUpdate(folder_id=None),
    )

    assert "folder_id" not in updated["metadata"]
    fake_rag.update_conversation_folder.assert_called_once_with(conversation_id, "root")


@pytest.mark.asyncio
async def test_conversation_update_can_enable_zdr_for_compatible_models(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-zdr-compatible"
    private_preset = next(preset for preset in main.config.MODEL_PRESETS if preset["id"] == "private")

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {
            "chairman_model": private_preset["chairman_model"],
            "council_models": private_preset["council_models"],
        },
    )

    updated = await main.update_conversation(
        conversation_id,
        main.ConversationUpdate(zdr_enabled=True),
    )

    assert updated["metadata"]["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_conversation_update_rejects_zdr_for_incompatible_models(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-zdr-incompatible"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {
            "chairman_model": "openai/gpt-5.5",
            "council_models": ["mistralai/mistral-large-2512", "deepseek/deepseek-v4-pro", "qwen/qwen3-max-thinking"],
        },
    )

    with pytest.raises(HTTPException) as exc:
        await main.update_conversation(
            conversation_id,
            main.ConversationUpdate(zdr_enabled=True),
        )

    assert exc.value.status_code == 400
    assert "ZDR-capable models" in exc.value.detail


@pytest.mark.asyncio
async def test_conversation_update_cannot_disable_required_zdr_preset(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-private"
    private_preset = next(preset for preset in main.config.MODEL_PRESETS if preset["id"] == "private")

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {
            "preset_id": "private",
            "zdr_enabled": True,
            "chairman_model": private_preset["chairman_model"],
            "council_models": private_preset["council_models"],
        },
    )

    with pytest.raises(HTTPException) as exc:
        await main.update_conversation(
            conversation_id,
            main.ConversationUpdate(zdr_enabled=False),
        )

    assert exc.value.status_code == 400
    assert "requires ZDR" in exc.value.detail
