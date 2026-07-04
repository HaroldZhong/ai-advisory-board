import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


def configure_attachment_storage(monkeypatch, attachment_storage, tmp_path):
    base_dir = tmp_path / "attachments"
    monkeypatch.setattr(attachment_storage, "ATTACHMENTS_DIR", str(base_dir))
    monkeypatch.setattr(attachment_storage, "ATTACHMENTS_META_DIR", str(base_dir / "meta"))
    monkeypatch.setattr(attachment_storage, "ATTACHMENTS_RAW_DIR", str(base_dir / "raw"))
    monkeypatch.setattr(attachment_storage, "ATTACHMENTS_TEXT_DIR", str(base_dir / "text"))
    monkeypatch.setattr(attachment_storage, "CACHE_INDEX_PATH", str(base_dir / "cache_index.json"))


def test_delete_attachment_removes_artifacts_and_cache(monkeypatch, tmp_path):
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)

    attachment = attachment_storage.create_attachment(
        b"private file contents",
        "private.txt",
        "text/plain",
    )
    attachment_storage.update_attachment_status(attachment.attachment_id, "success")
    attachment_storage.save_attachment_text(attachment.attachment_id, "private extracted text")

    raw_path = os.path.join(attachment_storage.ATTACHMENTS_RAW_DIR, f"{attachment.attachment_id}.bin")
    text_path = os.path.join(attachment_storage.ATTACHMENTS_TEXT_DIR, f"{attachment.attachment_id}.txt")
    meta_path = os.path.join(attachment_storage.ATTACHMENTS_META_DIR, f"{attachment.attachment_id}.json")
    assert os.path.exists(raw_path)
    assert os.path.exists(text_path)
    assert os.path.exists(meta_path)
    assert attachment_storage.get_cache_index()[attachment.sha256] == attachment.attachment_id

    result = attachment_storage.delete_attachment(attachment.attachment_id)

    assert result["found"] is True
    assert result["deleted"] is True
    assert result["retained"] is False
    assert not os.path.exists(raw_path)
    assert not os.path.exists(text_path)
    assert not os.path.exists(meta_path)
    assert attachment.sha256 not in attachment_storage.get_cache_index()


def test_delete_attachment_retains_referenced_files_without_force(monkeypatch, tmp_path):
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)

    attachment = attachment_storage.create_attachment(
        b"shared file contents",
        "shared.txt",
        "text/plain",
    )
    attachment_storage.link_attachments_to_conversation(
        [attachment.attachment_id],
        "conv-shared",
    )

    result = attachment_storage.delete_attachment(attachment.attachment_id)

    assert result["found"] is True
    assert result["deleted"] is False
    assert result["retained"] is True
    retained = attachment_storage.get_attachment(attachment.attachment_id)
    assert retained is not None
    assert retained.conversation_ids == ["conv-shared"]


def test_force_delete_attachment_overrides_existing_references(monkeypatch, tmp_path):
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)

    attachment = attachment_storage.create_attachment(
        b"force delete contents",
        "force.txt",
        "text/plain",
    )
    attachment_storage.update_attachment_status(attachment.attachment_id, "success")
    attachment_storage.save_attachment_text(attachment.attachment_id, "force delete text")
    attachment_storage.link_attachments_to_conversation(
        [attachment.attachment_id],
        "conv-force-delete",
    )

    result = attachment_storage.delete_attachment(attachment.attachment_id, force=True)

    assert result["found"] is True
    assert result["deleted"] is True
    assert result["retained"] is False
    assert result["conversation_ids"] == []
    assert attachment_storage.get_attachment(attachment.attachment_id) is None
    assert attachment.sha256 not in attachment_storage.get_cache_index()


@pytest.mark.asyncio
async def test_delete_conversation_purges_unshared_attachment_artifacts(monkeypatch, tmp_path):
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    main = import_module_with_api_key(monkeypatch, "backend.main")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path / "conversations"))
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(delete_conversation_memories=Mock()))

    conversation_id = "conv-delete-attachments"
    storage.create_conversation(conversation_id)
    attachment = attachment_storage.create_attachment(
        b"delete with conversation",
        "delete-me.txt",
        "text/plain",
    )
    attachment_storage.update_attachment_status(attachment.attachment_id, "success")
    attachment_storage.save_attachment_text(attachment.attachment_id, "delete with conversation")
    linked = attachment_storage.link_attachments_to_conversation(
        [attachment.attachment_id],
        conversation_id,
    )
    storage.add_user_message(
        conversation_id,
        "Use this file",
        attachment_ids=[attachment.attachment_id],
        attachments=[linked[0].model_dump()],
    )

    result = await main.delete_conversation(conversation_id)

    assert result["success"] is True
    assert result["attachments"]["deleted"] == 1
    assert storage.get_conversation(conversation_id) is None
    assert attachment_storage.get_attachment(attachment.attachment_id) is None
    main.rag_system.delete_conversation_memories.assert_called_once_with(conversation_id)


@pytest.mark.asyncio
async def test_delete_conversation_retains_shared_attachment_until_last_reference(monkeypatch, tmp_path):
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    main = import_module_with_api_key(monkeypatch, "backend.main")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path / "conversations"))
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(delete_conversation_memories=Mock()))

    attachment = attachment_storage.create_attachment(
        b"shared across conversations",
        "shared.txt",
        "text/plain",
    )

    for conversation_id in ["conv-a", "conv-b"]:
        storage.create_conversation(conversation_id)
        linked = attachment_storage.link_attachments_to_conversation(
            [attachment.attachment_id],
            conversation_id,
        )
        storage.add_user_message(
            conversation_id,
            "Use this shared file",
            attachment_ids=[attachment.attachment_id],
            attachments=[linked[0].model_dump()],
        )

    first_delete = await main.delete_conversation("conv-a")

    assert first_delete["attachments"]["retained"] == 1
    retained = attachment_storage.get_attachment(attachment.attachment_id)
    assert retained is not None
    assert retained.conversation_ids == ["conv-b"]

    second_delete = await main.delete_conversation("conv-b")

    assert second_delete["attachments"]["deleted"] == 1
    assert attachment_storage.get_attachment(attachment.attachment_id) is None


def test_truncated_messages_release_attachment_references(monkeypatch, tmp_path):
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    main = import_module_with_api_key(monkeypatch, "backend.main")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path / "conversations"))

    conversation_id = "conv-truncate-attachment"
    storage.create_conversation(conversation_id)
    attachment = attachment_storage.create_attachment(
        b"remove by edit",
        "remove-by-edit.txt",
        "text/plain",
    )
    linked = attachment_storage.link_attachments_to_conversation(
        [attachment.attachment_id],
        conversation_id,
    )
    storage.add_user_message(
        conversation_id,
        "Use this file",
        attachment_ids=[attachment.attachment_id],
        attachments=[linked[0].model_dump()],
    )
    storage.add_chat_message(conversation_id, "Response")

    result = main.delete_truncated_message_attachments(conversation_id, keep_count=0)
    storage.truncate_messages(conversation_id, 0)

    assert result["deleted"] == 1
    assert attachment_storage.get_attachment(attachment.attachment_id) is None


def test_truncated_attachment_survives_when_resent_via_keep_ids(monkeypatch, tmp_path):
    """Codex review finding (P3-T8 round 2, item 1): an edit/regenerate that
    resends an attachment id only referenced by the truncated tail must not
    have that attachment's files deleted out from under the resend — the
    relink in prepare_message_attachments would then fail on a missing
    record. keep_ids protects ids the caller is about to resend."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    main = import_module_with_api_key(monkeypatch, "backend.main")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path / "conversations"))

    conversation_id = "conv-truncate-keep-ids"
    storage.create_conversation(conversation_id)
    resent = attachment_storage.create_attachment(b"resent file", "resent.txt", "text/plain")
    dropped = attachment_storage.create_attachment(b"dropped file", "dropped.txt", "text/plain")
    linked = attachment_storage.link_attachments_to_conversation(
        [resent.attachment_id, dropped.attachment_id],
        conversation_id,
    )
    storage.add_user_message(
        conversation_id,
        "Use these files",
        attachment_ids=[resent.attachment_id, dropped.attachment_id],
        attachments=[a.model_dump() for a in linked],
    )
    storage.add_chat_message(conversation_id, "Response")

    result = main.delete_truncated_message_attachments(
        conversation_id,
        keep_count=0,
        keep_ids={resent.attachment_id},
    )
    storage.truncate_messages(conversation_id, 0)

    # The resent id is skipped entirely — not even counted as retained/deleted.
    assert resent.attachment_id not in result["attachment_ids"]
    assert attachment_storage.get_attachment(resent.attachment_id) is not None

    # A truncated-only attachment NOT being resent is still cleaned up.
    assert dropped.attachment_id in result["attachment_ids"]
    assert result["deleted"] == 1
    assert attachment_storage.get_attachment(dropped.attachment_id) is None

    # The relink prepare_message_attachments performs on resend must still work.
    relinked = main.prepare_message_attachments(conversation_id, [resent.attachment_id])
    assert len(relinked) == 1
    assert relinked[0]["attachment_id"] == resent.attachment_id


@pytest.mark.asyncio
async def test_edit_regenerate_resent_attachment_survives_end_to_end(monkeypatch, tmp_path):
    """Full run_turn path: editing a message and resending its attachment_id
    must not lose the attachment (Codex review finding, P3-T8 round 2 item 1)."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    main = import_module_with_api_key(monkeypatch, "backend.main")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path / "conversations"))

    conversation_id = "conv-edit-regen-attachment"
    storage.create_conversation(conversation_id)
    attachment = attachment_storage.create_attachment(b"edit regen file", "edit.txt", "text/plain")
    linked = attachment_storage.link_attachments_to_conversation(
        [attachment.attachment_id],
        conversation_id,
    )
    storage.add_user_message(
        conversation_id,
        "Original question",
        attachment_ids=[attachment.attachment_id],
        attachments=[linked[0].model_dump()],
    )
    storage.add_chat_message(conversation_id, "Original answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_chat_with_chairman(*args, **kwargs):
        return {"content": "New answer", "usage": {}}

    async def fake_retrieve_async(*args, **kwargs):
        # retrieve_async returns (context, usage) since PR #75.
        return "", None

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(
            content="Edited question",
            mode="chat",
            edit_index=0,
            attachment_ids=[attachment.attachment_id],
        ),
    )

    assert result["type"] == "chat"
    conversation = storage.get_conversation(conversation_id)
    assert conversation["messages"][0]["attachment_ids"] == [attachment.attachment_id]
    assert conversation["messages"][0]["attachments"][0]["attachment_id"] == attachment.attachment_id
    assert attachment_storage.get_attachment(attachment.attachment_id) is not None
