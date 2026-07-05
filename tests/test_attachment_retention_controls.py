import importlib
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(delete_conversation_memories=AsyncMock()))

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
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(delete_conversation_memories=AsyncMock()))

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


def test_deleted_attachment_ids_excludes_an_attachment_retained_by_another_conversation(monkeypatch, tmp_path):
    """Codex round 23 P2: delete_truncated_message_attachments' attachment_ids
    is only the CANDIDATE list built from the removed messages -- an
    attachment referenced by ANOTHER conversation too is RETAINED by
    delete_attachment (its files survive, refcounted across conversation_ids),
    even though it's still in the candidate list. deleted_attachment_ids
    (Codex round 23) is the narrower list of ids delete_attachment actually
    deleted, computed from each result's own "deleted" flag -- this is the
    list callers must use to purge document MEMORY, or a retained
    attachment's memory gets purged even though its files (and the other
    conversation's reference) are still there."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    main = import_module_with_api_key(monkeypatch, "backend.main")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path / "conversations"))

    conversation_id = "conv-truncate-shared"
    other_conversation_id = "conv-other-shares-it"
    storage.create_conversation(conversation_id)
    storage.create_conversation(other_conversation_id)

    shared = attachment_storage.create_attachment(b"shared file", "shared.txt", "text/plain")
    dropped = attachment_storage.create_attachment(b"dropped file", "dropped.txt", "text/plain")
    linked = attachment_storage.link_attachments_to_conversation(
        [shared.attachment_id, dropped.attachment_id],
        conversation_id,
    )
    # The OTHER conversation also references the shared attachment.
    attachment_storage.link_attachments_to_conversation([shared.attachment_id], other_conversation_id)

    storage.add_user_message(
        conversation_id,
        "Use these files",
        attachment_ids=[shared.attachment_id, dropped.attachment_id],
        attachments=[a.model_dump() for a in linked],
    )
    storage.add_chat_message(conversation_id, "Response")

    result = main.delete_truncated_message_attachments(conversation_id, keep_count=0)
    storage.truncate_messages(conversation_id, 0)

    # Both ids are CANDIDATES (both were in the truncated tail).
    assert set(result["attachment_ids"]) == {shared.attachment_id, dropped.attachment_id}
    # But only "dropped" was ACTUALLY deleted -- "shared" is retained
    # (the other conversation's reference keeps its files alive).
    assert result["deleted_attachment_ids"] == [dropped.attachment_id]
    assert attachment_storage.get_attachment(shared.attachment_id) is not None
    assert attachment_storage.get_attachment(dropped.attachment_id) is None


@pytest.mark.asyncio
async def test_edit_truncation_does_not_purge_document_memory_for_a_retained_attachment(monkeypatch, tmp_path):
    """Codex round 23 P2 end-to-end: an attachment shared with another
    conversation survives an edit-truncation's cleanup (files retained via
    refcounting) -- its document memory must survive too. Only a
    truncated-only attachment's memory gets purged. Fails pre-fix: the old
    code passed attachment_ids (the broader candidate list, including the
    retained "shared" id) to purge_document_memories, purging a memory
    whose attachment (and the other conversation's own reference to it)
    was never actually deleted."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    main = import_module_with_api_key(monkeypatch, "backend.main")
    from unittest.mock import AsyncMock
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path / "conversations"))

    conversation_id = "conv-truncate-shared-e2e"
    other_conversation_id = "conv-other-shares-it-e2e"
    storage.create_conversation(conversation_id)
    storage.create_conversation(other_conversation_id)

    shared = attachment_storage.create_attachment(b"shared file", "shared.txt", "text/plain")
    dropped = attachment_storage.create_attachment(b"dropped file", "dropped.txt", "text/plain")
    linked = attachment_storage.link_attachments_to_conversation(
        [shared.attachment_id, dropped.attachment_id],
        conversation_id,
    )
    attachment_storage.link_attachments_to_conversation([shared.attachment_id], other_conversation_id)

    storage.add_user_message(
        conversation_id,
        "Use these files",
        attachment_ids=[shared.attachment_id, dropped.attachment_id],
        attachments=[a.model_dump() for a in linked],
    )
    storage.add_chat_message(conversation_id, "Response")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        return {"content": "Regenerated", "usage": {}}

    async def fake_topics(*args, **kwargs):
        return (["topic"], {})

    purge_document_memories = AsyncMock(return_value=1)
    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_topics)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            retrieve_async=fake_retrieve_async,
            index_chat_turn=AsyncMock(return_value=None),
            refresh_hybrid_index=lambda *a, **k: None,
            purge_truncated_memories=AsyncMock(return_value=0),
            purge_document_memories=purge_document_memories,
            store={},
        ),
    )

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Edited question", mode="chat", edit_index=0),
    )

    # Only the truncated-only, ACTUALLY-deleted attachment's memory is purged.
    purge_document_memories.assert_awaited_once_with([dropped.attachment_id], conversation_id=conversation_id)


def test_attachment_in_kept_prefix_and_truncated_tail_survives(monkeypatch, tmp_path):
    """Codex round 24 P2: delete_attachment's refcounting is
    CONVERSATION-level (one entry per conversation_id), not message-level
    -- an attachment referenced by BOTH a kept-prefix message and the
    truncated tail has only ONE ref for this conversation, so calling
    delete_attachment(id, conversation_id=this_conv) used to strip that ref
    and report deleted=True, deleting files a still-kept message needs
    (and, since round 23, purging its memory too). This predates the
    round-23 memory-purge fix -- round 23 merely made the existing
    file-deletion bug visible by also purging the now-orphaned memory.

    Fixed at the sharper seam Codex suggested: exclude kept-prefix
    attachment ids from the candidate list itself, in
    delete_truncated_message_attachments, the same way keep_ids excludes
    ids about to be resent. delete_attachment must never even be CALLED
    for an id still referenced by a surviving message -- asserted via a
    spy, not just via the end result (which could also pass if
    delete_attachment happened to no-op for some other reason)."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    main = import_module_with_api_key(monkeypatch, "backend.main")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path / "conversations"))

    conversation_id = "conv-prefix-and-tail"
    storage.create_conversation(conversation_id)

    shared = attachment_storage.create_attachment(b"shared across turns", "shared.txt", "text/plain")
    linked = attachment_storage.link_attachments_to_conversation([shared.attachment_id], conversation_id)

    # Message 0 (KEPT prefix, keep_count=1): references the attachment.
    storage.add_user_message(
        conversation_id,
        "First use of the file",
        attachment_ids=[shared.attachment_id],
        attachments=[a.model_dump() for a in linked],
    )
    storage.add_chat_message(conversation_id, "First response")
    # Message 2 (TRUNCATED tail): references the SAME attachment again.
    storage.add_user_message(
        conversation_id,
        "Second use of the same file",
        attachment_ids=[shared.attachment_id],
        attachments=[a.model_dump() for a in linked],
    )
    storage.add_chat_message(conversation_id, "Second response")

    calls = []
    real_delete_attachment = main.delete_attachment

    def spy_delete_attachment(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("attachment_id"))
        return real_delete_attachment(*args, **kwargs)

    monkeypatch.setattr(main, "delete_attachment", spy_delete_attachment)

    # keep_count=2: messages 0-1 survive, messages 2-3 are truncated away.
    result = main.delete_truncated_message_attachments(conversation_id, keep_count=2)
    storage.truncate_messages(conversation_id, 2)

    # delete_attachment must never even be called for the shared id.
    assert calls == []
    assert shared.attachment_id not in result["attachment_ids"]
    assert shared.attachment_id not in result["deleted_attachment_ids"]
    assert attachment_storage.get_attachment(shared.attachment_id) is not None

    # The relink prepare_message_attachments performs for the kept message
    # must still work (files genuinely untouched, not just metadata-retained).
    relinked = main.prepare_message_attachments(conversation_id, [shared.attachment_id])
    assert len(relinked) == 1
    assert relinked[0]["attachment_id"] == shared.attachment_id


def test_attachment_only_in_truncated_tail_is_still_deleted(monkeypatch, tmp_path):
    """Codex round 24 P2: the kept-prefix filter must not become
    overbroad -- an attachment referenced ONLY by messages in the
    truncated tail (not by anything in the kept prefix) is still deleted
    exactly as before."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    main = import_module_with_api_key(monkeypatch, "backend.main")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path / "conversations"))

    conversation_id = "conv-tail-only"
    storage.create_conversation(conversation_id)

    kept_only = attachment_storage.create_attachment(b"kept file", "kept.txt", "text/plain")
    tail_only = attachment_storage.create_attachment(b"tail file", "tail.txt", "text/plain")
    linked = attachment_storage.link_attachments_to_conversation(
        [kept_only.attachment_id, tail_only.attachment_id],
        conversation_id,
    )

    # Message 0 (KEPT): references kept_only.
    storage.add_user_message(
        conversation_id,
        "Use the kept file",
        attachment_ids=[kept_only.attachment_id],
        attachments=[linked[0].model_dump()],
    )
    storage.add_chat_message(conversation_id, "Response 1")
    # Message 2 (TRUNCATED): references tail_only, NOT kept_only.
    storage.add_user_message(
        conversation_id,
        "Use the tail file",
        attachment_ids=[tail_only.attachment_id],
        attachments=[linked[1].model_dump()],
    )
    storage.add_chat_message(conversation_id, "Response 2")

    result = main.delete_truncated_message_attachments(conversation_id, keep_count=2)
    storage.truncate_messages(conversation_id, 2)

    assert result["attachment_ids"] == [tail_only.attachment_id]
    assert result["deleted_attachment_ids"] == [tail_only.attachment_id]
    assert attachment_storage.get_attachment(tail_only.attachment_id) is None
    # The kept-prefix-only attachment is untouched (not even a candidate).
    assert attachment_storage.get_attachment(kept_only.attachment_id) is not None


def test_keep_ids_resend_behavior_unchanged_alongside_kept_prefix_filter(monkeypatch, tmp_path):
    """Codex round 24 P2: (c) keep_ids (ids being resent with the edited
    message, round-2 fix) and the new kept-prefix filter are independent
    exclusions that must compose correctly -- a resent id that's ONLY in
    the truncated tail (not the kept prefix) is still protected by
    keep_ids exactly as before."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    attachment_storage = import_module_with_api_key(monkeypatch, "backend.attachment_storage")
    main = import_module_with_api_key(monkeypatch, "backend.main")
    configure_attachment_storage(monkeypatch, attachment_storage, tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path / "conversations"))

    conversation_id = "conv-keep-ids-plus-prefix"
    storage.create_conversation(conversation_id)

    resent = attachment_storage.create_attachment(b"resent file", "resent.txt", "text/plain")
    dropped = attachment_storage.create_attachment(b"dropped file", "dropped.txt", "text/plain")
    linked = attachment_storage.link_attachments_to_conversation(
        [resent.attachment_id, dropped.attachment_id],
        conversation_id,
    )

    storage.add_user_message(conversation_id, "Use these files")  # kept, no attachments
    storage.add_chat_message(conversation_id, "Response 1")
    storage.add_user_message(
        conversation_id,
        "Use these files again",
        attachment_ids=[resent.attachment_id, dropped.attachment_id],
        attachments=[a.model_dump() for a in linked],
    )
    storage.add_chat_message(conversation_id, "Response 2")

    result = main.delete_truncated_message_attachments(
        conversation_id,
        keep_count=2,
        keep_ids={resent.attachment_id},
    )
    storage.truncate_messages(conversation_id, 2)

    # resent is skipped via keep_ids (not even a candidate); dropped is a
    # normal tail-only deletion.
    assert resent.attachment_id not in result["attachment_ids"]
    assert attachment_storage.get_attachment(resent.attachment_id) is not None
    assert dropped.attachment_id in result["attachment_ids"]
    assert result["deleted_attachment_ids"] == [dropped.attachment_id]
    assert attachment_storage.get_attachment(dropped.attachment_id) is None

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

    async def fake_topics(*args, **kwargs):
        return (["topic"], {})

    async def fake_index_chat_turn(*args, **kwargs):
        return None

    async def fake_purge_truncated_memories(*args, **kwargs):
        return 0

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_topics)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            retrieve_async=fake_retrieve_async,
            index_chat_turn=fake_index_chat_turn,
            refresh_hybrid_index=lambda *a, **k: None,
            purge_truncated_memories=fake_purge_truncated_memories,
            store={},
        ),
    )

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
