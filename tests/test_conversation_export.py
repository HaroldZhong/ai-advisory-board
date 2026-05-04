import importlib
import logging

import pytest

from backend.conversation_export import get_conversation_export_filename


def import_main(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module("backend.main")


def test_exports_dir_lives_under_app_data_root(monkeypatch, tmp_path):
    from backend import app_paths

    monkeypatch.setenv("AAB_DATA_DIR", str(tmp_path))

    assert app_paths.get_exports_dir() == tmp_path / "exports"


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("What is React?", "what_is_react.md"),
        ("\U0001f389\U0001f389\U0001f389", "conversation_abcdef12.md"),
        ("   ", "conversation_abcdef12.md"),
        ("\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440", "\u043f\u0440\u0438\u0432\u0435\u0442_\u043c\u0438\u0440.md"),
    ],
)
def test_export_filename_sanitizes_titles_without_all_underscore_outputs(title, expected):
    assert get_conversation_export_filename(title, "abcdef1234567890") == expected


@pytest.mark.parametrize("title", ["CON", "prn", "AUX", "nul", "COM1", "LPT9"])
def test_export_filename_escapes_windows_reserved_names(title):
    assert get_conversation_export_filename(title, "abcdef1234567890") == f"{title.lower()}_export.md"


def test_export_filename_truncates_long_titles_to_filesystem_safe_length():
    filename = get_conversation_export_filename("a" * 400, "abcdef1234567890")

    assert filename.endswith(".md")
    assert len(filename) <= 204


@pytest.mark.asyncio
async def test_export_conversation_writes_markdown_to_exact_app_data_path(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-export"
    conversations_dir = tmp_path / "data" / "conversations"
    exports_dir = tmp_path / "exports"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(main.app_paths, "get_exports_dir", lambda: exports_dir)

    conversation = main.storage.create_conversation(
        conversation_id,
        {
            "council_models": ["provider/model-a", "provider/model-b"],
            "chairman_model": "provider/chair",
        },
    )
    conversation["title"] = "Smoke Export: Stage 2?"
    conversation["created_at"] = "2026-05-04T20:00:00.000000"
    conversation["messages"] = [
        {"role": "user", "content": "Question?"},
        {
            "role": "assistant",
            "stage1": [{"model": "provider/model-a", "response": "Answer A"}],
            "stage2": [
                {
                    "model": "provider/model-b",
                    "ranking": "provider/model-b did not return a Stage 2 evaluation.",
                    "parsed_ranking": [],
                    "status": "unavailable",
                }
            ],
            "stage3": {"model": "provider/chair", "response": "Final answer", "confidence": "medium"},
        },
    ]
    main.storage.save_conversation(conversation)

    result = await main.export_conversation(conversation_id)

    export_path = exports_dir / "smoke_export_stage_2.md"
    assert result == {
        "filename": "smoke_export_stage_2.md",
        "path": str(export_path),
    }
    assert export_path.exists()
    content = export_path.read_text(encoding="utf-8")
    assert "Date: 2026-05-04 20:00:00 UTC" in content
    assert "### Stage 2: Peer Rankings" in content
    assert "**Evaluator: model-b**" in content
    assert "did not return a Stage 2 evaluation" in content


@pytest.mark.asyncio
async def test_export_conversation_uses_unique_filename_on_collision(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    exports_dir = tmp_path / "exports"
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path / "data" / "conversations"))
    monkeypatch.setattr(main.app_paths, "get_exports_dir", lambda: exports_dir)

    first = main.storage.create_conversation("conv-one")
    first["title"] = "What is React?"
    first["messages"] = [{"role": "user", "content": "First"}]
    main.storage.save_conversation(first)

    second = main.storage.create_conversation("conv-two")
    second["title"] = "What is React!"
    second["messages"] = [{"role": "user", "content": "Second"}]
    main.storage.save_conversation(second)

    first_result = await main.export_conversation("conv-one")
    second_result = await main.export_conversation("conv-two")

    assert first_result["filename"] == "what_is_react.md"
    assert second_result["filename"] == "what_is_react_2.md"
    assert (exports_dir / "what_is_react.md").read_text(encoding="utf-8") != (
        exports_dir / "what_is_react_2.md"
    ).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_export_conversation_catches_unicode_write_errors(monkeypatch, tmp_path, caplog):
    main = import_main(monkeypatch)
    caplog.set_level(logging.ERROR, logger="LLMCouncil")

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path / "data" / "conversations"))
    monkeypatch.setattr(main.app_paths, "get_exports_dir", lambda: tmp_path / "exports")

    conversation = main.storage.create_conversation("conv-surrogate")
    conversation["title"] = "Unicode failure"
    conversation["messages"] = [{"role": "user", "content": "\ud800"}]
    main.storage.save_conversation(conversation)

    with pytest.raises(main.HTTPException) as exc:
        await main.export_conversation("conv-surrogate")

    assert exc.value.status_code == 500
    assert "Failed to export conversation conv-surrogate" in caplog.text


@pytest.mark.asyncio
async def test_export_conversation_includes_chat_messages_and_excludes_system_messages(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    exports_dir = tmp_path / "exports"
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path / "data" / "conversations"))
    monkeypatch.setattr(main.app_paths, "get_exports_dir", lambda: exports_dir)

    conversation = main.storage.create_conversation("conv-chat")
    conversation["title"] = "Chat export"
    conversation["messages"] = [
        {"role": "system", "content": "Hidden instruction"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    main.storage.save_conversation(conversation)

    result = await main.export_conversation("conv-chat")
    content = (exports_dir / result["filename"]).read_text(encoding="utf-8")

    assert "## User" in content
    assert "## Assistant" in content
    assert "Hi there" in content
    assert "Hidden instruction" not in content
    assert "System messages excluded" in content


@pytest.mark.asyncio
async def test_export_conversation_returns_404_for_missing_conversation(monkeypatch, tmp_path):
    main = import_main(monkeypatch)

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path / "conversations"))

    with pytest.raises(main.HTTPException) as exc:
        await main.export_conversation("missing-conversation")

    assert exc.value.status_code == 404
