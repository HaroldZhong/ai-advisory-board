import importlib

import pytest


def import_main(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module("backend.main")


def test_exports_dir_lives_under_app_data_root(monkeypatch, tmp_path):
    from backend import app_paths

    monkeypatch.setenv("AAB_DATA_DIR", str(tmp_path))

    assert app_paths.get_exports_dir() == tmp_path / "exports"


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

    export_path = exports_dir / "smoke_export__stage_2_.md"
    assert result == {
        "filename": "smoke_export__stage_2_.md",
        "path": str(export_path),
    }
    assert export_path.exists()
    content = export_path.read_text(encoding="utf-8")
    assert "### Stage 2: Peer Rankings" in content
    assert "**Evaluator: model-b**" in content
    assert "did not return a Stage 2 evaluation" in content


@pytest.mark.asyncio
async def test_export_conversation_returns_404_for_missing_conversation(monkeypatch, tmp_path):
    main = import_main(monkeypatch)

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path / "conversations"))

    with pytest.raises(main.HTTPException) as exc:
        await main.export_conversation("missing-conversation")

    assert exc.value.status_code == 404
