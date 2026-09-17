"""Unit tests for rewrite_query heuristics (backend/council.py)."""
import pytest
from backend import council


@pytest.mark.asyncio
async def test_long_query_skips_rewrite(monkeypatch):
    async def explode(*args, **kwargs):
        raise AssertionError("must not call the LLM for self-contained queries")
    monkeypatch.setattr(council, "query_model", explode)
    query = "What are the main differences between RAG and fine-tuning for domain adaptation tasks?"
    assert await council.rewrite_query(query, [{"role": "user", "content": "hi"}] * 4) == query


@pytest.mark.asyncio
async def test_no_history_skips_rewrite(monkeypatch):
    async def explode(*args, **kwargs):
        raise AssertionError("must not call the LLM without context")
    monkeypatch.setattr(council, "query_model", explode)
    assert await council.rewrite_query("why?", []) == "why?"


@pytest.mark.asyncio
async def test_short_followup_uses_llm_rewrite(monkeypatch):
    async def fake_llm(model, messages, **kwargs):
        return {"content": "How does RAG handle document updates?", "usage": {}}
    monkeypatch.setattr(council, "query_model", fake_llm)
    history = [
        {"role": "user", "content": "How does RAG work?"},
        {"role": "assistant", "stage3": {"response": "RAG combines retrieval with generation..."}},
    ]
    assert await council.rewrite_query("what about updates?", history) == "How does RAG handle document updates?"


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_original(monkeypatch):
    async def fail(*args, **kwargs):
        return None
    monkeypatch.setattr(council, "query_model", fail)
    history = [{"role": "user", "content": "How does RAG work?"},
               {"role": "assistant", "content": "..."}]
    assert await council.rewrite_query("what about it?", history) == "what about it?"


@pytest.mark.asyncio
@pytest.mark.parametrize('query', ['Explain [S1.1]', '比较[S1.1]和[S2.3]。'])
async def test_explicit_current_citations_skip_rewrite(monkeypatch, query):
    from unittest.mock import AsyncMock
    rewrite = AsyncMock(return_value={'content': query})
    monkeypatch.setattr(council, 'query_model', rewrite)
    history = [{'role': 'user', 'content': 'Old topic'},
               {'role': 'assistant', 'content': 'Different old excerpt [S1.1]'}]
    assert await council.rewrite_query(query, history) == query
    rewrite.assert_not_awaited()
