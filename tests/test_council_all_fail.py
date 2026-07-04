"""run_full_council must return a 5-tuple even when every model fails (audit §4.2)."""
import pytest
from backend import council
from backend.tools.types import EvidencePack


@pytest.mark.asyncio
async def test_run_full_council_all_models_fail_returns_five_tuple(monkeypatch):
    async def all_fail_parallel(models, messages, **kwargs):
        return {m: None for m in models}

    async def fail_single(model, messages, **kwargs):
        return None

    monkeypatch.setattr(council, "query_models_parallel", all_fail_parallel)
    monkeypatch.setattr(council, "query_model", fail_single)

    result = await council.run_full_council("What should we do?")

    assert len(result) == 5, "must unpack cleanly at main.py:831"
    stage1, stage2, stage3, metadata, evidence_pack = result
    assert stage1 == []
    assert stage2 == []
    assert stage3["model"] == "error"
    assert "failed" in stage3["response"].lower()
    assert isinstance(metadata, dict)
    assert isinstance(evidence_pack, EvidencePack)
