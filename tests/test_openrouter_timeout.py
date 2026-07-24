import asyncio
import importlib
import inspect
import json
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend import council
from backend.tools.types import EvidencePack, UsageLimits


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_query_model_enforces_wall_clock_timeout(monkeypatch):
    openrouter = import_module_with_api_key(monkeypatch, "backend.openrouter")

    class HangingAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            await asyncio.sleep(1)
            raise AssertionError("query_model did not cancel the hanging request")

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", HangingAsyncClient)

    started = time.monotonic()
    result = await openrouter.query_model(
        "anthropic/claude-opus-4.7",
        [{"role": "user", "content": "hi"}],
        timeout=0.01,
    )

    assert result is None
    assert time.monotonic() - started < 0.5


# --- C1 reliability gate (v1.3.0 plan Phase C) ---------------------------------
# C1 is verification-first: the per-member 120s timeout ALREADY exists (proved by
# test_query_model_enforces_wall_clock_timeout above), so C1 ships no new runtime
# guard — it proves the existing one covers the removed Stage-3 effort cap. Four
# rows:
#   1. per-member isolation — a slow member's deadline does not stall its peers
#   2. the 120s wall-clock default stays put when B3 deletes the cap
#   3. incident-config regression — the v1.2.0 outage config degrades cleanly
#   4. cap-absence — after B3, ANY downgrade of the requested effort fails
# Row 4's skip is gated on the legacy cap/floor SYMBOLS being present, never on the
# observed effort: skipping because "the effort was downgraded" would self-skip on
# precisely the regression the row exists to catch.


@pytest.mark.asyncio
async def test_slow_member_times_out_without_stalling_fast_members(monkeypatch):
    """C1 per-member isolation (checklist row). One member blowing its own deadline
    must not stall the others: every council member is dispatched through its OWN
    query_model call, so the deadline is per-member and the calls run concurrently.
    The slow member degrades to the clean None path while the fast members return
    normally, and the fan-out costs one deadline rather than the slow member's full
    hang. Uses the real query_model timeout (not a patched-out one) via a transport
    that hangs only for the slow model.

    Council-level aggregation (a failed member is dropped and the turn still
    returns) is covered by the Stage-1 tests in test_council_all_fail.py; this row
    owns the dispatch-level isolation C1 depends on."""
    openrouter = import_module_with_api_key(monkeypatch, "backend.openrouter")

    SLOW = "slow/member"
    FAST = ["fast/one", "fast/two"]

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class SelectiveClient:
        """Hangs only for SLOW; answers immediately for everyone else."""

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, headers=None, json=None, **kwargs):
            if json["model"] == SLOW:
                await asyncio.sleep(5)  # far beyond the per-call deadline
                raise AssertionError("slow member was not cancelled by its own deadline")
            return _FakeResponse()

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", SelectiveClient)

    started = time.monotonic()
    results = await asyncio.gather(*[
        openrouter.query_model(model, [{"role": "user", "content": "hi"}], timeout=0.05)
        for model in [SLOW, *FAST]
    ])
    elapsed = time.monotonic() - started

    slow_result, *fast_results = results
    assert slow_result is None, "the slow member must degrade to the clean None path"
    assert all(r is not None and r["content"] == "ok" for r in fast_results), (
        "fast members must complete despite a peer blowing its deadline"
    )
    assert elapsed < 1.0, (
        f"fan-out took {elapsed:.2f}s — the slow member's 5s hang leaked into the "
        "others; per-member deadlines must isolate it"
    )


def test_query_model_timeout_default_is_120s_unchanged(monkeypatch):
    """Decision-locked #6: the per-member wall-clock timeout default must stay
    120s. Removing the Stage-3 effort cap (B3) must not touch this number.
    Source-level assertion so any change to the constant trips the C1 gate."""
    openrouter = import_module_with_api_key(monkeypatch, "backend.openrouter")
    default = inspect.signature(openrouter.query_model).parameters["timeout"].default
    assert default == 120.0


@pytest.mark.asyncio
async def test_incident_config_stage3_timeout_degrades_cleanly(monkeypatch, tmp_path):
    """C1 retry/degradation regression (the incident config).

    The v1.2.0 outage config: Budget preset (chairman google/gemini-2.5-flash-lite)
    whose Stage-3 synthesis blew the 120s wall-clock. This is the DEGRADATION half
    of the C1 gate: it proves that exact config degrades cleanly under the CURRENT
    guards — no 'Unable to generate final synthesis' persisted, no failure turn, a
    retryable 500, spent cost still recorded — carried by the already-existing
    per-member timeout + the v1.2.0 clean-degradation path (retry-on-None).

    A chairman that blows its per-member deadline surfaces to the caller as
    query_model -> None (the timeout is caught there), so we simulate the timeout
    by returning None from the Stage-3 query.

    NOTE: the Stage-3 effort cap/floor (council.py) and the per-preset cap (main.py)
    are BOTH removed as of B3, so this proves clean degradation with NO effort cap in
    place — reliability rests on the per-member timeout + degradation, not on capping
    effort. The cap-DELETION itself is gated separately: the council symbols by
    test_stage3_effort_cap_and_floor_removed_after_b3, and the main.py per-preset cap
    by the thinking-effort backend suite. This test proves only 'degrades cleanly'."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    main = importlib.import_module("backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))

    # The real incident config: the Budget preset resolves the chairman to
    # google/gemini-2.5-flash-lite (backend/model_registry.json).
    conversation = await main.create_conversation(
        main.CreateConversationRequest(preset_id="budget")
    )
    conv_id = conversation["id"]

    async def fake_steward(*args, **kwargs):
        return (
            EvidencePack(run_id="r", query="q", tools_used=[], key_facts=[], limits=UsageLimits()),
            {"prompt_tokens": 1000, "completion_tokens": 1000},
        )

    async def fake_stage1_progressive(*args, **kwargs):
        result = {
            "model": "openai/gpt-4o-mini",
            "response": "Answer A",
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
        }
        yield "model_complete", 0, result
        yield "complete", [result], None

    async def fake_stage2(*args, **kwargs):
        return (
            [{
                "model": "openai/gpt-4o-mini",
                "ranking": "1. Response A",
                "parsed_ranking": ["Response A"],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
            }],
            {"Response A": "openai/gpt-4o-mini"},
        )

    async def fake_title(*args, **kwargs):
        return "title"

    stage3_models = []

    async def timing_out_stage3_chairman(model, messages, **kwargs):
        # query_model returns None on a blown deadline; record which model was
        # asked so we prove the incident chairman (flash-lite) was exercised.
        stage3_models.append(model)
        return None

    indexed = []
    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", fake_stage1_progressive)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr(council, "query_model", timing_out_stage3_chairman)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            index_session=lambda *a, **k: indexed.append(a),
            refresh_hybrid_index=lambda *a, **k: None,
        ),
    )

    with pytest.raises(HTTPException) as excinfo:
        await main.send_message(conv_id, main.SendMessageRequest(content="hi", mode="council"))

    # Clean degradation, not a crash or a persisted failure answer.
    assert excinfo.value.status_code == 500
    assert "final synthesis" in excinfo.value.detail
    # The incident chairman was actually exercised, and v1.2.0 retry-on-None is
    # not regressed: two Stage-3 attempts, both flash-lite.
    assert stage3_models == [
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash-lite",
    ]
    assert indexed == [], "a timed-out synthesis must not be RAG-indexed"

    saved = main.storage.get_conversation(conv_id)
    assert [m["role"] for m in saved["messages"]] == ["user"], "no failure turn persisted"
    assert "Unable to generate final synthesis" not in json.dumps(saved)
    assert saved["total_cost"] > 0, "spent cost (steward + stage1 + stage2) still recorded"


@pytest.mark.asyncio
async def test_stage3_high_effort_reaches_flash_lite_uncapped_after_b3(monkeypatch):
    """C1 cap-absence gate (BEHAVIORAL — the other half of the C1 gate).

    B3 removes the Stage-3 effort cap that downgraded the incident chairman. A
    symbol-name check can't catch an inline or renamed re-cap, so this exercises
    stage3_synthesize_final with the incident chairman (google/gemini-2.5-flash-lite)
    at HIGH effort and asserts the chairman's query_model call receives 'high'
    UNCAPPED. With B3 landed the legacy cap/floor symbols are gone, so this
    hard-asserts (the structural skip below is a historical fallback that no longer
    fires); an inline or renamed re-cap after B3 fails the assertion rather than
    silently passing. Green here was a prerequisite for the B3 merge (plan: B3
    gated on C1)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    captured = {}

    async def capture_effort_query(model, messages, **kwargs):
        captured["model"] = model
        captured["thinking_effort"] = kwargs.get("thinking_effort")
        return {"content": "synthesis", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    monkeypatch.setattr(council, "query_model", capture_effort_query)

    await council.stage3_synthesize_final(
        user_query="q",
        stage1_results=[{"model": "openai/gpt-4o-mini", "response": "A"}],
        stage2_results=[{"model": "openai/gpt-4o-mini", "ranking": "1. Response A"}],
        label_to_model={"Response A": "openai/gpt-4o-mini"},
        quality_metrics={},
        chairman_model="google/gemini-2.5-flash-lite",
        thinking_effort="high",
    )

    assert captured["model"] == "google/gemini-2.5-flash-lite"

    # The skip precondition is STRUCTURAL (legacy cap/floor symbols still present),
    # never behavioural. Skipping on "the effort was downgraded" would self-skip on
    # the exact regression this gate exists to catch — an inline or renamed re-cap
    # after B3 would silently skip instead of failing.
    legacy_cap_symbols = [
        name
        for name in (
            "STAGE3_THINKING_EFFORT_MAX_BY_MODEL",
            "ensure_minimum_thinking_effort",
            "cap_thinking_effort",
            "resolve_stage3_thinking_effort",
        )
        if hasattr(council, name)
    ]
    if legacy_cap_symbols:
        pytest.skip(
            "B3 has not removed the legacy Stage-3 cap/floor symbols yet "
            f"({', '.join(legacy_cap_symbols)}) — cap-absence gate pending"
        )

    # B3 has landed: ANY downgrade now fails, whatever mechanism produced it.
    assert captured["thinking_effort"] == "high", (
        f"Stage-3 effort was downgraded to {captured['thinking_effort']!r} for flash-lite "
        "after B3 removed the legacy cap/floor — an inline or renamed downgrade is a "
        "regression; the requested effort must reach the chairman uncapped"
    )


@pytest.mark.asyncio
async def test_cap_absence_gate_fails_on_post_b3_inline_recap(monkeypatch):
    """Meta-test: proves the cap-absence gate above is LOAD-BEARING, i.e. that it
    fails rather than self-skips once B3 has landed.

    An earlier version skipped whenever the effort came back downgraded — which
    self-skipped on exactly the regression it existed to catch. Here we simulate
    B3 (all legacy cap/floor symbols removed) plus an INLINE re-cap that downgrades
    flash-lite anyway, and require the gate to raise AssertionError. If the gate
    ever regresses to skipping on the observed effort, pytest.raises won't see an
    AssertionError and this row goes red."""
    for name in (
        "STAGE3_THINKING_EFFORT_MAX_BY_MODEL",
        "ensure_minimum_thinking_effort",
        "cap_thinking_effort",
        "resolve_stage3_thinking_effort",
    ):
        monkeypatch.delattr(council, name, raising=False)

    async def recapping_stage3(*args, **kwargs):
        effort = kwargs.get("thinking_effort")
        chairman = kwargs.get("chairman_model")
        if chairman == "google/gemini-2.5-flash-lite" and effort == "high":
            effort = "medium"  # inline downgrade, no legacy symbol involved
        await council.query_model(
            chairman, [{"role": "user", "content": "x"}], thinking_effort=effort
        )
        return {}

    monkeypatch.setattr(council, "stage3_synthesize_final", recapping_stage3)

    with pytest.raises(AssertionError):
        await test_stage3_high_effort_reaches_flash_lite_uncapped_after_b3(monkeypatch)
