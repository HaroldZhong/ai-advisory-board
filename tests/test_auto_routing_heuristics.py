import pytest

from backend import budget_router
from backend.rag_utils import detect_task_signal


@pytest.mark.parametrize(
    ("query", "has_files", "expected"),
    [
        ("write a quick note about the meeting", False, "quick"),
        ("explain vector databases briefly", False, "quick"),
        ("summarize this article", False, "quick"),
        ("quickly explain the key idea", False, "quick"),
        ("how do I configure model presets?", False, "standard"),
        ("research market structure for local AI tools", False, "research"),
        ("researching market structure for local AI tools", False, "research"),
        ("compare Gemini and Claude for coding", False, "research"),
        ("comparing Gemini and Claude for coding", False, "research"),
        ("cite papers about long context RAG", False, "research"),
        ("find papers about long context RAG", False, "research"),
        ("analyzing long context failure modes", False, "research"),
        ("investigate the source of this regression", False, "research"),
        ("investigating the source of this regression", False, "research"),
        ("show sources for this security claim", False, "research"),
        ("what evidence supports this roadmap?", False, "research"),
        ("review the literature on context rot", False, "research"),
        ("I attached a contract, what are the key risks?", True, "research"),
        ("x" * 205, False, "research"),
    ],
)
def test_detect_task_signal_classifies_common_user_queries(query, has_files, expected):
    assert detect_task_signal(query, has_files=has_files) == expected


@pytest.mark.parametrize(
    "query",
    [
        "paperclip import failed in Python",
        "how severe is the server shortage risk?",
    ],
)
def test_detect_task_signal_avoids_keyword_substring_false_positives(query):
    assert detect_task_signal(query) == "standard"


@pytest.mark.parametrize(
    "query",
    [
        "briefly compare these two proposals",
        "summarize this as research with sources",
    ],
)
def test_detect_task_signal_preserves_research_precedence_over_quick(query):
    assert detect_task_signal(query) == "research"


def _stub_budget(monkeypatch, pct, *, has_budget=True):
    monkeypatch.setattr(budget_router, "get_budget_spent_percentage", lambda *_: pct)
    monkeypatch.setattr(
        budget_router,
        "get_session_policy",
        lambda *_: {"budget_usd": 2.0 if has_budget else None},
    )


@pytest.mark.parametrize(
    ("query", "expected_mode", "expected_rag"),
    [
        ("summarize this article", "quick", "low"),
        ("how do I configure model presets?", "standard", "medium"),
        ("compare Gemini and Claude for coding", "research", "high"),
    ],
)
def test_create_run_plan_auto_maps_task_signals_without_budget(
    monkeypatch,
    query,
    expected_mode,
    expected_rag,
):
    _stub_budget(monkeypatch, None, has_budget=False)

    run_plan = budget_router.create_run_plan(
        query,
        conversation_id="conv-auto-routing",
    )

    assert run_plan.mode == expected_mode
    assert run_plan.rag_preset == expected_rag
    assert run_plan.policy_reason == "no_budget"


@pytest.mark.parametrize(
    ("pct", "expected_mode", "expected_rag", "expected_reason"),
    [
        (0.75, "research", "high", "budget_under_75"),
        (0.76, "standard", "medium", "budget_75_85"),
        (0.85, "standard", "medium", "budget_75_85"),
        (0.86, "quick", "low", "budget_85_100"),
        (1.00, "quick", "low", "budget_85_100"),
        (1.01, "quick", "low", "budget_over_100"),
    ],
)
def test_create_run_plan_budget_threshold_boundaries(
    monkeypatch,
    pct,
    expected_mode,
    expected_rag,
    expected_reason,
):
    _stub_budget(monkeypatch, pct)

    run_plan = budget_router.create_run_plan(
        "compare Gemini and Claude for coding",
        conversation_id="conv-budget-boundaries",
    )

    assert run_plan.mode == expected_mode
    assert run_plan.rag_preset == expected_rag
    assert run_plan.policy_reason == expected_reason


def test_create_run_plan_advanced_overrides_win_under_budget_pressure(monkeypatch):
    _stub_budget(monkeypatch, 0.93)

    run_plan = budget_router.create_run_plan(
        "compare Gemini and Claude for coding",
        conversation_id="conv-override-budget-pressure",
        chairman_model="anthropic/claude-opus-4.7",
        execution_mode="research",
        rag_preset="max",
        model_tier="budget",
    )

    assert run_plan.mode == "research"
    assert run_plan.rag_preset == "max"
    assert run_plan.rag_max_tokens == 32000
    assert run_plan.chairman_model == "google/gemini-2.5-flash-lite"
    assert run_plan.policy_reason == "advanced_override"
