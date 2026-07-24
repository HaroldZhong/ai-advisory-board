"""Budget Router: Decides resource allocation based on session budget and task signal."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from .config import RAG_SETTINGS  # D4: the budget-router no longer consumes BUDGET_POLICY (it shadowed the demoted bracket ladder)
from .rag_utils import detect_task_signal, get_budget_for_task_signal
from .storage import get_session_policy, get_session_usage, get_budget_spent_percentage
from .execution_modes import get_execution_mode, select_chairman_for_tier
from .logger import logger


@dataclass
class RunPlan:
    """Observable routing decision for a single message."""
    mode: str                    # "quick", "standard", "research"
    rag_preset: str              # "low", "medium", "high", "auto"
    rag_max_tokens: int          # Resolved token budget
    model_tier: str              # "budget", "mid", "premium" (for future use)
    chairman_model: Optional[str] # Resolved chairman model for this run
    predicted_cost: float        # Estimated cost in USD
    policy_reason: str           # Why this decision was made
    task_signal: str             # Detected task type
    budget_pct: Optional[float]  # Current budget spent percentage
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_run_plan(
    query: str,
    conversation_id: str,
    has_files: bool = False,
    chairman_model: str = None,
    execution_mode: str = "auto",
    rag_preset: str = "auto",
    model_tier: str = "auto",
) -> RunPlan:
    """
    Create a Run Plan based on query, budget status, and task signal.
    
    Args:
        query: User's query text
        conversation_id: Conversation ID for budget lookup
        has_files: Whether files are attached
        chairman_model: Selected chairman model (for cost estimation)
        execution_mode: Optional user override: auto/quick/standard/research
        rag_preset: Optional user override: auto/low/medium/high/max
        model_tier: Optional user override: auto/budget/mid/premium
        
    Returns:
        RunPlan with routing decisions
    """
    normalized_execution_mode = execution_mode or "auto"
    normalized_rag_preset = rag_preset or "auto"
    normalized_model_tier = model_tier or "auto"

    # 1. Detect task signal
    task_signal = detect_task_signal(query, has_files)
    logger.info("[ROUTER] Task signal: %s", task_signal)
    
    # 2. Get budget status
    budget_pct = get_budget_spent_percentage(conversation_id)
    policy = get_session_policy(conversation_id)
    has_budget = policy.get("budget_usd") is not None and policy.get("budget_usd", 0) > 0
    
    # 3. Determine policy bracket
    if budget_pct is None or not has_budget:
        # No budget set - use task signal directly
        policy_reason = "no_budget"
        rag_tokens, rag_preset = get_budget_for_task_signal(task_signal)
        mode = task_signal
    else:
        # v1.3.0 D4 (§5.1): budget brackets are ADVISORY, not enforcement. Route by
        # the task signal (what the query actually needs) at EVERY spend level; the
        # crossed bracket only ANNOTATES a suggestion via policy_reason -- it never
        # forces a mode/rag downgrade the user did not choose (they own their spend;
        # D3 made the hard cap opt-in). ZDR/model-tier routing is untouched.
        rag_tokens, rag_preset = get_budget_for_task_signal(task_signal)
        mode = task_signal
        pct = budget_pct * 100  # Convert to percentage
        if pct <= 75:
            policy_reason = "budget_under_75"
        elif pct <= 85:
            policy_reason = "budget_75_85_advisory"
        elif pct <= 100:
            policy_reason = "budget_85_100_advisory"
        else:
            policy_reason = "budget_over_100_advisory"
    
    overrides_applied = False
    if normalized_execution_mode != "auto":
        mode = normalized_execution_mode
        if normalized_rag_preset == "auto":
            rag_tokens, rag_preset = get_budget_for_task_signal(mode)
        overrides_applied = True

    if normalized_rag_preset != "auto":
        rag_preset = normalized_rag_preset
        rag_tokens = RAG_SETTINGS["presets"][rag_preset]["tokens"]
        overrides_applied = True

    resolved_model_tier = "mid"
    resolved_chairman_model = chairman_model
    if normalized_model_tier != "auto":
        resolved_model_tier = normalized_model_tier
        resolved_chairman_model = select_chairman_for_tier(
            resolved_model_tier,
            chairman_model,
        )
        overrides_applied = True

    if overrides_applied:
        policy_reason = "advanced_override"

    # 4. Estimate cost (simplified - Phase 2 uses rough estimate)
    predicted_cost = estimate_message_cost(mode, rag_tokens, resolved_chairman_model)
    
    run_plan = RunPlan(
        mode=mode,
        rag_preset=rag_preset,
        rag_max_tokens=rag_tokens,
        model_tier=resolved_model_tier,
        chairman_model=resolved_chairman_model,
        predicted_cost=predicted_cost,
        policy_reason=policy_reason,
        task_signal=task_signal,
        budget_pct=budget_pct,
    )
    
    logger.info(
        "[ROUTER] Run Plan: mode=%s, rag=%s (%d tokens), policy=%s, predicted=$%.4f",
        run_plan.mode, run_plan.rag_preset, run_plan.rag_max_tokens,
        run_plan.policy_reason, run_plan.predicted_cost
    )
    
    return run_plan


def estimate_message_cost(mode: str, rag_tokens: int, chairman_model: str = None) -> float:
    """
    Rough cost estimate for a message based on mode and RAG budget.
    
    This is intentionally conservative (overestimates).
    """
    from .config import CURATED_MODELS
    
    # Base token estimates by mode
    mode_estimates = {
        "quick": {"input": 2000, "output": 500},
        "standard": {"input": 4000, "output": 1000},
        "research": {"input": 8000, "output": 2000},
    }
    
    estimate = mode_estimates.get(mode, mode_estimates["standard"])
    total_input = estimate["input"] + rag_tokens
    total_output = estimate["output"]
    
    # Get pricing for chairman model
    input_price = 1.0  # Default $/M
    output_price = 5.0
    
    if chairman_model:
        model_config = next((m for m in CURATED_MODELS if m["id"] == chairman_model), None)
        if model_config:
            pricing = model_config.get("pricing", {})
            input_price = pricing.get("input", 1.0)
            output_price = pricing.get("output", 5.0)
    
    cost = (total_input / 1_000_000) * input_price + (total_output / 1_000_000) * output_price
    return round(cost, 6)


# v1.3.0 D3 (§5.1): rough per-call token estimates for a pre-send TURN estimate.
# Deliberately conservative and clearly APPROXIMATE -- these are heuristics, never
# measured. The honest billed total always comes from usage.cost (D1/D2).
_TURN_TOKEN_ESTIMATES = {
    "council_stage1": {"input": 4000, "output": 1200},   # each member drafts a full answer
    "council_stage2": {"input": 5000, "output": 400},     # each member ranks the others
    "chairman_stage3": {"input": 8000, "output": 1500},   # chairman synthesizes everything
    "chat": {"input": 4000, "output": 1000},              # a single chairman chat turn
}


def _model_call_cost(model_id: Optional[str], input_tokens: int, output_tokens: int) -> float:
    """USD cost of one model call at CURATED_MODELS pricing (fallback $1/$5 per M tokens)."""
    from .config import CURATED_MODELS

    model = next((m for m in CURATED_MODELS if m.get("id") == model_id), None)
    pricing = (model or {}).get("pricing", {})
    input_price = pricing.get("input", 1.0)
    output_price = pricing.get("output", 5.0)
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


def estimate_turn_cost(
    mode: str,
    council_models: Optional[list] = None,
    chairman_model: Optional[str] = None,
    rag_tokens: int = 0,
) -> float:
    """Rough, conservative pre-send USD estimate for the next turn (APPROXIMATE).

    Unlike ``estimate_message_cost`` (chairman-only), this accounts for the COUNCIL
    fan-out -- each member runs stage 1 (draft) + stage 2 (rank), then the chairman
    synthesizes (stage 3) -- so a council turn is not silently underestimated as a
    single chat call (the honesty failure a chairman-only figure would introduce).
    Never authoritative: the billed total comes from usage.cost.
    """
    from .config import CHAIRMAN_MODEL, COUNCIL_MODELS

    chairman = chairman_model or CHAIRMAN_MODEL

    if mode != "council":
        chat = _TURN_TOKEN_ESTIMATES["chat"]
        return round(_model_call_cost(chairman, chat["input"] + rag_tokens, chat["output"]), 6)

    members = council_models or COUNCIL_MODELS
    stage1 = _TURN_TOKEN_ESTIMATES["council_stage1"]
    stage2 = _TURN_TOKEN_ESTIMATES["council_stage2"]
    stage3 = _TURN_TOKEN_ESTIMATES["chairman_stage3"]

    total = 0.0
    for member in members:
        total += _model_call_cost(member, stage1["input"] + rag_tokens, stage1["output"])
        total += _model_call_cost(member, stage2["input"], stage2["output"])
    total += _model_call_cost(chairman, stage3["input"], stage3["output"])
    return round(total, 6)
