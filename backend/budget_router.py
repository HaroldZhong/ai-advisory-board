"""Budget Router: Decides resource allocation based on session budget and task signal."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from .config import RAG_SETTINGS, BUDGET_POLICY, TASK_SIGNALS
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
) -> RunPlan:
    """
    Create a Run Plan based on query, budget status, and task signal.
    
    Args:
        query: User's query text
        conversation_id: Conversation ID for budget lookup
        has_files: Whether files are attached
        chairman_model: Selected chairman model (for cost estimation)
        
    Returns:
        RunPlan with routing decisions
    """
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
        # Apply budget policy
        pct = budget_pct * 100  # Convert to percentage
        
        if pct <= 70:
            policy_reason = "budget_under_70"
            rag_tokens, rag_preset = get_budget_for_task_signal(task_signal)
            mode = task_signal
        elif pct <= 85:
            policy_reason = "budget_70_85"
            rag_preset = "medium"
            rag_tokens = RAG_SETTINGS["presets"]["medium"]["tokens"]
            mode = "standard"
        elif pct <= 100:
            policy_reason = "budget_85_100"
            rag_preset = "low"
            rag_tokens = RAG_SETTINGS["presets"]["low"]["tokens"]
            mode = "quick"
        else:
            policy_reason = "budget_over_100"
            rag_preset = "low"
            rag_tokens = RAG_SETTINGS["presets"]["low"]["tokens"]
            mode = "quick"
    
    # 4. Estimate cost (simplified - Phase 2 uses rough estimate)
    predicted_cost = estimate_message_cost(mode, rag_tokens, chairman_model)
    
    # 5. Model tier (placeholder for Phase 3)
    model_tier = "mid"  # Default for now
    
    run_plan = RunPlan(
        mode=mode,
        rag_preset=rag_preset,
        rag_max_tokens=rag_tokens,
        model_tier=model_tier,
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
