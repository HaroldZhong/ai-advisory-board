"""Budget Policy: Decision rules for resource allocation based on budget status."""

from typing import Dict, Any, Tuple
from .config import BUDGET_POLICY, RAG_SETTINGS
from .logger import logger


def get_policy_bracket(spent_pct: float) -> str:
    """
    Determine which budget policy bracket applies.
    
    Args:
        spent_pct: Percentage of budget spent (0.0 to N, can exceed 1.0)
        
    Returns:
        Bracket name: "normal", "balanced", "reduced", "minimal"
    """
    if spent_pct is None:
        return "normal"
    
    pct = spent_pct * 100  # Convert to percentage
    
    if pct <= 70:
        return "normal"
    elif pct <= 85:
        return "balanced"
    elif pct <= 100:
        return "reduced"
    else:
        return "minimal"


def get_policy_settings(bracket: str) -> Dict[str, Any]:
    """
    Get policy settings for a budget bracket.
    
    Returns dict with:
        - rag_preset: RAG budget preset
        - mode: Execution mode
        - allow_council: Whether council mode is allowed
    """
    policies = {
        "normal": {
            "rag_preset": "auto",
            "mode": "from_task",  # Use task signal
            "allow_council": True,
        },
        "balanced": {
            "rag_preset": "medium",
            "mode": "standard",
            "allow_council": True,
        },
        "reduced": {
            "rag_preset": "low",
            "mode": "quick",
            "allow_council": False,  # Suggest chat instead
        },
        "minimal": {
            "rag_preset": "low",
            "mode": "quick",
            "allow_council": False,
        },
    }
    return policies.get(bracket, policies["normal"])


def apply_budget_policy(
    task_signal: str,
    spent_pct: float,
    is_first_message: bool = False,
) -> Tuple[str, str, int, str]:
    """
    Apply budget policy to determine execution parameters.
    
    Args:
        task_signal: Detected task type (quick/standard/research)
        spent_pct: Percentage of budget spent
        is_first_message: Whether this is the first message (council always allowed)
        
    Returns:
        Tuple of (mode, rag_preset, rag_tokens, policy_reason)
    """
    bracket = get_policy_bracket(spent_pct)
    settings = get_policy_settings(bracket)
    
    # Determine mode
    if settings["mode"] == "from_task":
        mode = task_signal
    else:
        mode = settings["mode"]
    
    # Get RAG tokens
    rag_preset = settings["rag_preset"]
    if rag_preset == "auto":
        # Map task signal to preset
        task_to_preset = {
            "quick": "low",
            "standard": "medium",
            "research": "high",
        }
        rag_preset = task_to_preset.get(task_signal, "medium")
    
    rag_tokens = RAG_SETTINGS["presets"].get(rag_preset, {}).get("tokens", 8000)
    
    policy_reason = f"budget_{bracket}"
    
    logger.info(
        "[POLICY] Bracket=%s, mode=%s, rag=%s (%d tokens)",
        bracket, mode, rag_preset, rag_tokens
    )
    
    return mode, rag_preset, rag_tokens, policy_reason


def should_suggest_chat(spent_pct: float) -> bool:
    """
    Check if we should suggest chat mode instead of council.
    
    Returns True if budget is tight and chat would be more economical.
    """
    if spent_pct is None:
        return False
    
    bracket = get_policy_bracket(spent_pct)
    settings = get_policy_settings(bracket)
    return not settings["allow_council"]


def get_quality_floor_settings() -> Dict[str, Any]:
    """
    Get the minimum quality settings (never go below these).
    """
    return BUDGET_POLICY.get("quality_floor", {
        "always_respond": True,
        "min_rag_chunks": 1,
    })
