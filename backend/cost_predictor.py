"""Cost Predictor: Estimates marginal cost before execution."""

from typing import Dict, Any, Optional
from .config import CURATED_MODELS, RAG_SETTINGS
from .logger import logger


# Token estimates by execution mode
MODE_TOKEN_ESTIMATES = {
    "quick": {
        "base_input": 1500,   # System prompt + minimal context
        "base_output": 400,   # Shorter responses
        "council_multiplier": 0,  # No council in quick mode
    },
    "standard": {
        "base_input": 3000,
        "base_output": 800,
        "council_multiplier": 0,  # Chat mode, no council
    },
    "research": {
        "base_input": 5000,
        "base_output": 1500,
        "council_multiplier": 0,
    },
    "council": {
        "base_input": 2000,   # Per council member
        "base_output": 600,
        "council_multiplier": 6,  # Default council size
    },
}


def get_model_pricing(model_id: str) -> Dict[str, float]:
    """Get pricing for a model from config."""
    for model in CURATED_MODELS:
        if model["id"] == model_id:
            return model.get("pricing", {"input": 1.0, "output": 5.0})
    return {"input": 1.0, "output": 5.0}  # Conservative default


def estimate_chat_cost(
    rag_tokens: int,
    chairman_model: str,
    mode: str = "standard",
) -> float:
    """
    Estimate cost for a chat message.
    
    Args:
        rag_tokens: RAG context tokens
        chairman_model: Model ID for the chairman
        mode: Execution mode (quick/standard/research)
        
    Returns:
        Estimated USD cost
    """
    estimates = MODE_TOKEN_ESTIMATES.get(mode, MODE_TOKEN_ESTIMATES["standard"])
    
    total_input = estimates["base_input"] + rag_tokens
    total_output = estimates["base_output"]
    
    pricing = get_model_pricing(chairman_model)
    
    cost = (
        (total_input / 1_000_000) * pricing["input"] +
        (total_output / 1_000_000) * pricing["output"]
    )
    
    return round(cost, 6)


def estimate_council_cost(
    council_models: list,
    chairman_model: str,
    rag_tokens: int = 0,
) -> float:
    """
    Estimate cost for a full council run.
    
    Args:
        council_models: List of council model IDs
        chairman_model: Chairman model ID
        rag_tokens: RAG context tokens (usually 0 for first message)
        
    Returns:
        Estimated USD cost
    """
    estimates = MODE_TOKEN_ESTIMATES["council"]
    total_cost = 0.0
    
    # Stage 1: Each council member responds
    for model_id in council_models:
        pricing = get_model_pricing(model_id)
        stage1_cost = (
            (estimates["base_input"] / 1_000_000) * pricing["input"] +
            (estimates["base_output"] / 1_000_000) * pricing["output"]
        )
        total_cost += stage1_cost
    
    # Stage 2: Each council member ranks (longer input with all responses)
    stage2_input = estimates["base_input"] + (estimates["base_output"] * len(council_models))
    for model_id in council_models:
        pricing = get_model_pricing(model_id)
        stage2_cost = (
            (stage2_input / 1_000_000) * pricing["input"] +
            (estimates["base_output"] / 1_000_000) * pricing["output"]
        )
        total_cost += stage2_cost
    
    # Stage 3: Chairman synthesizes
    stage3_input = stage2_input * 2  # All responses + all rankings
    pricing = get_model_pricing(chairman_model)
    stage3_cost = (
        (stage3_input / 1_000_000) * pricing["input"] +
        (estimates["base_output"] * 2 / 1_000_000) * pricing["output"]
    )
    total_cost += stage3_cost
    
    logger.info("[COST] Estimated council cost: $%.4f", total_cost)
    return round(total_cost, 6)


def estimate_turn_cost(
    mode: str,
    rag_tokens: int,
    chairman_model: str,
    council_models: list = None,
    is_council_mode: bool = False,
) -> float:
    """
    Unified cost estimation for any turn.
    
    Args:
        mode: Execution mode (quick/standard/research)
        rag_tokens: RAG context tokens
        chairman_model: Chairman model ID
        council_models: Council models (for council mode)
        is_council_mode: Whether this is a council deliberation
        
    Returns:
        Estimated USD cost
    """
    if is_council_mode and council_models:
        return estimate_council_cost(council_models, chairman_model, rag_tokens)
    else:
        return estimate_chat_cost(rag_tokens, chairman_model, mode)
