"""Execution Modes: Defines different execution strategies based on task and budget."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .config import CURATED_MODELS, RAG_SETTINGS
from .logger import logger


@dataclass
class ExecutionMode:
    """Configuration for an execution mode."""
    name: str
    label: str
    description: str
    rag_preset: str
    use_council: bool
    council_size: int  # 0 = chat mode, >0 = council mode
    max_output_tokens: int
    model_tier: str  # "budget", "mid", "premium"


# Execution mode definitions
EXECUTION_MODES = {
    "quick": ExecutionMode(
        name="quick",
        label="Quick Answer",
        description="Fast, concise response with minimal context",
        rag_preset="low",
        use_council=False,
        council_size=0,
        max_output_tokens=500,
        model_tier="budget",
    ),
    "standard": ExecutionMode(
        name="standard",
        label="Work Mode",
        description="Balanced response with good context",
        rag_preset="medium",
        use_council=False,
        council_size=0,
        max_output_tokens=1000,
        model_tier="mid",
    ),
    "research": ExecutionMode(
        name="research",
        label="Research Mode",
        description="Thorough response with full context",
        rag_preset="high",
        use_council=False,
        council_size=0,
        max_output_tokens=2000,
        model_tier="mid",
    ),
    "council": ExecutionMode(
        name="council",
        label="Council Deliberation",
        description="Full multi-model deliberation",
        rag_preset="auto",
        use_council=True,
        council_size=6,
        max_output_tokens=2000,
        model_tier="premium",
    ),
}


def get_execution_mode(mode_name: str) -> ExecutionMode:
    """Get execution mode configuration by name."""
    return EXECUTION_MODES.get(mode_name, EXECUTION_MODES["standard"])


def get_mode_for_task_signal(task_signal: str, is_first_message: bool = False) -> str:
    """
    Map task signal to execution mode.
    
    Args:
        task_signal: Detected task type (quick/standard/research)
        is_first_message: If True, use council mode
        
    Returns:
        Mode name
    """
    if is_first_message:
        return "council"
    
    return task_signal  # quick, standard, research map directly


# Model tier definitions
MODEL_TIERS = {
    "budget": {
        "label": "Economy",
        "description": "Fast, cost-effective models",
        "max_input_price": 0.5,  # $/M tokens
        "preferred_models": [
            "google/gemini-2.5-flash-lite",
            "openai/gpt-4.1-mini",
            "x-ai/grok-4-fast",
        ],
    },
    "mid": {
        "label": "Balanced",
        "description": "Good balance of quality and cost",
        "max_input_price": 3.0,
        "preferred_models": [
            "google/gemini-2.5-flash",
            "openai/gpt-5.1",
            "anthropic/claude-sonnet-4.5",
        ],
    },
    "premium": {
        "label": "Premium",
        "description": "Highest quality, higher cost",
        "max_input_price": 20.0,
        "preferred_models": [
            "openai/gpt-5.2",
            "anthropic/claude-opus-4.5",
            "google/gemini-3-pro-preview",
        ],
    },
}


def get_models_for_tier(tier: str) -> List[str]:
    """Get list of models for a tier."""
    tier_config = MODEL_TIERS.get(tier, MODEL_TIERS["mid"])
    return tier_config.get("preferred_models", [])


def select_chairman_for_tier(tier: str, current_chairman: str = None) -> str:
    """
    Select the best chairman model for a tier.
    
    If current_chairman is set and matches the tier, keep it.
    Otherwise, select the first preferred model for the tier.
    """
    tier_config = MODEL_TIERS.get(tier, MODEL_TIERS["mid"])
    preferred = tier_config.get("preferred_models", [])
    
    # Keep current if it matches tier preference
    if current_chairman and current_chairman in preferred:
        return current_chairman
    
    # Check current chairman pricing against tier
    if current_chairman:
        for model in CURATED_MODELS:
            if model["id"] == current_chairman:
                input_price = model.get("pricing", {}).get("input", 5.0)
                if input_price <= tier_config.get("max_input_price", 5.0):
                    return current_chairman
    
    # Return first preferred model
    if preferred:
        return preferred[0]
    
    # Fallback
    return "google/gemini-2.5-flash"


def get_execution_summary(mode: ExecutionMode) -> Dict[str, Any]:
    """Get a summary of execution mode settings for UI display."""
    rag_tokens = RAG_SETTINGS["presets"].get(mode.rag_preset, {}).get("tokens", 8000)
    
    return {
        "name": mode.name,
        "label": mode.label,
        "description": mode.description,
        "rag_tokens": rag_tokens,
        "model_tier": mode.model_tier,
        "use_council": mode.use_council,
    }
