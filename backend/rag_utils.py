"""RAG utility functions for budget resolution and task awareness."""

from .config import RAG_SETTINGS, TASK_SIGNALS
from .logger import logger


def resolve_rag_budget(metadata: dict) -> tuple[int, str]:
    """
    Resolve RAG budget from conversation metadata.
    
    Args:
        metadata: Conversation metadata dict, may contain 'rag_budget'
        
    Returns:
        Tuple of (max_tokens, budget_key) where budget_key is for analytics/logging.
        Handles: preset names (case-insensitive), integers, invalid values, None.
    """
    raw = metadata.get("rag_budget")
    presets = RAG_SETTINGS["presets"]
    absolute_max = RAG_SETTINGS["absolute_max_tokens"]
    min_tokens = RAG_SETTINGS["min_custom_tokens"]
    default = RAG_SETTINGS["default_preset"]
    
    # Handle None or missing
    if raw is None:
        tokens = min(presets[default]["tokens"], absolute_max)
        return tokens, default
    
    # Handle string preset (case-insensitive)
    if isinstance(raw, str):
        key = raw.lower().strip()
        if key in presets:
            tokens = min(presets[key]["tokens"], absolute_max)
            return tokens, key
        # Unrecognized string → default
        logger.warning("[RAG] Unrecognized budget preset '%s', using default", raw)
        return min(presets[default]["tokens"], absolute_max), default
    
    # Handle integer (custom value)
    if isinstance(raw, int):
        clamped = min(max(raw, min_tokens), absolute_max)
        return clamped, f"custom:{clamped}"
    
    # Fallback for any other type
    logger.warning("[RAG] Invalid budget type %s, using default", type(raw).__name__)
    return min(presets[default]["tokens"], absolute_max), default


def detect_task_signal(query: str, has_files: bool = False) -> str:
    """
    Detect task signal from query using heuristics.
    
    Args:
        query: User query string
        has_files: Whether files are attached
        
    Returns:
        Task signal: "quick", "standard", or "research"
    """
    query_lower = query.lower()
    
    # Check for research indicators
    if has_files:
        return "research"
    
    if len(query) > TASK_SIGNALS["research_query_length"]:
        return "research"
    
    for keyword in TASK_SIGNALS["research_keywords"]:
        if keyword in query_lower:
            return "research"
    
    # Check for quick indicators
    for keyword in TASK_SIGNALS["quick_keywords"]:
        if keyword in query_lower:
            return "quick"
    
    # Default
    return "standard"


def get_budget_for_task_signal(task_signal: str, base_preset: str = None) -> tuple[int, str]:
    """
    Get recommended RAG budget based on task signal.
    
    Args:
        task_signal: "quick", "standard", or "research"
        base_preset: Override preset (optional)
        
    Returns:
        Tuple of (max_tokens, budget_key)
    """
    presets = RAG_SETTINGS["presets"]
    absolute_max = RAG_SETTINGS["absolute_max_tokens"]
    
    if base_preset and base_preset in presets:
        tokens = min(presets[base_preset]["tokens"], absolute_max)
        return tokens, base_preset
    
    # Map task signal to preset
    signal_to_preset = {
        "quick": "low",
        "standard": "medium",
        "research": "high",
    }
    
    preset_key = signal_to_preset.get(task_signal, "medium")
    tokens = min(presets[preset_key]["tokens"], absolute_max)
    return tokens, preset_key
