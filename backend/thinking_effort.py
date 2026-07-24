"""Single-source reasoning-effort ladder (v1.3.0 B1).

The ordinal effort ladder was duplicated across backend/main.py, backend/council.py
(and mirrored in frontend/src/utils/thinkingEffort.js). This is the one backend
authority; main and council import from here instead of each defining their own.

Correction #1 (normalized interface) + #7 (Auto = omit): `to_reasoning_effort()`
maps a universal level to OpenRouter's normalized `reasoning.effort` object, and
returns None for the Auto/native-default state so NO reasoning object is sent.
"""
from typing import Dict, Optional

# Ordinal ladder. The universal UI knob uses low/medium/high; minimal/xhigh exist
# for models that expose them.
THINKING_EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh")
VALID_THINKING_EFFORTS = set(THINKING_EFFORT_LEVELS)
THINKING_EFFORT_ORDER = {level: index for index, level in enumerate(THINKING_EFFORT_LEVELS)}


def to_reasoning_effort(level: Optional[str]) -> Optional[Dict[str, str]]:
    """OpenRouter normalized `reasoning` object for a universal level, or None for
    the Auto/native-default state.

    Returns None when `level` is None/empty (Auto -> send no reasoning object,
    correction #7) or is not a known effort (never fabricate an effort we cannot
    map). Otherwise the normalized object `{"effort": level}` (correction #1).
    """
    if level and level in VALID_THINKING_EFFORTS:
        return {"effort": level}
    return None
