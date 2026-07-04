"""Configuration for the AI Advisory Board."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from . import app_paths
from .model_registry import load_model_registry

PROJECT_ROOT = app_paths.PROJECT_ROOT
ENV_PATH = app_paths.get_env_path()

paths_logger = logging.getLogger("LLMCouncil.paths")
try:
    app_paths.migrate_env_file(logger=paths_logger)
except RuntimeError:
    paths_logger.exception("Failed to migrate legacy .env; continuing without migrated key")
paths_logger.info(
    "Using app data root=%s env_path=%s frozen=%s override_set=%s",
    app_paths.get_data_root(),
    ENV_PATH,
    app_paths.is_frozen(),
    bool(os.getenv(app_paths.DATA_DIR_ENV)),
)
load_dotenv(ENV_PATH)


def get_openrouter_api_key() -> Optional[str]:
    """Return the current OpenRouter API key, if configured."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return api_key or None


def has_openrouter_api_key() -> bool:
    """Check whether an OpenRouter API key is configured for this process."""
    return get_openrouter_api_key() is not None


def save_openrouter_api_key(api_key: str) -> None:
    """Persist an OpenRouter API key and apply it to the current process."""
    global OPENROUTER_API_KEY

    cleaned = api_key.strip()
    if not cleaned:
        raise ValueError("API key is required")

    lines = []
    key_exists = False
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines(keepends=True)

    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        key_part = stripped.removeprefix("export ").lstrip()
        env_name = key_part.split("=", 1)[0].strip() if "=" in key_part else ""
        if env_name == "OPENROUTER_API_KEY":
            new_lines.append(f"OPENROUTER_API_KEY={cleaned}\n")
            key_exists = True
        else:
            new_lines.append(line)

    if not key_exists:
        new_lines.append(f"OPENROUTER_API_KEY={cleaned}\n")

    app_paths.write_text_atomic(ENV_PATH, "".join(new_lines))
    os.environ["OPENROUTER_API_KEY"] = cleaned
    OPENROUTER_API_KEY = cleaned


# Legacy alias for modules that still inspect this value at import time.
# Request code should call get_openrouter_api_key() instead.
OPENROUTER_API_KEY = get_openrouter_api_key()

MODEL_REGISTRY = load_model_registry()

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = MODEL_REGISTRY["council_models"]

# Models known to support reasoning/thinking
# Capabilities:
# - use_field: Look for 'reasoning' or 'reasoning_details' in API response
# - parse_tags: Look for <think> or <thinking> tags in content
REASONING_MODELS = {
    "openai/gpt-5.5": {"use_field": True},
    "openai/gpt-5.5-pro": {"use_field": True},
    "openai/gpt-5.4": {"use_field": True},
    "openai/gpt-5.4-mini": {"use_field": True},
    "anthropic/claude-opus-4.7": {"parse_tags": True},
    "anthropic/claude-sonnet-4.6": {"parse_tags": True},
    "deepseek/deepseek-v4-pro": {"parse_tags": True},
    "deepseek/deepseek-v4-flash": {"parse_tags": True},
    "moonshotai/kimi-k2.6": {"parse_tags": True},
    "z-ai/glm-5.1": {"use_field": True},
    "qwen/qwen3-max-thinking": {"parse_tags": True},
    "google/gemini-3.1-pro-preview": {"use_field": True},
}

# Curated Models Registry (for UI selection)
CURATED_MODELS = MODEL_REGISTRY["models"]

# Conversation creation presets returned by /api/models
MODEL_PRESETS = MODEL_REGISTRY["presets"]


# Legacy alias for backwards compatibility
AVAILABLE_MODELS = CURATED_MODELS

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = MODEL_REGISTRY["chairman_model"]

# Cheap fast model for titles/topics/rewrites/RAG extraction/vision description/
# PDF parsing — change here, not inline.
UTILITY_MODEL = "google/gemini-2.5-flash"

# OpenRouter API endpoint. Override the base to route through an
# OpenAI-compatible relay/proxy (e.g. where openrouter.ai is blocked).
# Blank/whitespace values fall back to the default (an optional env var
# left empty must not produce a scheme-less URL).
OPENROUTER_BASE_URL = (
    os.getenv("OPENROUTER_BASE_URL", "").strip()
    or "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_API_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

# Data directory for conversation storage
DATA_DIR = str(app_paths.get_conversations_dir())

# Phase 1 Feature Flags
ENABLE_QUERY_REWRITE = True  # Can flip to False if issues arise

# =============================================================================
# RAG CONFIGURATION
# =============================================================================
RAG_SETTINGS = {
    "default_preset": "auto",
    "absolute_max_tokens": 32000,
    "min_custom_tokens": 1000,
    "max_chunk_tokens": 1500,
    "score_threshold": 0.001,
    "presets": {
        "auto": {"tokens": 8000, "label": "Auto (recommended)"},
        "low": {"tokens": 4000, "label": "Minimal context"},
        "medium": {"tokens": 8000, "label": "Balanced"},
        "high": {"tokens": 16000, "label": "Extended context"},
        "max": {"tokens": 32000, "label": "Largest context"},
    }
}

# =============================================================================
# SESSION BUDGET CONFIGURATION
# =============================================================================
SESSION_POLICY_DEFAULTS = {
    "budget_usd": None,  # None = no budget limit (default)
    "notify_thresholds": [0.75, 0.85, 1.00],
    "mode": "auto",
    "allow_overage": True,
}

# Budget policy: strategy based on spent percentage
BUDGET_POLICY = {
    "thresholds": {
        75: {"rag_preset": "auto", "mode": "from_task"},
        85: {"rag_preset": "medium", "mode": "standard"},
        100: {"rag_preset": "low", "mode": "quick"},
    },
    "post_100_behavior": {
        "stay_minimal": True,
        "one_warning_only": True,
    },
    "quality_floor": {
        "always_respond": True,
        "min_rag_chunks": 1,
    }
}

# Task awareness heuristics (keywords for routing)
TASK_SIGNALS = {
    "research_keywords": [
        "cite",
        "cites",
        "cited",
        "citing",
        "paper",
        "papers",
        "compare",
        "compares",
        "compared",
        "comparing",
        "analyze",
        "analyzes",
        "analyzed",
        "analyzing",
        "research",
        "researches",
        "researched",
        "researching",
        "study",
        "studies",
        "studied",
        "studying",
        "investigate",
        "investigates",
        "investigated",
        "investigating",
        "sources",
        "evidence",
        "literature",
    ],
    "quick_keywords": [
        "quick",
        "quickly",
        "briefly",
        "short",
        "summary",
        "summarize",
        "recap",
        "tldr",
    ],
    "research_query_length": 200,  # chars
}
