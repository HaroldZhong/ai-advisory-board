"""Configuration for the AI Advisory Board."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Please add it to your .env file. "
        "Get your API key at https://openrouter.ai/"
    )

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4-fast",
    "moonshotai/kimi-k2-thinking",
    "deepseek/deepseek-v3.2-exp",
]

# Models known to support reasoning/thinking
# Capabilities:
# - use_field: Look for 'reasoning' or 'reasoning_details' in API response
# - parse_tags: Look for <think> or <thinking> tags in content
REASONING_MODELS = {
    "moonshotai/kimi-k2-thinking": {"parse_tags": True},
    "deepseek/deepseek-v3.2-exp": {"parse_tags": True},
    "deepseek/deepseek-chat-v3.1": {"parse_tags": True},
    "deepseek/deepseek-chat-v3-0324:free": {"parse_tags": True},
    "anthropic/claude-sonnet-4.5": {"parse_tags": True},
    "anthropic/claude-sonnet-4.0": {"parse_tags": True},
    "google/gemini-3-pro-preview": {"use_field": True},
    "google/gemini-2.5-pro": {"use_field": True},
    # Add others as needed
}

# Curated Models Registry (for UI selection)
# - id: OpenRouter model identifier
# - capabilities: Custom tags for filtering
# - type: chairman (synthesis), council (deliberation), or both
# - pricing/name: Fetched live from OpenRouter API (fallback values kept for offline use)
CURATED_MODELS = [
    # ============================================
    # CHAIRMAN TIER (highest quality, use sparingly)
    # ============================================
    {"id": "openai/gpt-5.2", "capabilities": ["frontier", "reasoning", "agentic"], "type": "chairman", "name": "GPT-5.2", "pricing": {"input": 5.0, "output": 20.0}},
    {"id": "openai/gpt-5.2-pro", "capabilities": ["frontier", "reasoning"], "type": "chairman", "name": "GPT-5.2 Pro", "pricing": {"input": 10.0, "output": 30.0}},
    {"id": "anthropic/claude-opus-4.5", "capabilities": ["frontier", "reasoning", "tool-use"], "type": "chairman", "name": "Claude Opus 4.5", "pricing": {"input": 15.0, "output": 75.0}},
    {"id": "moonshotai/kimi-k2-thinking", "capabilities": ["thinking", "long-context"], "type": "chairman", "name": "Kimi K2 Thinking", "pricing": {"input": 0.45, "output": 2.35}},
    
    # ============================================
    # COUNCIL "WORKHORSE" TIER (great quality per dollar)
    # ============================================
    # --- OpenAI ---
    {"id": "openai/gpt-5.1", "capabilities": ["reasoning", "generalist"], "type": "both", "name": "GPT-5.1", "pricing": {"input": 3.0, "output": 15.0}},
    {"id": "openai/gpt-5.2-chat", "capabilities": ["generalist", "fast"], "type": "both", "name": "GPT-5.2 Chat", "pricing": {"input": 2.5, "output": 10.0}},
    {"id": "openai/gpt-5-mini", "capabilities": ["generalist"], "type": "both", "name": "GPT-5 Mini", "pricing": {"input": 0.4, "output": 1.6}},
    {"id": "openai/gpt-4.1-mini", "capabilities": ["generalist"], "type": "council", "name": "GPT-4.1 Mini", "pricing": {"input": 0.2, "output": 0.8}},
    {"id": "openai/gpt-4o-mini", "capabilities": ["vision", "fast"], "type": "council", "name": "GPT-4o Mini", "pricing": {"input": 0.15, "output": 0.6}},
    {"id": "openai/gpt-oss-120b", "capabilities": ["open-weight", "value"], "type": "council", "name": "GPT-OSS 120B", "pricing": {"input": 0.1, "output": 0.3}},
    
    # --- Anthropic (Claude) ---
    {"id": "anthropic/claude-sonnet-4.5", "capabilities": ["vision", "reasoning", "thinking"], "type": "both", "name": "Claude Sonnet 4.5", "pricing": {"input": 3.0, "output": 15.0}},
    {"id": "anthropic/claude-haiku-4.5", "capabilities": ["fast", "value"], "type": "council", "name": "Claude Haiku 4.5", "pricing": {"input": 0.8, "output": 4.0}},
    
    # --- Google (Gemini) ---
    {"id": "google/gemini-3-pro-preview", "capabilities": ["thinking", "vision", "reasoning"], "type": "both", "name": "Gemini 3 Pro Preview", "pricing": {"input": 2.0, "output": 12.0}},
    {"id": "google/gemini-2.5-pro", "capabilities": ["reasoning", "vision"], "type": "both", "name": "Gemini 2.5 Pro", "pricing": {"input": 1.25, "output": 5.0}},
    {"id": "google/gemini-2.5-flash", "capabilities": ["vision", "fast"], "type": "both", "name": "Gemini 2.5 Flash", "pricing": {"input": 0.3, "output": 2.5}},
    {"id": "google/gemini-2.5-flash-lite", "capabilities": ["fast"], "type": "council", "name": "Gemini 2.5 Flash Lite", "pricing": {"input": 0.1, "output": 0.4}},
    {"id": "google/gemini-2.0-flash-001", "capabilities": ["fast", "vision"], "type": "council", "name": "Gemini 2.0 Flash", "pricing": {"input": 0.1, "output": 0.4}},
    {"id": "google/gemini-2.0-flash-lite-001", "capabilities": ["fast"], "type": "council", "name": "Gemini 2.0 Flash Lite", "pricing": {"input": 0.075, "output": 0.3}},
    
    # --- xAI (Grok) ---
    {"id": "x-ai/grok-4", "capabilities": ["reasoning"], "type": "both", "name": "Grok 4", "pricing": {"input": 3.0, "output": 15.0}},
    {"id": "x-ai/grok-4-fast", "capabilities": ["reasoning", "fast"], "type": "both", "name": "Grok 4 Fast", "pricing": {"input": 0.2, "output": 0.5}},
    {"id": "x-ai/grok-4.1-fast", "capabilities": ["fast"], "type": "council", "name": "Grok 4.1 Fast", "pricing": {"input": 0.2, "output": 0.5}},
    {"id": "x-ai/grok-code-fast-1", "capabilities": ["coding", "fast"], "type": "council", "name": "Grok Code Fast 1", "pricing": {"input": 0.2, "output": 1.5}},
    
    # --- DeepSeek ---
    {"id": "deepseek/deepseek-chat-v3.1", "capabilities": ["reasoning", "coding"], "type": "both", "name": "DeepSeek V3.1", "pricing": {"input": 0.2, "output": 0.8}},
    {"id": "deepseek/deepseek-v3.2-exp", "capabilities": ["reasoning", "coding", "thinking"], "type": "both", "name": "DeepSeek V3.2 Exp", "pricing": {"input": 0.216, "output": 0.328}},
    {"id": "deepseek/deepseek-r1-distill-llama-70b", "capabilities": ["reasoning", "value"], "type": "council", "name": "DeepSeek R1 Distill 70B", "pricing": {"input": 0.2, "output": 0.5}},
    
    # --- Mistral (Devstral) ---
    {"id": "mistralai/devstral-2512", "capabilities": ["coding", "agentic"], "type": "council", "name": "Devstral 2512", "pricing": {"input": 0.1, "output": 0.3}},
    
    # --- Perplexity (Web Search) ---
    {"id": "perplexity/sonar-reasoning", "capabilities": ["web-search", "research"], "type": "council", "name": "Sonar Reasoning", "pricing": {"input": 1.0, "output": 5.0}},
    
    # --- Moonshot AI (Kimi) ---
    {"id": "moonshotai/kimi-k2", "capabilities": ["long-context"], "type": "council", "name": "Kimi K2 (Instruct)", "pricing": {"input": 0.456, "output": 1.84}},
    
    # --- MiniMax ---
    {"id": "minimax/minimax-m2", "capabilities": ["roleplay"], "type": "council", "name": "MiniMax M2", "pricing": {"input": 0.08, "output": 0.6}},
    
    # --- Z.AI (GLM) ---
    {"id": "z-ai/glm-4.6", "capabilities": ["generalist"], "type": "council", "name": "GLM 4.6", "pricing": {"input": 0.2, "output": 0.8}},
    
    # --- Qwen ---
    {"id": "qwen/qwen3-coder-30b-a3b-instruct", "capabilities": ["coding"], "type": "council", "name": "Qwen3 Coder 30B", "pricing": {"input": 0.24, "output": 0.72}},
    
    # ============================================
    # FREE TIER (great for scaling, tests, fallback)
    # ============================================
    {"id": "openai/gpt-oss-120b:free", "capabilities": ["open-weight", "free"], "type": "council", "name": "GPT-OSS 120B (Free)", "pricing": {"input": 0.0, "output": 0.0}},
    {"id": "openai/gpt-oss-20b:free", "capabilities": ["open-weight", "free"], "type": "council", "name": "GPT-OSS 20B (Free)", "pricing": {"input": 0.0, "output": 0.0}},
    {"id": "mistralai/devstral-2512:free", "capabilities": ["coding", "free"], "type": "council", "name": "Devstral 2512 (Free)", "pricing": {"input": 0.0, "output": 0.0}},
]

# Legacy alias for backwards compatibility
AVAILABLE_MODELS = CURATED_MODELS

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "google/gemini-2.5-flash"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

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
    "notify_thresholds": [0.70, 0.85, 1.00],
    "mode": "auto",
    "allow_overage": True,
}

# Budget policy: strategy based on spent percentage
BUDGET_POLICY = {
    "thresholds": {
        70: {"rag_preset": "auto", "mode": "from_task"},
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
    "research_keywords": ["cite", "paper", "compare", "analyze", "research", "study"],
    "quick_keywords": ["quick", "briefly", "short", "summary", "tldr"],
    "research_query_length": 200,  # chars
}
