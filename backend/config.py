"""Configuration for the LLM Council."""

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

# Available Models Registry (for UI selection)
AVAILABLE_MODELS = [
    # --- OpenAI ---
    {"id": "openai/gpt-5.1", "name": "GPT-5.1", "pricing": {"input": 3.0, "output": 15.0}, "capabilities": ["reasoning", "generalist"], "type": "both"},
    {"id": "openai/gpt-5-mini", "name": "GPT-5 Mini", "pricing": {"input": 0.4, "output": 1.6}, "capabilities": ["generalist"], "type": "both"},
    {"id": "openai/gpt-4.1-mini", "name": "GPT-4.1 Mini", "pricing": {"input": 0.2, "output": 0.8}, "capabilities": ["generalist"], "type": "council"},
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "pricing": {"input": 0.15, "output": 0.6}, "capabilities": ["vision", "fast"], "type": "council"},
    
    # --- Google (Gemini) ---
    {"id": "google/gemini-3-pro-preview", "name": "Gemini 3 Pro Preview", "pricing": {"input": 2.0, "output": 12.0}, "capabilities": ["thinking", "vision", "reasoning"], "type": "both"},
    {"id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "pricing": {"input": 1.25, "output": 5.0}, "capabilities": ["reasoning", "vision"], "type": "both"},
    {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "pricing": {"input": 0.3, "output": 2.5}, "capabilities": ["vision", "fast"], "type": "both"},
    {"id": "google/gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash Lite", "pricing": {"input": 0.1, "output": 0.4}, "capabilities": ["fast"], "type": "council"},
    {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash", "pricing": {"input": 0.1, "output": 0.4}, "capabilities": ["fast", "vision"], "type": "council"},
    {"id": "google/gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite", "pricing": {"input": 0.075, "output": 0.3}, "capabilities": ["fast"], "type": "council"},
    
    # --- xAI (Grok) ---
    {"id": "x-ai/grok-code-fast-1", "name": "Grok Code Fast 1", "pricing": {"input": 0.2, "output": 1.5}, "capabilities": ["coding", "fast"], "type": "council"},
    {"id": "x-ai/grok-4.1-fast:free", "name": "Grok 4.1 Fast (Free)", "pricing": {"input": 0.0, "output": 0.0}, "capabilities": ["research"], "type": "council"},
    {"id": "x-ai/grok-4-fast", "name": "Grok 4 Fast", "pricing": {"input": 0.2, "output": 0.5}, "capabilities": ["reasoning", "fast"], "type": "both"},
    {"id": "x-ai/grok-4", "name": "Grok 4", "pricing": {"input": 3.0, "output": 15.0}, "capabilities": ["reasoning"], "type": "both"},
    
    # --- Anthropic (Claude) ---
    {"id": "anthropic/claude-sonnet-4.5", "name": "Claude Sonnet 4.5", "pricing": {"input": 3.0, "output": 15.0}, "capabilities": ["vision", "reasoning", "thinking"], "type": "both"},
    {"id": "anthropic/claude-sonnet-4.0", "name": "Claude Sonnet 4", "pricing": {"input": 3.0, "output": 15.0}, "capabilities": ["reasoning", "vision"], "type": "both"},
    
    # --- DeepSeek ---
    {"id": "deepseek/deepseek-chat-v3.1", "name": "DeepSeek V3.1", "pricing": {"input": 0.2, "output": 0.8}, "capabilities": ["reasoning", "coding"], "type": "both"},
    {"id": "deepseek/deepseek-v3.2-exp", "name": "DeepSeek V3.2 Exp", "pricing": {"input": 0.216, "output": 0.328}, "capabilities": ["reasoning", "coding", "thinking"], "type": "both"},
    
    # --- Moonshot AI (Kimi) ---
    {"id": "moonshotai/kimi-k2", "name": "Kimi K2 (Instruct)", "pricing": {"input": 0.456, "output": 1.84}, "capabilities": ["long-context"], "type": "council"},
    {"id": "moonshotai/kimi-k2-thinking", "name": "Kimi K2 Thinking", "pricing": {"input": 0.45, "output": 2.35}, "capabilities": ["thinking", "long-context"], "type": "chairman"},
    
    # --- MiniMax ---
    {"id": "minimax/minimax-m2", "name": "MiniMax M2", "pricing": {"input": 0.08, "output": 0.6}, "capabilities": ["roleplay"], "type": "council"},
    
    # --- Zhipu AI ---
    {"id": "zhipu/glm-4.6", "name": "GLM 4.6", "pricing": {"input": 0.2, "output": 0.8}, "capabilities": ["generalist"], "type": "council"},
    
    # --- Qwen ---
    {"id": "qwen/qwen3-coder-30b-a3b-instruct", "name": "Qwen3 Coder 30B", "pricing": {"input": 0.24, "output": 0.72}, "capabilities": ["coding"], "type": "council"},
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "google/gemini-2.5-flash"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# Phase 1 Feature Flags
ENABLE_QUERY_REWRITE = True  # Can flip to False if issues arise
