import sys
import os

# Add current directory to path so we can import backend modules
sys.path.append(os.getcwd())

from backend.openrouter import extract_reasoning

# Registry-driven fixtures: pick one real model per capability from
# backend/model_registry.json so this test tracks the single source of truth.
NON_REASONING_MODEL = "openai/gpt-5.3-chat"  # supports_reasoning: false
FIELD_MODEL = "google/gemini-3.1-pro-preview"  # reasoning_extraction: "field"
TAGS_MODEL = "deepseek/deepseek-v4-pro"  # reasoning_extraction: "tags"
# Anthropic models get reasoning via OpenRouter's normalized fields, never
# inline <think> tags (docs: guides/best-practices/reasoning-tokens).
ANTHROPIC_MODELS = ("anthropic/claude-opus-4.7", "anthropic/claude-sonnet-4.6")


def test_capability_check():
    content = "Here is the answer. <think>Hidden reasoning</think>"
    message = {}

    clean_content, reasoning = extract_reasoning(content, message, NON_REASONING_MODEL)

    assert clean_content == content
    assert reasoning == ""


def test_field_extraction():
    content = "Final answer"
    message = {"reasoning_details": "This is the reasoning field"}

    clean_content, reasoning = extract_reasoning(content, message, FIELD_MODEL)

    assert reasoning == "This is the reasoning field"
    assert clean_content == "Final answer"


def test_field_extraction_list_of_text_blocks():
    content = "Final answer"
    message = {
        "reasoning_details": [
            {"type": "reasoning.text", "text": "Step 1"},
            {"type": "reasoning.text", "text": "Step 2"},
        ]
    }

    clean_content, reasoning = extract_reasoning(content, message, FIELD_MODEL)

    assert reasoning == "Step 1\n\nStep 2"
    assert clean_content == "Final answer"


def test_field_extraction_list_of_summary_blocks():
    content = "Final answer"
    message = {
        "reasoning_details": [
            {"type": "reasoning.summary", "summary": "Summary 1"},
            {"type": "reasoning.summary", "summary": "Summary 2"},
        ]
    }

    clean_content, reasoning = extract_reasoning(content, message, FIELD_MODEL)

    assert reasoning == "Summary 1\n\nSummary 2"
    assert clean_content == "Final answer"


def test_field_extraction_dict_without_text_or_summary_falls_back_to_str():
    content = "Final answer"
    message = {"reasoning_details": {"type": "reasoning.opaque", "value": "mystery"}}

    clean_content, reasoning = extract_reasoning(content, message, FIELD_MODEL)

    assert reasoning == str(message["reasoning_details"])
    assert clean_content == "Final answer"


def test_anthropic_models_extract_field_reasoning():
    content = "Final answer"
    message = {"reasoning": "Claude thinking arrives in the normalized field"}

    for model in ANTHROPIC_MODELS:
        clean_content, reasoning = extract_reasoning(content, message, model)
        assert reasoning == "Claude thinking arrives in the normalized field", model
        assert clean_content == "Final answer"


def test_field_mode_falls_back_to_tag_parsing():
    """A field-mode model whose provider inlines <think> blocks anyway must
    still have reasoning extracted and the tags stripped (PR #66 review)."""
    content = "<think>Inline reasoning despite field mode</think>Visible answer"
    message = {}

    clean_content, reasoning = extract_reasoning(content, message, FIELD_MODEL)

    assert clean_content == "Visible answer"
    assert "Inline reasoning despite field mode" in reasoning


def test_tag_parsing():
    content = "<think>Step 1: Think\nStep 2: Solve</think>Final Answer"
    message = {}

    clean_content, reasoning = extract_reasoning(content, message, TAGS_MODEL)

    assert clean_content == "Final Answer"
    assert "Step 1: Think" in reasoning


def test_malformed_tags():
    content = "<think>Unclosed thinking block... Final Answer"
    message = {}

    clean_content, reasoning = extract_reasoning(content, message, TAGS_MODEL)

    # Regex shouldn't match unclosed tag, so content remains same, reasoning empty
    assert clean_content == content
    assert reasoning == ""


def test_truncation():
    content = "<think>" + "A" * 3000 + "</think>Answer"
    message = {}

    clean_content, reasoning = extract_reasoning(content, message, TAGS_MODEL)

    assert len(reasoning) > 2000
    assert "truncated" in reasoning
