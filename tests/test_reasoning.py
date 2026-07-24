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


def test_field_mode_never_strips_tag_markup_from_answers():
    """Field-mode answers that merely mention <think> markup keep it verbatim —
    tag parsing stays gated to tags-mode models (PR #66 review, round 3)."""
    content = "<think>example of a thinking tag</think>And the visible answer"
    message = {}

    clean_content, reasoning = extract_reasoning(content, message, FIELD_MODEL)

    assert clean_content == content
    assert reasoning == ""


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


# --- B5: per-member post-turn reasoning actuals (honesty: tokens, never text) ---

def test_reasoning_tokens_from_usage_reports_count_or_none():
    from backend.openrouter import reasoning_tokens_from_usage
    assert reasoning_tokens_from_usage({"completion_tokens_details": {"reasoning_tokens": 42}}) == 42
    # explicit 0, absent, or non-numeric -> "not available" (None)
    assert reasoning_tokens_from_usage({"completion_tokens_details": {"reasoning_tokens": 0}}) is None
    assert reasoning_tokens_from_usage({}) is None
    assert reasoning_tokens_from_usage(None) is None
    assert reasoning_tokens_from_usage({"completion_tokens_details": {"reasoning_tokens": "x"}}) is None


def test_build_stage1_result_keys_actuals_on_tokens_not_reasoning_text():
    from backend.council import build_stage1_result
    # the honesty trap: reasoning TEXT present but reasoning_tokens == 0 must still
    # report "not available" (None), never inferred from the text.
    trap = build_stage1_result("m/1", {
        "content": "answer",
        "reasoning_details": "I reasoned about it",
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
    })
    assert trap["reasoning_tokens"] is None
    assert trap.get("reasoning")  # the text is still carried separately (unchanged)
    # real reasoning tokens -> the count is reported
    real = build_stage1_result("m/2", {
        "content": "answer",
        "usage": {"completion_tokens_details": {"reasoning_tokens": 128}},
    })
    assert real["reasoning_tokens"] == 128


# --- correction #2: single capability authority (sidecar-first, registry fallback) ---

def test_model_supports_reasoning_single_authority():
    from backend import openrouter
    from backend.reasoning_capability import model_fingerprint

    model_id = "openai/gpt-5.3-chat"  # a real registry model
    entry = openrouter._lookup_registry_model(model_id)
    assert entry is not None
    registry_verdict = entry.get("supports_reasoning") is True

    # 1. UN-PROBED (empty sidecar) -> DEPRECATED registry fallback (non-regressing)
    assert openrouter.model_supports_reasoning(model_id, capability_records={}) is registry_verdict

    # 2. a PROBED record OVERRIDES the registry -- levels surface -> True
    probed_true = {"model_id": model_id, "probed": True,
                   "fingerprint": model_fingerprint(entry),
                   "control_surface": "levels", "supports_reasoning": True}
    assert openrouter.model_supports_reasoning(model_id, capability_records={model_id: probed_true}) is True

    # 3. a PROBED "none" surface -> False, overriding the registry the other way
    probed_none = {"model_id": model_id, "probed": True,
                   "fingerprint": model_fingerprint(entry),
                   "control_surface": "none", "supports_reasoning": False}
    assert openrouter.model_supports_reasoning(model_id, capability_records={model_id: probed_none}) is False

    # 4. STALE fingerprint -> not authoritative -> registry fallback again
    stale = {**probed_true, "fingerprint": "STALE"}
    assert openrouter.model_supports_reasoning(model_id, capability_records={model_id: stale}) is registry_verdict


def test_model_supports_reasoning_runtime_path_uses_cached_sidecar():
    # The runtime call (no explicit records) reads the checked-in sidecar (empty
    # scaffold today), so it falls back to the registry -- proving the rewire is
    # non-regressing until the probe populates the sidecar.
    from backend import openrouter
    openrouter.reset_reasoning_capability_cache()
    model_id = "openai/gpt-5.3-chat"
    entry = openrouter._lookup_registry_model(model_id)
    assert openrouter.model_supports_reasoning(model_id) is (entry.get("supports_reasoning") is True)


def test_resolve_model_reasoning_uses_probed_surface_and_pins_endpoint():
    from backend import openrouter
    from backend.reasoning_capability import model_fingerprint
    model_id = FIELD_MODEL  # registry reasoning_extraction == "field" -> runtime CAN parse
    entry = openrouter._lookup_registry_model(model_id)
    fp = model_fingerprint(entry)

    def rec(**kw):
        base = {"model_id": model_id, "probed": True, "fingerprint": fp,
                "provider_pinned": "google", "supports_reasoning": True}
        base.update(kw)
        return {model_id: base}

    # each probed surface yields its correct shape AND pins the endpoint it was learned on
    for surface_kw, expected in (
        (dict(control_surface="levels", levels=["low", "medium", "high"]), {"effort": "high"}),
        (dict(control_surface="onoff", native_default_on=False), {"enabled": True}),
        (dict(control_surface="budget", budget={"min": 1000, "max": 10000}), {"max_tokens": 8000}),
    ):
        payload, pin = openrouter.resolve_model_reasoning(model_id, "high", capability_records=rec(**surface_kw))
        assert payload == expected and pin == "google"
    # none -> omit entirely; no shape sent -> no pin needed
    payload, pin = openrouter.resolve_model_reasoning(
        model_id, "high", capability_records=rec(control_surface="none", supports_reasoning=False))
    assert payload is None and pin is None


def test_resolve_model_reasoning_omits_shape_the_runtime_cannot_extract():
    # The sidecar can flip a model to reasoning-supported, but if the runtime has no
    # extraction mode for it (registry reasoning_extraction unset) it would burn
    # reasoning tokens while dropping the text -> omit the shape instead.
    from backend import openrouter
    from backend.reasoning_capability import model_fingerprint
    model_id = NON_REASONING_MODEL  # registry reasoning_extraction is unset
    entry = openrouter._lookup_registry_model(model_id)
    assert entry.get("reasoning_extraction") not in ("field", "tags")  # precondition
    probed = {model_id: {"model_id": model_id, "probed": True, "fingerprint": model_fingerprint(entry),
                         "provider_pinned": "openai", "supports_reasoning": True,
                         "control_surface": "levels", "levels": ["low", "medium", "high"]}}
    assert openrouter.resolve_model_reasoning(model_id, "high", capability_records=probed) == (None, None)


def test_resolve_model_reasoning_unprobed_fallback_is_non_regressing():
    from backend import openrouter
    model_id = "openai/gpt-5.3-chat"
    entry = openrouter._lookup_registry_model(model_id)
    registry_supported = entry.get("supports_reasoning") is True
    # un-probed + effort -> plain effort object iff registry supports it; NO pin (default routing)
    payload, pin = openrouter.resolve_model_reasoning(model_id, "high", capability_records={})
    assert payload == ({"effort": "high"} if registry_supported else None)
    assert pin is None
    # Auto (no effort) + un-probed -> no reasoning object at all
    payload, pin = openrouter.resolve_model_reasoning(model_id, None, capability_records={})
    assert payload is None and pin is None


def test_zdr_turns_use_neither_the_probe_pin_nor_its_endpoint_specific_shape():
    """A probed row describes whichever endpoint ORDINARY (non-ZDR) routing selected,
    and /endpoints publishes no per-endpoint ZDR field, so it says nothing about the
    endpoint a ZDR turn reaches. Pinning it would intersect with zdr:true under
    allow_fallbacks:false and yield NO ROUTE; sending its native shape to a different,
    unverified ZDR endpoint is the mis-application the pin exists to prevent. ZDR
    turns therefore fall back to the normalized reasoning.effort interface."""
    from backend import openrouter
    from backend.reasoning_capability import model_fingerprint
    model_id = FIELD_MODEL
    entry = openrouter._lookup_registry_model(model_id)
    records = {model_id: {
        "model_id": model_id, "probed": True, "fingerprint": model_fingerprint(entry),
        "provider_pinned": "google", "supports_reasoning": True,
        "control_surface": "onoff", "native_default_on": False,
    }}

    # Ordinary turn: the probed native shape AND its pin are both applied.
    payload, pin = openrouter.resolve_model_reasoning(
        model_id, "high", capability_records=records)
    assert payload == {"enabled": True} and pin == "google"

    # ZDR turn: no pin, and NOT the endpoint-specific shape -- the normalized
    # interface instead (preflight rule 1), which is provider-agnostic.
    payload, pin = openrouter.resolve_model_reasoning(
        model_id, "high", capability_records=records, zdr_enabled=True)
    assert pin is None, "a ZDR request must never be narrowed to a probed endpoint"
    assert payload == {"effort": "high"}, payload
