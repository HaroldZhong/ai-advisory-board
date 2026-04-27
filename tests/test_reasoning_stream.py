import pytest

from backend.reasoning_stream import ReasoningStreamState


def test_openrouter_unified_emits_reasoning_and_content_separately():
    state = ReasoningStreamState("openrouter_unified")

    reasoning_events = state.consume_delta(
        {
            "reasoning_details": [
                {
                    "type": "reasoning.text",
                    "text": "Compare the constraints first.",
                    "id": "reasoning-1",
                    "format": "anthropic-claude-v1",
                    "index": 0,
                }
            ]
        }
    )
    content_events = state.consume_delta({"content": "Final answer."})

    assert reasoning_events == [
        {
            "type": "reasoning_delta",
            "text": "Compare the constraints first.",
            "detail_type": "reasoning.text",
            "format": "anthropic-claude-v1",
            "index": 0,
        }
    ]
    assert content_events == [{"type": "content_delta", "text": "Final answer."}]
    assert state.reasoning_text == "Compare the constraints first."
    assert state.content_text == "Final answer."
    assert state.reasoning_details == [
        {
            "type": "reasoning.text",
            "text": "Compare the constraints first.",
            "id": "reasoning-1",
            "format": "anthropic-claude-v1",
            "index": 0,
        }
    ]


def test_openrouter_unified_accepts_plain_reasoning_field():
    state = ReasoningStreamState("openrouter_unified")

    events = state.consume_delta({"reasoning": "Short reasoning chunk."})

    assert events == [{"type": "reasoning_delta", "text": "Short reasoning chunk."}]
    assert state.reasoning_text == "Short reasoning chunk."


def test_openrouter_unified_accepts_string_reasoning_details():
    state = ReasoningStreamState("openrouter_unified")

    events = state.consume_delta({"reasoning_details": "Detailed reasoning chunk."})

    assert events == [
        {
            "type": "reasoning_delta",
            "text": "Detailed reasoning chunk.",
            "detail_type": "reasoning.text",
        }
    ]
    assert state.reasoning_text == "Detailed reasoning chunk."
    assert state.reasoning_details == [
        {"type": "reasoning.text", "text": "Detailed reasoning chunk."}
    ]


def test_openrouter_unified_accepts_summary_reasoning_detail():
    state = ReasoningStreamState("openrouter_unified")

    events = state.consume_delta(
        {"reasoning_details": {"type": "reasoning.summary", "summary": "Summarized chain."}}
    )

    assert events == [
        {
            "type": "reasoning_delta",
            "text": "Summarized chain.",
            "detail_type": "reasoning.summary",
        }
    ]
    assert state.reasoning_text == "Summarized chain."


def test_openrouter_unified_ignores_encrypted_reasoning_without_text():
    state = ReasoningStreamState("openrouter_unified")

    events = state.consume_delta(
        {
            "reasoning_details": [
                {
                    "type": "reasoning.encrypted",
                    "data": "redacted",
                    "format": "openai-responses-v1",
                    "index": 0,
                }
            ],
            "content": "Visible answer.",
        }
    )

    assert events == [{"type": "content_delta", "text": "Visible answer."}]
    assert state.reasoning_text == ""
    assert state.reasoning_details[0]["type"] == "reasoning.encrypted"


def test_inline_tags_parse_reasoning_when_tags_span_chunks():
    state = ReasoningStreamState("inline_tags")

    events = []
    events.extend(state.consume_delta({"content": "<thi"}))
    events.extend(state.consume_delta({"content": "nk>Step one. "}))
    events.extend(state.consume_delta({"content": "Step two.</think>Final"}))
    events.extend(state.finish())

    assert events == [
        {"type": "reasoning_delta", "text": "Step one. "},
        {"type": "reasoning_delta", "text": "Step two."},
        {"type": "content_delta", "text": "Final"},
    ]
    assert state.reasoning_text == "Step one. Step two."
    assert state.content_text == "Final"


def test_inline_tags_treat_unclosed_think_block_as_reasoning_on_finish():
    state = ReasoningStreamState("inline_tags")

    events = []
    events.extend(state.consume_delta({"content": "<think>Still thinking"}))
    events.extend(state.finish())

    assert events == [{"type": "reasoning_delta", "text": "Still thinking"}]
    assert state.reasoning_text == "Still thinking"
    assert state.content_text == ""


def test_inline_tags_drop_nested_opening_tag_inside_reasoning():
    state = ReasoningStreamState("inline_tags")

    events = []
    events.extend(
        state.consume_delta({"content": "<think>outer <think>inner</think> still outer</think>Answer"})
    )
    events.extend(state.finish())

    assert events == [
        {"type": "reasoning_delta", "text": "outer "},
        {"type": "reasoning_delta", "text": "inner"},
        {"type": "reasoning_delta", "text": " still outer"},
        {"type": "content_delta", "text": "Answer"},
    ]
    assert state.reasoning_text == "outer inner still outer"
    assert state.content_text == "Answer"


def test_inline_tags_preserve_nested_depth_across_split_tags():
    state = ReasoningStreamState("inline_tags")

    events = []
    events.extend(state.consume_delta({"content": "<think>outer <th"}))
    events.extend(state.consume_delta({"content": "ink>inner</th"}))
    events.extend(state.consume_delta({"content": "ink> still outer</think>Answer"}))
    events.extend(state.finish())

    assert events == [
        {"type": "reasoning_delta", "text": "outer "},
        {"type": "reasoning_delta", "text": "inner"},
        {"type": "reasoning_delta", "text": " still outer"},
        {"type": "content_delta", "text": "Answer"},
    ]
    assert state.reasoning_text == "outer inner still outer"
    assert state.content_text == "Answer"


@pytest.mark.parametrize("format_name", ["none", None])
def test_non_reasoning_format_passes_content_through(format_name):
    state = ReasoningStreamState(format_name)

    events = state.consume_delta({"content": "Normal answer."})

    assert events == [{"type": "content_delta", "text": "Normal answer."}]
    assert state.reasoning_text == ""
    assert state.content_text == "Normal answer."
