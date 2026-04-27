"""Utilities for separating streamed reasoning from visible answer content."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


OPENING_THINK_TAGS = ("<thinking>", "<think>")
CLOSING_THINK_TAGS = ("</thinking>", "</think>")


class ReasoningStreamState:
    """Accumulate reasoning/content from provider stream deltas.

    The class is intentionally transport-agnostic: callers pass the decoded
    streaming delta object and receive normalized UI events.
    """

    def __init__(self, streaming_format: Optional[str] = "none"):
        self.streaming_format = streaming_format or "none"
        self.reasoning_text = ""
        self.content_text = ""
        self.reasoning_details: List[Dict[str, Any]] = []

        self._inline_buffer = ""
        self._inline_mode = "content"

    def consume_delta(self, delta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consume one provider delta and return normalized stream events."""
        if self.streaming_format == "openrouter_unified":
            return self._consume_openrouter_unified(delta)
        if self.streaming_format == "inline_tags":
            return self._consume_inline_tags(delta.get("content") or "")
        return self._consume_content(delta.get("content") or "")

    def finish(self) -> List[Dict[str, Any]]:
        """Flush buffered inline tag content at end-of-stream."""
        if self.streaming_format != "inline_tags":
            return []
        return self._drain_inline_buffer(final=True)

    def _consume_openrouter_unified(self, delta: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            events.append(self._append_reasoning(reasoning))

        reasoning_details = delta.get("reasoning_details") or []
        if isinstance(reasoning_details, dict):
            reasoning_details = [reasoning_details]

        for detail in reasoning_details:
            if not isinstance(detail, dict):
                continue

            self.reasoning_details.append(dict(detail))
            text = self._extract_reasoning_detail_text(detail)
            if not text:
                continue

            event = self._append_reasoning(text)
            event["detail_type"] = detail.get("type", "reasoning.unknown")
            if detail.get("format") is not None:
                event["format"] = detail.get("format")
            if detail.get("index") is not None:
                event["index"] = detail.get("index")
            events.append(event)

        content = delta.get("content")
        if isinstance(content, str) and content:
            events.extend(self._consume_content(content))

        return events

    def _extract_reasoning_detail_text(self, detail: Dict[str, Any]) -> str:
        if isinstance(detail.get("text"), str):
            return detail["text"]
        if isinstance(detail.get("summary"), str):
            return detail["summary"]
        return ""

    def _consume_inline_tags(self, content: str) -> List[Dict[str, Any]]:
        if not content:
            return []
        self._inline_buffer += content
        return self._drain_inline_buffer(final=False)

    def _drain_inline_buffer(self, *, final: bool) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        while self._inline_buffer:
            tag_index = self._inline_buffer.find("<")

            if tag_index < 0:
                events.append(self._append_inline_text(self._inline_buffer))
                self._inline_buffer = ""
                break

            if tag_index > 0:
                chunk = self._inline_buffer[:tag_index]
                events.append(self._append_inline_text(chunk))
                self._inline_buffer = self._inline_buffer[tag_index:]
                continue

            matched_tag = self._match_tag(OPENING_THINK_TAGS)
            if matched_tag:
                if self._inline_mode == "content":
                    self._inline_mode = "reasoning"
                self._inline_buffer = self._inline_buffer[len(matched_tag):]
                continue

            matched_tag = self._match_tag(CLOSING_THINK_TAGS)
            if matched_tag and self._inline_mode == "reasoning":
                self._inline_mode = "content"
                self._inline_buffer = self._inline_buffer[len(matched_tag):]
                continue

            candidate_tags = (
                OPENING_THINK_TAGS if self._inline_mode == "content" else CLOSING_THINK_TAGS
            )
            if not final and self._is_partial_tag(candidate_tags):
                break

            events.append(self._append_inline_text("<"))
            self._inline_buffer = self._inline_buffer[1:]

        return [event for event in events if event["text"]]

    def _append_inline_text(self, text: str) -> Dict[str, Any]:
        if self._inline_mode == "reasoning":
            return self._append_reasoning(text)
        return self._append_content(text)

    def _consume_content(self, content: str) -> List[Dict[str, Any]]:
        if not content:
            return []
        return [self._append_content(content)]

    def _append_reasoning(self, text: str) -> Dict[str, Any]:
        self.reasoning_text += text
        return {"type": "reasoning_delta", "text": text}

    def _append_content(self, text: str) -> Dict[str, Any]:
        self.content_text += text
        return {"type": "content_delta", "text": text}

    def _match_tag(self, tags: tuple[str, ...]) -> Optional[str]:
        lower_buffer = self._inline_buffer.lower()
        for tag in tags:
            if lower_buffer.startswith(tag):
                return self._inline_buffer[:len(tag)]
        return None

    def _is_partial_tag(self, tags: tuple[str, ...]) -> bool:
        lower_buffer = self._inline_buffer.lower()
        return any(tag.startswith(lower_buffer) for tag in tags)
