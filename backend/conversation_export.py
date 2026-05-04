"""Markdown export helpers for saved conversations."""

import re
from typing import Any, Dict


def get_conversation_export_filename(title: str | None) -> str:
    safe_title = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "_", title or "conversation", flags=re.IGNORECASE)
    safe_title = safe_title.lower() or "conversation"
    return f"{safe_title}.md"


def _short_model_name(model: str | None, fallback: str = "Unknown") -> str:
    if not model:
        return fallback
    return model.split("/", 1)[1] if "/" in model else model


def build_conversation_markdown(conversation: Dict[str, Any]) -> str:
    title = conversation.get("title") or "Untitled Conversation"
    created_at = conversation.get("created_at") or "Unknown date"
    messages = conversation.get("messages") or []
    metadata = conversation.get("metadata") or {}
    total_cost = conversation.get("total_cost")

    has_council_data = any(message.get("stage1") or message.get("stage2") or message.get("stage3") for message in messages)
    is_council_mode = bool(metadata.get("council_models")) or has_council_data

    markdown = f"# {title}\n"
    markdown += f"Date: {created_at}\n\n"

    if is_council_mode and metadata.get("council_models"):
        council_names = ", ".join(_short_model_name(model) for model in metadata["council_models"])
        markdown += f"**Council**: {council_names}\n"
        if metadata.get("chairman_model"):
            markdown += f"**Chairman**: {_short_model_name(metadata['chairman_model'])}\n"
        markdown += "\n"

    markdown += "---\n\n"

    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue

        if role == "user":
            markdown += f"## User\n\n{message.get('content', '')}\n\n---\n\n"
            continue

        has_stages = message.get("stage1") or message.get("stage2") or message.get("stage3")
        if has_stages:
            markdown += "## AI Advisory Board\n\n"

            if message.get("stage1"):
                markdown += "### Stage 1: Individual Responses\n\n"
                for response in message["stage1"]:
                    markdown += f"**{_short_model_name(response.get('model'))}**\n"
                    markdown += f"{response.get('response', '')}\n\n"

            if message.get("stage2"):
                markdown += "### Stage 2: Peer Rankings\n\n"
                label_to_model = (message.get("metadata") or {}).get("label_to_model") or {}
                for ranking in message["stage2"]:
                    markdown += f"**Evaluator: {_short_model_name(ranking.get('model'))}**\n"
                    markdown += f"{ranking.get('ranking', '')}\n"

                    parsed_ranking = ranking.get("parsed_ranking") or []
                    if parsed_ranking:
                        parsed = []
                        for index, label in enumerate(parsed_ranking, start=1):
                            model = label_to_model.get(label, label)
                            parsed.append(f"{index}. {_short_model_name(model, fallback=label)}")
                        markdown += f"Extracted: {', '.join(parsed)}\n"
                    markdown += "\n"

                aggregate_rankings = (message.get("metadata") or {}).get("aggregate_rankings") or []
                if aggregate_rankings:
                    markdown += "| Rank | Model | Avg | Votes |\n"
                    markdown += "|------|-------|-----|-------|\n"
                    for index, aggregate in enumerate(aggregate_rankings, start=1):
                        average = aggregate.get("average_rank")
                        average_text = f"{average:.2f}" if average is not None else "N/A"
                        markdown += (
                            f"| {index} | {_short_model_name(aggregate.get('model'))} | "
                            f"{average_text} | {aggregate.get('rankings_count', 0)} |\n"
                        )
                    markdown += "\n"

            if message.get("stage3"):
                stage3 = message["stage3"]
                markdown += "### Stage 3: Final Answer\n"
                markdown += f"**Chairman**: {_short_model_name(stage3.get('model'), fallback='Chairman')}"
                if stage3.get("confidence"):
                    markdown += f" | **Confidence**: {stage3['confidence']}"
                markdown += f"\n\n{stage3.get('response', '')}\n\n"
        else:
            markdown += f"## Assistant\n\n{message.get('content', '')}\n\n"

        running_cost = message.get("running_cost")
        if running_cost is not None and running_cost > 0:
            markdown += f"*Turn Cost: ${running_cost:.4f}*\n\n"

        markdown += "---\n\n"

    if total_cost is not None and total_cost > 0:
        markdown += f"**Total Session Cost: ${total_cost:.4f}**\n\n"

    markdown += "*Note: System messages excluded from export.*\n"
    return markdown
