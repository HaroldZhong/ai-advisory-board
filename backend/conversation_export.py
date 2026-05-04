"""Markdown export helpers for saved conversations."""

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict


MAX_EXPORT_STEM_LENGTH = 200
RESERVED_WINDOWS_FILENAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _fallback_export_stem(conversation_id: str | None = None) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9]", "", conversation_id or "")[:8].lower()
    return f"conversation_{safe_id}" if safe_id else "conversation"


def _sanitize_export_stem(title: str | None) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in title or ""]
    return re.sub(r"_+", "_", "".join(chars)).strip("_")


def get_conversation_export_filename(title: str | None, conversation_id: str | None = None) -> str:
    safe_title = _sanitize_export_stem(title) or _fallback_export_stem(conversation_id)
    if safe_title.lower() in RESERVED_WINDOWS_FILENAMES:
        safe_title = f"{safe_title}_export"
    safe_title = safe_title[:MAX_EXPORT_STEM_LENGTH].rstrip("_") or _fallback_export_stem(conversation_id)
    return f"{safe_title}.md"


def resolve_unique_export_path(exports_dir: Path, filename: str) -> Path:
    path = exports_dir / filename
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        counter_suffix = f"_{counter}"
        candidate_stem = stem[: MAX_EXPORT_STEM_LENGTH - len(counter_suffix)].rstrip("_")
        candidate = exports_dir / f"{candidate_stem}{counter_suffix}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _short_model_name(model: str | None, fallback: str = "Unknown") -> str:
    if not model:
        return fallback
    return model.split("/", 1)[1] if "/" in model else model


def _format_created_at(created_at: Any) -> str:
    if not created_at:
        return "Unknown date"

    try:
        value = str(created_at)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return f"{parsed.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    except (TypeError, ValueError):
        return str(created_at)


def build_conversation_markdown(conversation: Dict[str, Any]) -> str:
    title = conversation.get("title") or "Untitled Conversation"
    created_at = _format_created_at(conversation.get("created_at"))
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
