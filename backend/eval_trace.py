"""Opt-in redacted observations, not a durable run log or client delivery receipt."""
import json
import os
from datetime import datetime, timezone

from . import app_paths
from .logger import logger


def record_turn_event(event, *, run_id, mode, elapsed_ms):
    if os.getenv("AAB_EVAL_TRACE") != "1":
        return
    record = {"schema_version": 1, "event": event.get("type"), "run_id": run_id,
              "mode": mode, "elapsed_ms": round(elapsed_ms, 2),
              "observed_at": datetime.now(timezone.utc).isoformat()}
    data = event.get("data") or {}
    if event.get("type") == "turn_state":
        record["state"] = {key: data.get(key, "unknown") for key in ("generation", "persistence", "memory")}
        metadata = event.get("metadata") or {}
        snapshot = metadata.get("evidence_snapshot") or {}
        record["source_count"] = len(snapshot.get("sources", []))
        record["evidence_hash"] = snapshot.get("content_hash")
        record["citation_statuses"] = [item["status"] for item in (metadata.get("citation_checks") or {}).get("items", [])]
    elif event.get("type") == "complete":
        record["recorded_cost"] = data.get("turn_cost")
    elif event.get("type") == "steward_complete":
        record["tool_count"] = len(data.get("tools_used", []))
    elif isinstance(data, list):
        record["result_count"] = len(data)
    try:
        path = app_paths.get_logs_dir() / "turn-events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, ValueError, TypeError):
        logger.warning("[EVAL] Observation log unavailable; answer persistence is independent")
