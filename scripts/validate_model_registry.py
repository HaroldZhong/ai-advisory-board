"""Validate the curated OpenRouter model registry against the live model list."""

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_REASONING_MODELS_URL = "https://openrouter.ai/api/v1/models?supported_parameters=reasoning"
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "backend" / "model_registry.json"


def load_registry(path: Path = REGISTRY_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_live_models(url: str = OPENROUTER_MODELS_URL) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def find_missing_model_ids(registry: Dict[str, Any], live_response: Dict[str, Any]) -> List[str]:
    live_ids = {model.get("id") for model in live_response.get("data", [])}
    return [
        model.get("id", "")
        for model in registry.get("models", [])
        if model.get("id") not in live_ids
    ]


def find_reasoning_metadata_mismatches(
    registry: Dict[str, Any],
    live_reasoning_response: Dict[str, Any],
    live_response: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    reasoning_ids = {model.get("id") for model in live_reasoning_response.get("data", [])}
    live_ids = None
    if live_response is not None:
        live_ids = {model.get("id") for model in live_response.get("data", [])}
    mismatches = []

    for model in registry.get("models", []):
        model_id = model.get("id")
        if live_ids is not None and model_id not in live_ids:
            continue
        expected = model_id in reasoning_ids
        actual = model.get("supports_reasoning")
        if expected != actual:
            mismatches.append({"id": model_id, "expected": expected, "actual": actual})

    return mismatches


def main() -> int:
    registry = load_registry()
    live_response = fetch_live_models()
    live_reasoning_response = fetch_live_models(OPENROUTER_REASONING_MODELS_URL)
    missing = find_missing_model_ids(registry, live_response)
    reasoning_mismatches = find_reasoning_metadata_mismatches(
        registry,
        live_reasoning_response,
        live_response,
    )

    print(f"registry_models={len(registry.get('models', []))}")
    print(f"live_models={len(live_response.get('data', []))}")
    print(f"missing={len(missing)}")
    for model_id in missing:
        print(model_id)
    print(f"reasoning_mismatches={len(reasoning_mismatches)}")
    for mismatch in reasoning_mismatches:
        print(
            f"{mismatch['id']} supports_reasoning={mismatch['actual']} "
            f"expected={mismatch['expected']}"
        )

    return 1 if missing or reasoning_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
