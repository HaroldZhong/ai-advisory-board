"""Validate the curated OpenRouter model registry against the live model list."""

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
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


def main() -> int:
    registry = load_registry()
    live_response = fetch_live_models()
    missing = find_missing_model_ids(registry, live_response)

    print(f"registry_models={len(registry.get('models', []))}")
    print(f"live_models={len(live_response.get('data', []))}")
    print(f"missing={len(missing)}")
    for model_id in missing:
        print(model_id)

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
