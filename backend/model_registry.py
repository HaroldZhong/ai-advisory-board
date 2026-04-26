"""Static model registry loader."""

import json
from pathlib import Path
from typing import Any, Dict


REGISTRY_PATH = Path(__file__).with_name("model_registry.json")


def load_model_registry(path: Path = REGISTRY_PATH) -> Dict[str, Any]:
    """Load the curated model registry from JSON."""
    with path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    registry.setdefault("models", [])
    registry.setdefault("council_models", [])
    registry.setdefault("chairman_model", "")
    return registry
