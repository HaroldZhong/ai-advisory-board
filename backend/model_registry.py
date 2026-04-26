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
    registry.setdefault("presets", [])
    _validate_presets(registry)
    return registry


def _validate_presets(registry: Dict[str, Any]) -> None:
    """Validate preset references while keeping registry data declarative."""
    models_by_id = {
        model.get("id"): model
        for model in registry.get("models", [])
        if model.get("id")
    }
    seen_ids = set()

    for preset in registry.get("presets", []):
        preset_id = preset.get("id")
        if not preset_id:
            raise ValueError("Model preset is missing id")
        if preset_id in seen_ids:
            raise ValueError(f"Duplicate model preset id: {preset_id}")
        seen_ids.add(preset_id)

        for field in ("label", "description"):
            if not isinstance(preset.get(field), str) or not preset[field].strip():
                raise ValueError(f"Preset {preset_id} is missing {field}")
        if not isinstance(preset.get("sort_order"), (int, float)):
            raise ValueError(f"Preset {preset_id} sort_order must be numeric")

        chairman_model = preset.get("chairman_model")
        if chairman_model not in models_by_id:
            raise ValueError(f"Preset {preset_id} references unknown chairman model: {chairman_model}")
        if models_by_id[chairman_model].get("type") not in {"chairman", "both"}:
            raise ValueError(f"Preset {preset_id} chairman model cannot serve as chairman: {chairman_model}")

        council_models = preset.get("council_models")
        if not isinstance(council_models, list) or not 3 <= len(council_models) <= 8:
            raise ValueError(f"Preset {preset_id} must include 3 to 8 council models")

        for model_id in council_models:
            if model_id not in models_by_id:
                raise ValueError(f"Preset {preset_id} references unknown council model: {model_id}")
            if models_by_id[model_id].get("type") not in {"council", "both"}:
                raise ValueError(f"Preset {preset_id} council model cannot serve on council: {model_id}")

        if not isinstance(preset.get("requires_zdr", False), bool):
            raise ValueError(f"Preset {preset_id} requires_zdr must be boolean")

        if preset.get("requires_zdr"):
            if models_by_id[chairman_model].get("supports_zdr") is not True:
                raise ValueError(f"Preset {preset_id} chairman model does not support ZDR: {chairman_model}")
            for model_id in council_models:
                if models_by_id[model_id].get("supports_zdr") is not True:
                    raise ValueError(f"Preset {preset_id} council model does not support ZDR: {model_id}")

    registry["presets"] = sorted(
        registry.get("presets", []),
        key=lambda preset: preset.get("sort_order", 0),
    )
