import importlib


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


def test_model_registry_loads_from_json_and_has_required_sections(monkeypatch):
    model_registry = import_module_with_api_key(monkeypatch, "backend.model_registry")

    registry = model_registry.load_model_registry()

    assert registry["chairman_model"] in {model["id"] for model in registry["models"]}
    assert len(registry["council_models"]) >= 5
    assert len(registry["models"]) >= 20
    assert len(registry["presets"]) >= 4


def test_curated_models_are_unique_and_schema_valid(monkeypatch):
    config = import_module_with_api_key(monkeypatch, "backend.config")
    valid_types = {"chairman", "council", "both", "search"}
    seen = set()

    for model in config.CURATED_MODELS:
        assert model["id"] not in seen, f"duplicate model id: {model['id']}"
        seen.add(model["id"])
        assert model["type"] in valid_types
        assert isinstance(model.get("capabilities"), list)
        assert isinstance(model.get("supports_zdr"), bool)
        assert isinstance(model.get("supports_reasoning"), bool)
        assert isinstance(model.get("default_council", False), bool)
        assert isinstance(model.get("pricing", {}).get("input"), (int, float))
        assert isinstance(model.get("pricing", {}).get("output"), (int, float))
        assert model["pricing"]["input"] >= 0
        assert model["pricing"]["output"] >= 0


def test_default_models_are_registry_available_and_not_search(monkeypatch):
    config = import_module_with_api_key(monkeypatch, "backend.config")
    by_id = {model["id"]: model for model in config.CURATED_MODELS}

    assert config.CHAIRMAN_MODEL in by_id
    assert by_id[config.CHAIRMAN_MODEL]["type"] in {"chairman", "both"}

    for model_id in config.COUNCIL_MODELS:
        assert model_id in by_id
        assert by_id[model_id]["type"] in {"council", "both"}
        assert by_id[model_id].get("default_council") is True
        assert by_id[model_id].get("available") is True


def test_model_presets_reference_registry_models(monkeypatch):
    config = import_module_with_api_key(monkeypatch, "backend.config")
    registry_ids = {model["id"] for model in config.CURATED_MODELS}
    preset_ids = set()
    valid_efforts = {"minimal", "low", "medium", "high", "xhigh"}

    assert config.MODEL_PRESETS
    assert [preset["sort_order"] for preset in config.MODEL_PRESETS] == sorted(
        preset["sort_order"] for preset in config.MODEL_PRESETS
    )
    for preset in config.MODEL_PRESETS:
        assert preset["id"] not in preset_ids
        preset_ids.add(preset["id"])
        assert preset["chairman_model"] in registry_ids
        assert preset["default_reasoning_effort"] in valid_efforts
        assert 3 <= len(preset["council_models"]) <= 8
        for model_id in preset["council_models"]:
            assert model_id in registry_ids


def test_presets_expose_expected_reasoning_effort_defaults(monkeypatch):
    config = import_module_with_api_key(monkeypatch, "backend.config")
    by_id = {preset["id"]: preset for preset in config.MODEL_PRESETS}

    assert by_id["balanced"]["default_reasoning_effort"] == "medium"
    assert by_id["research"]["default_reasoning_effort"] == "high"
    assert by_id["budget"]["default_reasoning_effort"] == "low"
    assert by_id["private"]["default_reasoning_effort"] == "medium"


def test_budget_preset_copy_explains_medium_effort_cap(monkeypatch):
    config = import_module_with_api_key(monkeypatch, "backend.config")
    by_id = {preset["id"]: preset for preset in config.MODEL_PRESETS}

    budget_description = by_id["budget"]["description"].lower()

    assert "medium" in budget_description
    assert "thinking" in budget_description


def test_private_preset_only_uses_zdr_models(monkeypatch):
    config = import_module_with_api_key(monkeypatch, "backend.config")
    by_id = {model["id"]: model for model in config.CURATED_MODELS}
    private = next(preset for preset in config.MODEL_PRESETS if preset["id"] == "private")

    assert private["requires_zdr"] is True
    assert by_id[private["chairman_model"]]["supports_zdr"] is True
    for model_id in private["council_models"]:
        assert by_id[model_id]["supports_zdr"] is True


def test_search_models_are_in_registry_for_cost_accounting(monkeypatch):
    config = import_module_with_api_key(monkeypatch, "backend.config")
    web_search = import_module_with_api_key(monkeypatch, "backend.web_search")
    by_id = {model["id"]: model for model in config.CURATED_MODELS}

    for model_id in web_search.SEARCH_MODELS.values():
        assert model_id in by_id
        assert by_id[model_id]["type"] == "search"

    main = import_module_with_api_key(monkeypatch, "backend.main")
    assert main.calculate_cost(
        {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        "perplexity/sonar",
    ) == 2.0
    assert main.calculate_cost(
        {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        "perplexity/sonar-pro",
    ) == 18.0


def test_enriched_models_expose_zdr_and_availability_metadata(monkeypatch):
    openrouter_client = import_module_with_api_key(monkeypatch, "backend.openrouter_client")

    async def fake_cached_models():
        return {
            "model/a": {
                "id": "model/a",
                "name": "Provider: Model A",
                "pricing": {"input": 1.0, "output": 2.0},
                "context_length": 1000,
                "supports_zdr": True,
            },
            "model/c": {
                "id": "model/c",
                "name": "Provider: Model C",
                "pricing": {"input": 3.0, "output": 4.0},
                "context_length": 2000,
                "supports_zdr": None,
            }
        }

    monkeypatch.setattr(openrouter_client, "get_openrouter_models_cached", fake_cached_models)

    import asyncio

    enriched = asyncio.run(openrouter_client.get_enriched_models([
        {
            "id": "model/a",
            "name": "Fallback A",
            "capabilities": ["test"],
            "type": "both",
            "supports_zdr": False,
        },
        {
            "id": "model/b",
            "name": "Fallback B",
            "capabilities": ["test"],
            "type": "council",
            "supports_zdr": True,
        },
        {
            "id": "model/c",
            "name": "Fallback C",
            "capabilities": ["test"],
            "type": "council",
            "supports_zdr": True,
        },
    ]))

    assert enriched[0]["available"] is True
    assert enriched[0]["supports_zdr"] is True
    assert enriched[0]["default_council"] is False
    assert enriched[1]["available"] is False
    assert enriched[1]["supports_zdr"] is True
    assert enriched[2]["available"] is True
    assert enriched[2]["supports_zdr"] is True


def test_models_endpoint_returns_registry_defaults(monkeypatch):
    main = import_module_with_api_key(monkeypatch, "backend.main")

    async def fake_enriched_models(models):
        return models

    monkeypatch.setattr("backend.openrouter_client.get_enriched_models", fake_enriched_models)

    import asyncio

    result = asyncio.run(main.get_models())

    assert result["defaults"]["chairman"] == main.config.CHAIRMAN_MODEL
    assert result["defaults"]["council"] == main.config.COUNCIL_MODELS
    assert result["presets"] == main.config.MODEL_PRESETS
    assert result["models"]


def test_validate_registry_finds_missing_live_ids():
    validate_registry = importlib.import_module("scripts.validate_model_registry")

    missing = validate_registry.find_missing_model_ids(
        registry={"models": [{"id": "model/a"}, {"id": "model/b"}]},
        live_response={"data": [{"id": "model/a"}]},
    )

    assert missing == ["model/b"]


def test_validate_registry_finds_reasoning_metadata_mismatches():
    validate_registry = importlib.import_module("scripts.validate_model_registry")

    mismatches = validate_registry.find_reasoning_metadata_mismatches(
        registry={
            "models": [
                {"id": "model/reasoning", "supports_reasoning": False},
                {"id": "model/plain", "supports_reasoning": True},
                {"id": "model/missing", "supports_reasoning": True},
            ]
        },
        live_reasoning_response={"data": [{"id": "model/reasoning"}]},
        live_response={
            "data": [
                {"id": "model/reasoning"},
                {"id": "model/plain"},
            ]
        },
    )

    assert mismatches == [
        {"id": "model/reasoning", "expected": True, "actual": False},
        {"id": "model/plain", "expected": False, "actual": True},
    ]
