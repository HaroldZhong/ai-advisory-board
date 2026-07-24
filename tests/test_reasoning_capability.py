import importlib
import json
from datetime import datetime, timedelta, timezone

import pytest


def _cap():
    return importlib.import_module("backend.reasoning_capability")


def _probed_record(cap, model_entry, **overrides):
    rec = cap.unknown_record(model_entry["id"], cap.model_fingerprint(model_entry))
    rec.update(probed=True, probed_at="2026-07-20T00:00:00+00:00",
               supports_reasoning=True, control_surface="levels",
               levels=["low", "high"], provider_pinned="openai")
    rec.update(overrides)
    return rec


def test_missing_sidecar_yields_all_unknown(tmp_path):
    cap = _cap()
    records = cap.load_capabilities(tmp_path / "does-not-exist.json")
    assert records == {}
    rec = cap.get_capability(records, "openai/gpt-4o-mini", {"id": "openai/gpt-4o-mini"})
    assert rec["control_surface"] == "unknown"
    assert rec["supports_reasoning"] is None
    assert rec["probed"] is False


def test_unknown_record_never_reads_as_supported_or_unsupported():
    cap = _cap()
    rec = cap.unknown_record("x/y")
    assert rec["supports_reasoning"] is None  # not False
    assert rec["control_surface"] == "unknown"  # not "none"


def test_probed_record_with_matching_fingerprint_is_returned():
    cap = _cap()
    entry = {"id": "m/1", "supports_reasoning": True, "reasoning_extraction": "field"}
    records = {"m/1": _probed_record(cap, entry)}
    rec = cap.get_capability(records, "m/1", entry)
    assert rec["probed"] is True
    assert rec["control_surface"] == "levels"


def test_stale_fingerprint_is_treated_as_unknown():
    cap = _cap()
    entry = {"id": "m/1", "supports_reasoning": True, "reasoning_extraction": "field"}
    records = {"m/1": _probed_record(cap, entry)}
    # The registry entry changes materially -> fingerprint rotates -> stale.
    changed = {"id": "m/1", "supports_reasoning": True, "reasoning_extraction": "tags"}
    rec = cap.get_capability(records, "m/1", changed)
    assert rec["control_surface"] == "unknown"
    assert rec["probed"] is False


def test_unprobed_record_is_unknown():
    cap = _cap()
    entry = {"id": "m/1"}
    records = {"m/1": cap.unknown_record("m/1", cap.model_fingerprint(entry))}
    assert cap.get_capability(records, "m/1", entry)["control_surface"] == "unknown"


def test_save_load_round_trip(tmp_path):
    cap = _cap()
    entry = {"id": "m/1", "supports_reasoning": True, "reasoning_extraction": "field"}
    path = tmp_path / "sidecar.json"
    cap.save_capabilities([_probed_record(cap, entry)], path)
    records = cap.load_capabilities(path)
    assert records["m/1"]["control_surface"] == "levels"
    # persisted file is data with provenance, not planning prose
    assert "capabilities" in json.loads(path.read_text(encoding="utf-8"))


def test_age_is_a_warning_not_runtime_invalidation():
    cap = _cap()
    entry = {"id": "m/1", "supports_reasoning": True, "reasoning_extraction": "field"}
    old = _probed_record(cap, entry, probed_at="2026-06-01T00:00:00+00:00")
    records = {"m/1": old}
    now = datetime(2026, 7, 24, tzinfo=timezone.utc)

    # Age flags a maintainer warning...
    assert "m/1" in cap.stale_by_age(records, now=now, max_age_days=30)
    # ...but does NOT invalidate the record at read time (fingerprint still matches).
    assert cap.get_capability(records, "m/1", entry)["probed"] is True

    recent = _probed_record(cap, entry, probed_at=(now - timedelta(days=5)).isoformat())
    assert cap.stale_by_age({"m/1": recent}, now=now, max_age_days=30) == []


# --- default_reasoning_effort range check (A1) ---------------------------------

def _minimal_registry(default_effort):
    models = [
        {"id": f"m/{i}", "type": "both", "supports_zdr": True} for i in range(4)
    ]
    return {
        "models": models,
        "presets": [{
            "id": "p", "label": "P", "description": "d", "sort_order": 1,
            "chairman_model": "m/0",
            "council_models": ["m/1", "m/2", "m/3"],
            "default_reasoning_effort": default_effort,
        }],
    }


def test_valid_default_reasoning_effort_passes(monkeypatch):
    reg = importlib.import_module("backend.model_registry")
    reg._validate_presets(_minimal_registry("high"))  # no raise


def test_out_of_range_default_reasoning_effort_fails_load():
    reg = importlib.import_module("backend.model_registry")
    with pytest.raises(ValueError, match="default_reasoning_effort"):
        reg._validate_presets(_minimal_registry("ultra"))


def test_absent_default_reasoning_effort_is_allowed():
    reg = importlib.import_module("backend.model_registry")
    registry = _minimal_registry("high")
    del registry["presets"][0]["default_reasoning_effort"]
    reg._validate_presets(registry)  # no raise
