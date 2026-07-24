import importlib


def _te():
    return importlib.import_module("backend.thinking_effort")


def test_single_source_ladder_shape():
    te = _te()
    assert te.THINKING_EFFORT_LEVELS == ("minimal", "low", "medium", "high", "xhigh")
    assert te.VALID_THINKING_EFFORTS == {"minimal", "low", "medium", "high", "xhigh"}
    assert te.THINKING_EFFORT_ORDER == {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}


def test_main_uses_the_single_source_ladder(monkeypatch):
    """main imports the ladder from thinking_effort -- the same objects, not an
    independent copy that can drift. (council retired its own ladder use in B3 when
    the Stage-3 effort cap/floor was removed, so it no longer references the ladder.)"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    te = _te()
    main = importlib.import_module("backend.main")
    council = importlib.import_module("backend.council")
    assert main.THINKING_EFFORT_ORDER is te.THINKING_EFFORT_ORDER
    assert main.VALID_THINKING_EFFORTS is te.VALID_THINKING_EFFORTS
    # council no longer defines/imports an effort ladder (B3 retired the Stage-3
    # cap/floor); guard against it silently reintroducing a divergent copy.
    assert not hasattr(council, "THINKING_EFFORT_ORDER")


def test_to_reasoning_effort_maps_levels_and_omits_auto():
    te = _te()
    # explicit level -> normalized reasoning object (correction #1)
    assert te.to_reasoning_effort("high") == {"effort": "high"}
    assert te.to_reasoning_effort("low") == {"effort": "low"}
    # Auto / native default -> None so no reasoning object is sent (correction #7)
    assert te.to_reasoning_effort(None) is None
    assert te.to_reasoning_effort("") is None
    # never fabricate an effort we can't map
    assert te.to_reasoning_effort("ultra") is None
