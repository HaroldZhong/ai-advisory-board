import importlib


def _te():
    return importlib.import_module("backend.thinking_effort")


def test_single_source_ladder_shape():
    te = _te()
    assert te.THINKING_EFFORT_LEVELS == ("minimal", "low", "medium", "high", "xhigh")
    assert te.VALID_THINKING_EFFORTS == {"minimal", "low", "medium", "high", "xhigh"}
    assert te.THINKING_EFFORT_ORDER == {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}


def test_main_uses_the_single_source_ladder(monkeypatch):
    """main imports its ladder validation set from thinking_effort -- the same
    object, not an independent copy that can drift.

    B3 retired the per-preset effort cap, which was main's only consumer of the
    ordinal THINKING_EFFORT_ORDER, so main now needs only VALID_THINKING_EFFORTS.
    Neither main nor council may reintroduce a divergent effort-order ladder (main's
    per-preset cap and council's Stage-3 cap/floor are both gone)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    te = _te()
    main = importlib.import_module("backend.main")
    council = importlib.import_module("backend.council")
    assert main.VALID_THINKING_EFFORTS is te.VALID_THINKING_EFFORTS
    # Load-bearing for B3: the effort-order ladder is gone from both modules; guard
    # against either silently reintroducing a divergent copy (a resurrected cap/floor).
    assert not hasattr(main, "THINKING_EFFORT_ORDER")
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
