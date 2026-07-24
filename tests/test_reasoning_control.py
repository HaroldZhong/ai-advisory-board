import importlib

import pytest


def _rc():
    return importlib.import_module("backend.reasoning_control")


def _cap(surface, **kw):
    rec = {"control_surface": surface}
    rec.update(kw)
    return rec


# --- rung d: levels -> snap to nearest supported --------------------------------

@pytest.mark.parametrize("level,expected", [("low", "low"), ("medium", "medium"), ("high", "high")])
def test_levels_full_support_passes_through(level, expected):
    rc = _rc()
    d = rc.translate_reasoning_control(_cap("levels", levels=["low", "medium", "high"]), level)
    assert d == {"kind": "effort", "effort": expected}


def test_levels_snaps_missing_medium_to_nearest():
    rc = _rc()
    # supported {low, high}; medium (rank 2) is equidistant -> nearest picks low (rank 1)
    d = rc.translate_reasoning_control(_cap("levels", levels=["low", "high"]), "medium")
    assert d["kind"] == "effort" and d["effort"] in {"low", "high"}
    # high snaps to high, low to low
    assert rc.translate_reasoning_control(_cap("levels", levels=["low", "high"]), "high")["effort"] == "high"


@pytest.mark.parametrize("level,supported,expected", [
    ("high", ["low", "medium"], "medium"),   # high->medium is nearer than high->low
    ("low", ["medium", "high"], "medium"),    # low->medium is nearer than low->high
    ("minimal", ["high", "xhigh"], "high"),   # far below range -> nearest is the floor
])
def test_levels_snaps_to_the_actually_nearest_when_not_equidistant(level, supported, expected):
    # Pins the nearest-distance branch on NON-equidistant inputs, so a min->max or
    # dropped-abs regression in _snap_to_supported can't ship green (the equidistant
    # case above accepts either side and would miss it).
    rc = _rc()
    assert rc.translate_reasoning_control(_cap("levels", levels=supported), level) == {"kind": "effort", "effort": expected}


# --- rung c: budget -> ratio, clamped ------------------------------------------

def test_budget_maps_level_to_clamped_tokens():
    rc = _rc()
    cap = _cap("budget", budget={"min": 1000, "max": 10000})
    assert rc.translate_reasoning_control(cap, "high")["max_tokens"] == 8000   # 0.8 * 10000
    assert rc.translate_reasoning_control(cap, "low")["max_tokens"] == 2000    # 0.2 * 10000
    # clamp: a tiny max floors at min
    small = _cap("budget", budget={"min": 500, "max": 500})
    assert rc.translate_reasoning_control(small, "low")["max_tokens"] == 500


# --- rung b: onoff -> explicit=on, Auto=native default -------------------------

def test_onoff_any_explicit_level_is_on():
    rc = _rc()
    assert rc.translate_reasoning_control(_cap("onoff", native_default_on=False), "low") == {"kind": "onoff", "on": True}


def test_onoff_auto_uses_native_default():
    rc = _rc()
    assert rc.translate_reasoning_control(_cap("onoff", native_default_on=True), None) == {"kind": "onoff", "on": True}
    assert rc.translate_reasoning_control(_cap("onoff", native_default_on=False), None) == {"kind": "onoff", "on": False}


# --- rung a + unknown: drop with distinct reasons ------------------------------

def test_none_drops_with_reason():
    rc = _rc()
    d = rc.translate_reasoning_control(_cap("none"), "high")
    assert d["kind"] == "drop" and "no reasoning control" in d["reason"]


def test_unknown_drops_as_not_verified():
    rc = _rc()
    d = rc.translate_reasoning_control(_cap("unknown"), "high")
    assert d["kind"] == "drop" and "not verified" in d["reason"]


# --- Auto (no level) omits an explicit effort (correction #7) -------------------

@pytest.mark.parametrize("surface", ["levels", "budget", "none", "unknown"])
def test_auto_omits_effort_for_non_onoff_surfaces(surface):
    rc = _rc()
    assert rc.translate_reasoning_control(_cap(surface), None) == {"kind": "auto"}
    assert rc.translate_reasoning_control(_cap(surface), "") == {"kind": "auto"}
