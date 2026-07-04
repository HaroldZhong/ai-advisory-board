# tests/test_desktop_error_format.py
"""Map raw startup tracebacks to actionable user hints (audit §4.4)."""
from desktop import format_user_error


def test_missing_key_hint():
    title, hint = format_user_error("ValueError: OPENROUTER_API_KEY environment variable is not set...")
    assert "API key" in hint
    assert "first-run setup" in hint.lower() or "setup" in hint.lower()

def test_network_hint():
    title, hint = format_user_error("httpx.ConnectError: [Errno 8] nodename nor servname")
    assert "network" in hint.lower() or "proxy" in hint.lower()

def test_port_in_use_hint():
    for tb in ("OSError: [Errno 48] Address already in use", "[WinError 10048] Only one usage"):
        title, hint = format_user_error(tb)
        assert "8001" in hint

def test_generic_fallback_points_to_log():
    title, hint = format_user_error("SomethingUnexpected: boom")
    assert "desktop.log" in hint
