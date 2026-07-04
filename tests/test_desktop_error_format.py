# tests/test_desktop_error_format.py
"""Map raw startup tracebacks to actionable user hints (audit §4.4)."""
import pytest

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

def test_real_bind_conflict_error_maps_to_port_hint():
    import socket
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    port = holder.getsockname()[1]
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError) as exc_info:
            probe.bind(("127.0.0.1", port))
    finally:
        probe.close()
        holder.close()
    title, hint = format_user_error(str(exc_info.value))
    assert "8001" in hint
