import importlib
import sys
import threading
from types import SimpleNamespace


def test_desktop_startup_url_loads_app_route_not_landing(monkeypatch):
    """Regression: packaged desktop launch must open the app, not the landing page."""
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *args, **kwargs: None))
    monkeypatch.setitem(
        sys.modules,
        "webview",
        SimpleNamespace(
            create_window=lambda *args, **kwargs: None,
            start=lambda *args, **kwargs: None,
        ),
    )

    desktop = importlib.reload(importlib.import_module("desktop"))

    assert desktop.get_app_url() == "http://127.0.0.1:8001/app"


def test_desktop_main_navigates_webview_to_app_route(monkeypatch):
    """Regression: pywebview startup must navigate to /app, not the landing page."""
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *args, **kwargs: None))

    loaded_urls = []
    loaded = threading.Event()
    fake_window = SimpleNamespace(
        load_url=lambda url: (loaded_urls.append(url), loaded.set()),
        load_html=lambda html: None,
    )
    fake_webview = SimpleNamespace(
        create_window=lambda *args, **kwargs: fake_window,
        start=lambda *args, **kwargs: loaded.wait(timeout=2),
    )
    monkeypatch.setitem(sys.modules, "webview", fake_webview)

    desktop = importlib.reload(importlib.import_module("desktop"))
    null_logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(desktop, "configure_logging", lambda: null_logger)
    monkeypatch.setattr(desktop, "wait_for_port", lambda *args, **kwargs: True)

    def fake_start_server():
        desktop.server_ready.set()

    monkeypatch.setattr(desktop, "start_server", fake_start_server)

    desktop.main()

    assert loaded_urls == ["http://127.0.0.1:8001/app"]
