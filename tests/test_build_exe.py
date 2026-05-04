import importlib
import sys
from types import SimpleNamespace


def test_build_wrapper_uses_canonical_spec(monkeypatch, tmp_path):
    frontend_dist = tmp_path / "frontend" / "dist"
    frontend_dist.mkdir(parents=True)
    spec_file = tmp_path / "packaging" / "ai-advisory-board.spec"
    spec_file.parent.mkdir()
    spec_file.write_text("# spec", encoding="utf-8")

    calls = []
    fake_pyinstaller = SimpleNamespace(
        __main__=SimpleNamespace(run=lambda args: calls.append(list(args)))
    )
    monkeypatch.setitem(sys.modules, "PyInstaller", fake_pyinstaller)
    monkeypatch.setitem(sys.modules, "PyInstaller.__main__", fake_pyinstaller.__main__)

    build_exe = importlib.reload(importlib.import_module("build_exe"))

    build_exe.build(repo_root=tmp_path)

    assert calls == [["--noconfirm", "--clean", str(spec_file)]]


def test_build_wrapper_fails_before_pyinstaller_when_frontend_missing(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "PyInstaller", raising=False)
    monkeypatch.delitem(sys.modules, "PyInstaller.__main__", raising=False)
    build_exe = importlib.reload(importlib.import_module("build_exe"))

    exit_code = build_exe.build(repo_root=tmp_path)

    assert exit_code == 1
