# tests/test_utf8_file_io.py
"""Every text-mode open() in I/O modules must pass encoding= (audit §4.4).
AST-based so it runs identically on all platforms (no locale games needed)."""
import ast
from pathlib import Path

MODULES = [
    "backend/storage.py",
    "backend/attachment_storage.py",
    "backend/analytics.py",
    "backend/config.py",
]

def _text_mode(call: ast.Call) -> bool:
    for arg in call.args[1:2]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "b" in arg.value:
            return False
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and "b" in kw.value.value:
            return False
    return True

def _has_encoding(call: ast.Call) -> bool:
    return any(kw.arg == "encoding" for kw in call.keywords)

def test_all_text_opens_pass_encoding():
    offenders = []
    for module in MODULES:
        tree = ast.parse(Path(module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_open = isinstance(func, ast.Name) and func.id == "open"
            is_read_text = isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text")
            if is_open and _text_mode(node) and not _has_encoding(node):
                offenders.append(f"{module}:{node.lineno} open() without encoding=")
            if is_read_text and not _has_encoding(node):
                offenders.append(f"{module}:{node.lineno} {func.attr}() without encoding=")
    assert offenders == [], "\n".join(offenders)
