# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"
ICON_PATH = ROOT / "app_icon.ico"

datas = [
    (str(ROOT / "backend"), "backend"),
    (str(FRONTEND_DIST), "frontend/dist"),
]
binaries = []
hiddenimports = [
    "backend",
    "backend.analytics",
    "backend.app_paths",
    "backend.attachment_storage",
    "backend.budget_policy",
    "backend.budget_router",
    "backend.config",
    "backend.council",
    "backend.execution_modes",
    "backend.file_processing",
    "backend.logger",
    "backend.main",
    "backend.model_registry",
    "backend.openrouter",
    "backend.openrouter_client",
    "backend.openrouter_pdf",
    "backend.rag",
    "backend.rag_utils",
    "backend.reasoning_stream",
    "backend.storage",
    "backend.tools",
    "backend.tools.parser",
    "backend.tools.registry",
    "backend.tools.router",
    "backend.tools.types",
    "backend.web_search",
    "bottle",
    "dotenv",
    "fastapi",
    "httpx",
    "platformdirs",
    "pydantic_core._pydantic_core",
    "sqlite3",
    "uvicorn",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
]


def collect_package(package_name, filter_submodules=lambda name: True):
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package_name,
        filter_submodules=filter_submodules,
    )
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)


collect_package(
    "webview",
    filter_submodules=lambda name: not name.startswith("webview.platforms.android"),
)

for package in ("pypdf", "docx", "pptx", "openpyxl", "bs4"):
    collect_package(package)


a = Analysis(
    [str(ROOT / "desktop.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AI Advisory Board",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON_PATH) if ICON_PATH.exists() else None,
)
