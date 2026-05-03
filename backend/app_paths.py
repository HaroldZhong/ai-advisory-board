"""Filesystem paths for source and packaged desktop runs."""

import os
import sys
from pathlib import Path
from typing import Optional

import platformdirs


APP_NAME = "AI Advisory Board"
APP_AUTHOR = "HaroldZhong"
DATA_DIR_ENV = "AAB_DATA_DIR"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """Return True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def get_executable_dir() -> Path:
    """Return the directory containing the frozen executable or Python binary."""
    return Path(sys.executable).resolve().parent


def get_data_root() -> Path:
    """Return the writable app data root for the current runtime."""
    override = os.getenv(DATA_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR)).resolve()
    return PROJECT_ROOT


def get_env_path() -> Path:
    return get_data_root() / ".env"


def get_conversations_dir() -> Path:
    return get_data_root() / "data" / "conversations"


def get_attachments_dir() -> Path:
    return get_conversations_dir() / "attachments"


def get_pageindex_dir() -> Path:
    return get_data_root() / "data"


def get_logs_dir() -> Path:
    return get_data_root() / "logs"


def get_desktop_log_path() -> Path:
    return get_logs_dir() / "desktop.log"


def _source_env_candidates() -> list[Path]:
    candidates = []
    if is_frozen():
        candidates.append(get_executable_dir() / ".env")
    candidates.append(PROJECT_ROOT / ".env")
    return candidates


def _env_has_openrouter_key(path: Path) -> bool:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("OPENROUTER_API_KEY="):
                return bool(stripped.split("=", 1)[1].strip())
    except OSError:
        return False
    return False


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def migrate_env_file(logger: Optional[object] = None) -> Optional[Path]:
    """Migrate a legacy .env into the effective app data location if needed.

    Returns the source path that was migrated, or None when no migration happened.
    The source file is intentionally left in place; migration is non-destructive.
    """
    target = get_env_path()
    if target.exists():
        return None

    for source in _source_env_candidates():
        if source == target or not source.exists():
            continue
        if not _env_has_openrouter_key(source):
            if logger is not None:
                logger.warning("Skipping .env migration from %s; OPENROUTER_API_KEY is missing", source)
            continue

        try:
            _write_text_atomic(target, source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"Failed to migrate .env from {source} to {target}") from exc

        if logger is not None:
            logger.info("Migrated .env from %s to %s", source, target)
        return source

    return None
