"""Filesystem paths for source and packaged desktop runs.

Development checkouts keep data project-local so contributors can inspect and
reset state without touching their profile directory. Frozen PyInstaller builds
use the per-user data directory returned by ``platformdirs`` because the
executable directory and PyInstaller extraction directory may be read-only or
ephemeral. ``AAB_DATA_DIR`` overrides both modes for tests, power users, and a
future portable-mode wrapper.

Legacy ``.env`` migration is non-destructive. Frozen builds check for the
README-documented adjacent-to-exe ``.env`` first, then the project-root ``.env``.
Only sources with a non-empty ``OPENROUTER_API_KEY`` are migrated, and sources
are left in place after the target is written.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import platformdirs
from dotenv import dotenv_values


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
        return _validate_data_root(Path(override).expanduser().resolve())
    if is_frozen():
        return _validate_data_root(Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR)).resolve())
    return _validate_data_root(PROJECT_ROOT)


def _validate_data_root(path: Path) -> Path:
    if path.exists() and not path.is_dir():
        raise ValueError(f"{DATA_DIR_ENV} must point to a directory, got file: {path}")
    return path


def get_env_path() -> Path:
    """Return the effective .env path for the current runtime."""
    return get_data_root() / ".env"


def get_conversations_dir() -> Path:
    """Return the directory used for conversation JSON files."""
    return get_data_root() / "data" / "conversations"


def get_attachments_dir() -> Path:
    """Return the directory used for attachment files and metadata."""
    return get_conversations_dir() / "attachments"


def get_pageindex_dir() -> Path:
    """Return the PageIndex data directory, shared with other data files."""
    return get_data_root() / "data"


def get_logs_dir() -> Path:
    """Return the directory used for backend and desktop logs."""
    return get_data_root() / "logs"


def get_desktop_log_path() -> Path:
    return get_logs_dir() / "desktop.log"


def _source_env_candidates() -> list[Path]:
    candidates = []
    if is_frozen():
        candidates.append(get_executable_dir() / ".env")
    candidates.append(PROJECT_ROOT / ".env")
    return candidates


def _dotenv_values_for_source(path: Path, logger: Optional[object] = None) -> Optional[dict]:
    try:
        return dotenv_values(path)
    except (OSError, UnicodeError) as exc:
        if logger is not None:
            logger.warning("Skipping .env migration from %s; failed to read source: %s", path, exc)
        return None


def _env_has_openrouter_key(path: Path, logger: Optional[object] = None) -> bool:
    values = _dotenv_values_for_source(path, logger=logger)
    if values is None:
        return False
    return bool((values.get("OPENROUTER_API_KEY") or "").strip())


def write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8", verify: bool = True) -> None:
    """Write text through a temp file, replace atomically, and verify content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding=encoding,
            newline="",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        if verify and path.read_text(encoding=encoding) != content:
            raise RuntimeError(f"Atomic write verification failed for {path}")
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def migrate_env_file(logger: Optional[object] = None) -> Optional[Path]:
    """Migrate a legacy .env into the effective app data location if needed.

    Returns the source path that was migrated, or None when no migration happened.
    The source file is intentionally left in place; migration is non-destructive.
    Frozen builds check the adjacent-to-exe .env first because the README
    documents that location for packaged users. Sources without a non-empty
    OPENROUTER_API_KEY are skipped. Source read failures warn and continue;
    target write failures raise RuntimeError for the caller to decide whether
    startup can continue.
    """
    target = get_env_path()
    if target.exists():
        return None

    candidates = _source_env_candidates()
    for source in candidates:
        if source == target or not source.exists():
            continue
        values = _dotenv_values_for_source(source, logger=logger)
        if values is None:
            continue
        if not (values.get("OPENROUTER_API_KEY") or "").strip():
            if logger is not None:
                logger.warning("Skipping .env migration from %s; OPENROUTER_API_KEY is missing", source)
            continue

        try:
            write_text_atomic(target, source.read_text(encoding="utf-8"))
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to migrate .env from {source} to {target}") from exc

        if logger is not None:
            logger.info("Migrated .env from %s to %s", source, target)
        return source

    if logger is not None:
        logger.info(
            "No legacy .env with OPENROUTER_API_KEY found in candidates: %s. First-run setup will capture the key.",
            [str(path) for path in candidates],
        )
    return None
