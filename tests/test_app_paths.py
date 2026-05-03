import importlib
import sys
from pathlib import Path

import pytest


class CapturingLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        self.warnings.append(args)


def import_app_paths():
    return importlib.import_module("backend.app_paths")


def clear_frozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)


def test_env_override_wins_in_dev_and_frozen(monkeypatch, tmp_path):
    app_paths = import_app_paths()
    override = tmp_path / "override-data"
    platform_default = tmp_path / "platform-data"

    monkeypatch.setenv("AAB_DATA_DIR", str(override))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        app_paths.platformdirs,
        "user_data_dir",
        lambda *args, **kwargs: str(platform_default),
    )

    assert app_paths.get_data_root() == override
    assert app_paths.get_env_path() == override / ".env"
    assert app_paths.get_conversations_dir() == override / "data" / "conversations"
    assert app_paths.get_attachments_dir() == override / "data" / "conversations" / "attachments"
    assert app_paths.get_pageindex_dir() == override / "data"
    assert app_paths.get_logs_dir() == override / "logs"
    assert app_paths.get_desktop_log_path() == override / "logs" / "desktop.log"


def test_dev_default_uses_project_root(monkeypatch):
    app_paths = import_app_paths()
    clear_frozen(monkeypatch)
    monkeypatch.delenv("AAB_DATA_DIR", raising=False)

    assert app_paths.get_data_root() == app_paths.PROJECT_ROOT
    assert app_paths.get_env_path() == app_paths.PROJECT_ROOT / ".env"


def test_frozen_default_uses_platform_user_data_dir(monkeypatch, tmp_path):
    app_paths = import_app_paths()
    frozen_root = tmp_path / "user-data"

    monkeypatch.delenv("AAB_DATA_DIR", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        app_paths.platformdirs,
        "user_data_dir",
        lambda appname, appauthor, **kwargs: str(frozen_root),
    )

    assert app_paths.get_data_root() == frozen_root
    assert app_paths.get_env_path() == frozen_root / ".env"


def test_env_migration_prefers_adjacent_exe_over_project_root(monkeypatch, tmp_path):
    app_paths = import_app_paths()
    logger = CapturingLogger()
    exe_dir = tmp_path / "exe"
    project_root = tmp_path / "project"
    data_root = tmp_path / "user-data"
    exe_dir.mkdir()
    project_root.mkdir()

    adjacent_env = exe_dir / ".env"
    project_env = project_root / ".env"
    adjacent_env.write_text("OPENROUTER_API_KEY=sk-or-adjacent\n")
    project_env.write_text("OPENROUTER_API_KEY=sk-or-project\n")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "AI Advisory Board.exe"))
    monkeypatch.setattr(app_paths, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("AAB_DATA_DIR", str(data_root))

    migrated_from = app_paths.migrate_env_file(logger=logger)

    assert migrated_from == adjacent_env
    assert (data_root / ".env").read_text() == "OPENROUTER_API_KEY=sk-or-adjacent\n"
    assert adjacent_env.exists()
    assert any("Migrated .env" in args[0] for args in logger.infos)


def test_env_migration_is_idempotent_when_target_exists(monkeypatch, tmp_path):
    app_paths = import_app_paths()
    logger = CapturingLogger()
    project_root = tmp_path / "project"
    data_root = tmp_path / "user-data"
    project_root.mkdir()
    data_root.mkdir()
    (project_root / ".env").write_text("OPENROUTER_API_KEY=sk-or-source\n")
    (data_root / ".env").write_text("OPENROUTER_API_KEY=sk-or-existing\n")

    clear_frozen(monkeypatch)
    monkeypatch.setattr(app_paths, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("AAB_DATA_DIR", str(data_root))

    migrated_from = app_paths.migrate_env_file(logger=logger)

    assert migrated_from is None
    assert (data_root / ".env").read_text() == "OPENROUTER_API_KEY=sk-or-existing\n"
    assert logger.infos == []


def test_env_migration_skips_empty_or_corrupt_sources(monkeypatch, tmp_path):
    app_paths = import_app_paths()
    logger = CapturingLogger()
    project_root = tmp_path / "project"
    data_root = tmp_path / "user-data"
    project_root.mkdir()
    (project_root / ".env").write_text("OTHER=value\nOPENROUTER_API_KEY=   \n")

    clear_frozen(monkeypatch)
    monkeypatch.setattr(app_paths, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("AAB_DATA_DIR", str(data_root))

    migrated_from = app_paths.migrate_env_file(logger=logger)

    assert migrated_from is None
    assert not (data_root / ".env").exists()
    assert any("Skipping .env migration" in args[0] for args in logger.warnings)


def test_env_migration_fails_loudly_when_target_write_fails(monkeypatch, tmp_path):
    app_paths = import_app_paths()
    project_root = tmp_path / "project"
    data_root = tmp_path / "user-data"
    project_root.mkdir()
    source = project_root / ".env"
    source.write_text("OPENROUTER_API_KEY=sk-or-source\n")

    def fail_write(*args, **kwargs):
        raise OSError("read-only target")

    clear_frozen(monkeypatch)
    monkeypatch.setattr(app_paths, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("AAB_DATA_DIR", str(data_root))
    monkeypatch.setattr(app_paths, "_write_text_atomic", fail_write)

    with pytest.raises(RuntimeError, match="Failed to migrate .env"):
        app_paths.migrate_env_file()

    assert source.exists()
    assert not (data_root / ".env").exists()
