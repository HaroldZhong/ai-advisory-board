from pathlib import Path
from typing import Optional


def build(repo_root: Optional[Path] = None) -> int:
    print("Starting AI Advisory Board Build Process...")

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent

    # CANARY (temporary, will be reverted): prove windows-build-smoke bites.
    return 1

    # Ensure frontend is built first
    frontend_dist = root / "frontend" / "dist"
    if not frontend_dist.exists():
        print(f"ERROR: Frontend dist folder not found at {frontend_dist}.")
        print("Please run 'npm run build' in the frontend directory first.")
        return 1

    spec_path = root / "packaging" / "ai-advisory-board.spec"
    if not spec_path.exists():
        print(f"ERROR: PyInstaller spec not found at {spec_path}.")
        return 1

    print(f"Found frontend build at: {frontend_dist}")

    print(f"Running PyInstaller with canonical spec: {spec_path}")

    import PyInstaller.__main__

    args = ["--noconfirm", "--clean", str(spec_path)]
    PyInstaller.__main__.run(args)

    print("\nBuild complete. Check the 'dist' folder for your executable.")
    return 0

if __name__ == "__main__":
    raise SystemExit(build())
