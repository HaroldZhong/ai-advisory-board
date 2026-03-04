import PyInstaller.__main__
import platform
import os
import shutil
import sys

def build():
    print("Starting AI Advisory Board Build Process...")
    
    # Ensure frontend is built first
    frontend_dist = os.path.join("frontend", "dist")
    if not os.path.exists(frontend_dist):
        print(f"ERROR: Frontend dist folder not found at {frontend_dist}.")
        print("Please run 'npm run build' in the frontend directory first.")
        sys.exit(1)
        
    print(f"Found frontend build at: {frontend_dist}")
    
    # Detect OS
    is_windows = platform.system() == "Windows"
    separator = ";" if is_windows else ":"
    
    # Data paths to include
    backend_data = f"backend{separator}backend"
    frontend_data = f"{frontend_dist}{separator}frontend/dist"
    
    print("Running PyInstaller...")
    
    # PyInstaller arguments
    args = [
        "desktop.py",
        "--name=AI Advisory Board",
        "--noconsole",
        "--onefile",
        "--clean",
        "--windowed",
        f"--icon=app_icon.ico",
        f"--add-data={backend_data}",
        f"--add-data={frontend_data}",
        # Collect entire packages (includes DLLs, data files, sub-modules)
        "--collect-all=webview",
        "--collect-all=litellm",
        "--collect-all=tiktoken",
        "--collect-all=pypdf",
        "--collect-all=docx",
        "--collect-all=pptx",
        "--collect-all=openpyxl",
        "--collect-all=bs4",
        # Hidden imports for packages without data files
        "--hidden-import=uvicorn",
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.loops",
        "--hidden-import=uvicorn.loops.auto",
        "--hidden-import=uvicorn.protocols",
        "--hidden-import=uvicorn.protocols.http",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.protocols.websockets",
        "--hidden-import=uvicorn.protocols.websockets.auto",
        "--hidden-import=uvicorn.lifespan",
        "--hidden-import=uvicorn.lifespan.on",
        "--hidden-import=fastapi",
        "--hidden-import=backend",
        "--hidden-import=backend.main",
        "--hidden-import=backend.config",
        "--hidden-import=backend.council",
        "--hidden-import=backend.openrouter",
        "--hidden-import=backend.rag",
        "--hidden-import=backend.storage",
        "--hidden-import=backend.analytics",
        "--hidden-import=backend.file_processing",
        "--hidden-import=backend.attachment_storage",
        "--hidden-import=backend.web_search",
        "--hidden-import=backend.rag_utils",
        "--hidden-import=backend.logger",
        "--hidden-import=backend.tools",
        "--hidden-import=backend.tools.router",
        "--hidden-import=backend.tools.registry",
        "--hidden-import=backend.tools.parser",
        "--hidden-import=backend.tools.types",
        "--hidden-import=backend.pageindex",
        "--hidden-import=sqlite3",
        "--hidden-import=httpx",
        "--hidden-import=bottle",
    ]
    
    PyInstaller.__main__.run(args)
    
    print("\nBuild complete. Check the 'dist' folder for your executable.")

if __name__ == "__main__":
    build()
