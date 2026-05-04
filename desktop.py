import sys
import os
import threading
import time
import logging
import socket
import traceback
import uvicorn
import webview
from backend.app_paths import get_data_root, get_desktop_log_path

# --- Determine paths ---
def get_base_path():
    """Get the absolute path to the resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

def get_app_dir():
    """Get the writable app data directory."""
    return str(get_data_root())

# --- Logging to file (critical for --noconsole builds) ---
APP_DIR = get_app_dir()
LOG_FILE = str(get_desktop_log_path())
logger = logging.getLogger("desktop")


def configure_logging() -> logging.Logger:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("desktop")

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8001
APP_ROUTE = "/app"

# Track server startup state
server_error = None
server_ready = threading.Event()

LOADING_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0; display: flex; align-items: center; justify-content: center;
        height: 100vh; background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif;
    }
    .loader {
        text-align: center;
    }
    .spinner {
        width: 48px; height: 48px; border: 4px solid #334155; border-top-color: #3b82f6;
        border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 24px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    h2 { font-weight: 500; margin: 0 0 8px; }
    p { color: #94a3b8; font-size: 14px; margin: 0; }
</style>
</head>
<body>
    <div class="loader">
        <div class="spinner"></div>
        <h2>AI Advisory Board</h2>
        <p>Starting server...</p>
    </div>
</body>
</html>
"""

ERROR_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
    body {{
        margin: 0; display: flex; align-items: center; justify-content: center;
        height: 100vh; background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif;
    }}
    .error {{ text-align: center; max-width: 600px; padding: 32px; }}
    h2 {{ color: #ef4444; margin: 0 0 16px; }}
    pre {{ background: #1e293b; padding: 16px; border-radius: 8px; text-align: left;
           font-size: 12px; overflow-x: auto; white-space: pre-wrap; color: #fca5a5; }}
    p {{ color: #94a3b8; font-size: 14px; }}
</style>
</head>
<body>
    <div class="error">
        <h2>Server Failed to Start</h2>
        <p>Check logs/desktop.log in the app data folder for details.</p>
        <pre>{error}</pre>
    </div>
</body>
</html>
"""

def start_server():
    """Run the FastAPI server with error capture."""
    global server_error
    try:
        base_path = get_base_path()
        sys.path.insert(0, base_path)
        logger.info("Base path: %s", base_path)
        logger.info("App dir: %s", APP_DIR)
        logger.info("Importing backend.main...")
        
        from backend.main import app
        logger.info("Backend imported successfully. Starting uvicorn...")
        
        server_ready.set()  # Signal that import succeeded
        uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
    except Exception as e:
        server_error = traceback.format_exc()
        logger.error("Server failed to start:\n%s", server_error)
        server_ready.set()  # Unblock the main thread so it can show the error

def wait_for_port(host, port, timeout=30):
    """Poll until the server is accepting connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError, socket.timeout):
            time.sleep(0.3)
    return False

def get_app_url(host=SERVER_HOST, port=SERVER_PORT):
    """Return the desktop startup URL for the actual app surface."""
    return f"http://{host}:{port}{APP_ROUTE}"

def main():
    global logger, server_error
    logger = configure_logging()
    logger.info("=== AI Advisory Board Desktop Starting ===")
    logger.info("Python: %s", sys.version)
    logger.info("Frozen: %s", getattr(sys, 'frozen', False))
    
    # Start the web server in a background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Create the webview window immediately with a loading screen
    logger.info("Creating webview window with loading screen...")
    window = webview.create_window(
        title="AI Advisory Board",
        html=LOADING_HTML,
        width=1200,
        height=800,
        resizable=True,
        min_size=(960, 600)
    )
    
    def on_loaded():
        """Called after the webview window is shown — wait for server, then navigate."""
        logger.info("Webview loaded. Waiting for server...")
        
        # Wait for the server thread to at least attempt the import
        server_ready.wait(timeout=30)
        
        if server_error:
            logger.error("Server had an error, showing error page.")
            safe_error = server_error.replace("\\", "\\\\").replace("`", "'")
            window.load_html(ERROR_HTML.format(error=safe_error))
            return
        
        # Now wait for the port to be accepting connections
        logger.info("Import succeeded, waiting for port %d...", SERVER_PORT)
        if wait_for_port(SERVER_HOST, SERVER_PORT, timeout=30):
            logger.info("Server is ready! Navigating to app...")
            window.load_url(get_app_url())
        else:
            logger.error("Server port never opened.")
            window.load_html(ERROR_HTML.format(error="Server started but port never opened. Check logs/desktop.log in the app data folder."))
    
    # Run on_loaded in a thread so it doesn't block webview.start()
    threading.Thread(target=on_loaded, daemon=True).start()
    
    # This blocks until the window is closed
    webview.start(private_mode=False)
    
    logger.info("Application closed.")

if __name__ == "__main__":
    main()
