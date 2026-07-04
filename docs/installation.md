# Installation & Packaging

This guide covers installed desktop usage and local packaging. Development setup stays in the main README.

## Desktop App

1. Download `AI Advisory Board.exe` from the project releases page.
2. Run the executable.
3. Complete the first-run setup flow:
   - **Connect**: paste your OpenRouter API key.
   - **Privacy**: choose the default Zero Data Retention mode for new conversations.
   - **Budget**: choose a starting session budget.

Do not create a `.env` next to the executable for new installs. The app writes the key to the per-user app data directory after first-run setup.

Legacy adjacent-to-exe `.env` files are still migrated non-destructively if present, but that path is only for older installs.

## Windows Notes

### SmartScreen

The current executable is unsigned. Windows may show a SmartScreen warning on first launch.

If you trust the release you downloaded, choose:

```text
More info -> Run anyway
```

Code signing is a distribution polish item and is not required for local use.

### WebView2 Runtime

The desktop window uses Microsoft Edge WebView2 through `pywebview`.

Most Windows 11 machines already include WebView2. Some Windows 10 machines may need the Evergreen Runtime from Microsoft:

```text
https://developer.microsoft.com/microsoft-edge/webview2/
```

If the app opens a blank window or fails before showing the setup screen, install WebView2 and relaunch.

## User Data Location

Packaged desktop builds use `platformdirs.user_data_dir("AI Advisory Board", "HaroldZhong")`.

On Windows this resolves to:

```text
%LOCALAPPDATA%\HaroldZhong\AI Advisory Board\
```

The folder contains:

| Path | Purpose |
|---|---|
| `.env` | Saved OpenRouter API key |
| `data/conversations/` | Conversation JSON files |
| `data/conversations/attachments/` | Uploaded attachment files and metadata |
| `data/pageindex_memory.json` | PageIndex memory index |
| `logs/app.log` | Backend log |
| `logs/desktop.log` | Desktop launcher log |

Development checkouts keep the same structure under the project root. Set `AAB_DATA_DIR` to override the location for tests, alternate drives, or portable wrappers.

## Uninstall And Data Cleanup

Deleting the executable removes the app binary only.

To remove saved API keys, conversations, attachments, memory, and logs, delete:

```text
%LOCALAPPDATA%\HaroldZhong\AI Advisory Board\
```

Keep this folder if you want conversations and settings to survive an app upgrade.

## Build The Executable Locally

From a clean checkout:

```powershell
cd frontend
npm ci
npm run build
cd ..

uv run --group packaging python build_exe.py
```

The build output is:

```text
dist\AI Advisory Board.exe
```

The canonical PyInstaller recipe is `packaging/ai-advisory-board.spec`; `build_exe.py` is only a thin wrapper that checks for `frontend/dist` and invokes that spec.

## Packaging Acceptance Smoke

Before publishing a release candidate:

1. Run the executable on a Windows machine without relying on a source checkout.
2. Complete first-run setup with a real OpenRouter key.
3. Send one message with a preset council.
4. Quit and relaunch; verify the key, conversations, and logs persist.
5. Move the executable to another folder and relaunch; verify data still loads from the app data directory.
6. Run as a standard user, not administrator.

The clean-VM acceptance test should use a Windows environment with no Python installed.

## Network access, proxies, and restricted regions

The app requires HTTPS access to `openrouter.ai`. Nothing is downloaded from
Hugging Face or any other model host — if the app cannot work, it is the
OpenRouter connection (or your API key), not a model download.

- **Proxy:** the backend honors the `HTTPS_PROXY` / `HTTP_PROXY` environment
  variables. It does **not** read the Windows system proxy settings — a VPN
  that only works in your browser (system-proxy mode) will not cover this app.
  Either run your VPN in TUN/global mode, or set `HTTPS_PROXY` before launching.
- **Relay:** set `OPENROUTER_BASE_URL` to route all API calls through an
  OpenAI-compatible relay (default: `https://openrouter.ai/api/v1`).
- **Diagnose:** `GET http://127.0.0.1:8001/api/config/connectivity` reports
  whether the backend can reach OpenRouter and whether your API key is valid
  (network / key). Credit problems are reported at chat time, not by this probe.
- **Logs:** check `logs/app.log` under the app data folder
  (`%LOCALAPPDATA%\HaroldZhong\AI Advisory Board\` on Windows). On the old
  v1.0.0 build, `desktop.log` is next to the `.exe` instead.
