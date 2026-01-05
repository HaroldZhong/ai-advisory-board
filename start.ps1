$ErrorActionPreference = "Stop"

Write-Host "Starting AI Advisory Board..."
Write-Host ""

# Start backend
Write-Host "Starting backend on http://localhost:8001..."
$backendProcess = Start-Process -FilePath "uv" -ArgumentList "run python -m backend.main" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

# Start frontend
Write-Host "Starting frontend on http://localhost:5173..."
Push-Location frontend
$frontendProcess = Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -PassThru -NoNewWindow
Pop-Location

Write-Host ""
Write-Host "✓ AI Advisory Board is running!"
Write-Host "  Backend:  http://localhost:8001"
Write-Host "  Frontend: http://localhost:5173"
Write-Host ""
Write-Host "Press Ctrl+C to stop both servers"

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "Stopping servers..."
    Stop-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $frontendProcess.Id -ErrorAction SilentlyContinue
}
