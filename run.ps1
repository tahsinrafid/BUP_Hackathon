$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectPath

$venvPython = Join-Path $projectPath ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    python -m venv .venv
    & $venvPython -m pip install -r requirements-dev.txt
}

$port = if ($env:PORT) { $env:PORT } else { "8000" }
& $venvPython -m uvicorn app.main:app --host 0.0.0.0 --port $port --reload
