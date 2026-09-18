$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectPath

$venvPython = Join-Path $projectPath ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    python -m venv .venv
}

& $venvPython -m pip install -r requirements.txt
& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
