param(
    [switch]$LiveLlm
)

$arguments = @("-m", "scripts.run_public_samples")
if ($LiveLlm) {
    $arguments += "--live-llm"
}

& "$PSScriptRoot\.venv\Scripts\python.exe" @arguments
exit $LASTEXITCODE
