param([switch]$Preflight)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment' }
}
& $taskPython -m pip install -r requirements-local.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if (-not (Test-Path data\train.jsonl)) {
    & $taskPython prepare_data.py
    if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed' }
}
& $taskPython make_website_tests.py
if ($LASTEXITCODE -ne 0) { throw 'Website test preparation failed' }
if ($Preflight) { & $taskPython launch.py } else { & $taskPython launch.py --run }
if ($LASTEXITCODE -ne 0) { throw 'Training launcher did not complete successfully' }
