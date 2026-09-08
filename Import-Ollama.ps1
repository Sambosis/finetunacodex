param([Parameter(Mandatory=$true)][string]$ArtifactDirectory, [string]$ModelName = 'qwen35-web-tools:4b')
$ErrorActionPreference = 'Stop'

function Get-Sha256Hash([string]$FilePath) {
    if (Get-Command Get-FileHash -ErrorAction SilentlyContinue) {
        return (Get-FileHash -LiteralPath $FilePath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $stream = [System.IO.File]::OpenRead($FilePath)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $hashBytes = $sha.ComputeHash($stream)
        return ([System.BitConverter]::ToString($hashBytes)).Replace("-", "").ToLowerInvariant()
    } finally {
        $stream.Close()
    }
}

$taskArtifact = (Resolve-Path -LiteralPath $ArtifactDirectory).Path
$taskManifest = Get-Content -LiteralPath (Join-Path $taskArtifact 'SHA256SUMS.json') -Raw | ConvertFrom-Json
$taskModelFile = Join-Path $taskArtifact 'qwen35-4b-web-tools-Q4_K_M.gguf'
$taskExpectedHash = $taskManifest.'qwen35-4b-web-tools-Q4_K_M.gguf'

Write-Host "Verifying GGUF SHA256 hash..."
$actualHash = Get-Sha256Hash $taskModelFile
if (-not $taskExpectedHash -or $actualHash -ne $taskExpectedHash) {
    throw "GGUF checksum mismatch. Expected: $taskExpectedHash, Actual: $actualHash"
}
Write-Host "Checksum verified: $actualHash"

Write-Host "Importing language GGUF and vision projector together..."
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'import_ollama.py') --artifacts $taskArtifact --model $ModelName
if ($LASTEXITCODE -ne 0) { throw 'Vision-preserving Ollama import failed' }

Write-Host "Running evaluation probes on $ModelName..."
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'evaluate.py') --model $ModelName --data (Join-Path $PSScriptRoot 'data\website-test.jsonl') --output (Join-Path $taskArtifact 'ollama-evaluation.json')
if ($LASTEXITCODE -ne 0) { throw 'Ollama evaluation failed' }
Write-Host "Ollama evaluation complete! Results saved to $(Join-Path $taskArtifact 'ollama-evaluation.json')"
