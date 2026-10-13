$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$runtime = 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path $runtime) { & $runtime app.py } else { python app.py }
