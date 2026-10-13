$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$runtime = 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path $runtime)) { $runtime = 'python' }
& $runtime -m pip install -r requirements.txt --target vendor
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task' -OutFile models/pose_landmarker_lite.task
& $runtime build_model.py
