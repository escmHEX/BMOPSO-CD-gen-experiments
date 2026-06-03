param(
  [string]$HostName = "127.0.0.1",
  [int]$Port = 4173,
  [string]$LmStudio = "http://127.0.0.1:1234",
  [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path (Join-Path $Root $VenvPath) "Scripts\python.exe"
$Server = Join-Path $Root "server.py"

if (-not (Test-Path $VenvPython)) {
  throw "No se encontro $VenvPython. Ejecuta scripts\setup_backend_env.ps1 primero."
}

if (-not (Test-Path $Server)) {
  throw "No se encontro $Server"
}

& $VenvPython $Server --host $HostName --port $Port --lm-studio $LmStudio
