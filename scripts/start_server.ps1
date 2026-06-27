param(
  [string]$HostName = "127.0.0.1",
  [int]$Port = 4173,
  [string]$LmStudio = "http://127.0.0.1:11434",
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

$LocalComparatorConfig = Join-Path $Root "baselines\comparator_config.local.json"
if (Test-Path $LocalComparatorConfig) {
  $env:COMPARATOR_CONFIG_PATH = $LocalComparatorConfig
}

& $VenvPython $Server --host $HostName --port $Port --lm-studio $LmStudio
