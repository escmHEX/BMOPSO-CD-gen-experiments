param(
  [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvFullPath = Join-Path $Root $VenvPath
$VenvPython = Join-Path $VenvFullPath "Scripts\python.exe"
$Requirements = Join-Path $Root "requirements.backend.txt"

if (-not (Test-Path $Requirements)) {
  throw "No se encontro $Requirements"
}

if (-not (Test-Path $VenvPython)) {
  Write-Host "Creando entorno backend en $VenvFullPath"
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.13 -m venv $VenvFullPath
  } else {
    & python -m venv $VenvFullPath
  }
}

Write-Host "Actualizando pip"
& $VenvPython -m pip install --upgrade pip

Write-Host "Instalando dependencias del backend desde $Requirements"
& $VenvPython -m pip install -r $Requirements

Write-Host "Verificando sentence_transformers"
& $VenvPython -c "import sentence_transformers; print(sentence_transformers.__version__)"

Write-Host "Entorno backend listo: $VenvPython"
