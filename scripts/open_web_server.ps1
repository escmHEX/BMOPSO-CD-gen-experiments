param(
  [string]$HostName = "127.0.0.1",
  [int]$Port = 4173,
  [string]$LmStudio = "http://127.0.0.1:11434",
  [string]$VenvPath = ".venv",
  [switch]$NoBrowser,
  [switch]$NoMessageBox
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$StartServerScript = Join-Path $PSScriptRoot "start_server.ps1"
$BaseUrl = "http://${HostName}:$Port"
$AppUrl = "$BaseUrl/LLM/"
$HealthUrl = $AppUrl
$LauncherLogDir = Join-Path $Root "runs\server-launcher"

function Test-WebServerReady {
  try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -TimeoutSec 8
    return $response.StatusCode -eq 200
  } catch {
    return $false
  }
}

function Show-LauncherError([string]$Message) {
  if ($NoMessageBox) {
    Write-Host $Message
    return
  }
  try {
    Add-Type -AssemblyName PresentationFramework -ErrorAction Stop
    [System.Windows.MessageBox]::Show($Message, "Comparador de propuestas") | Out-Null
  } catch {
    Write-Host $Message
  }
}

if (-not (Test-Path $StartServerScript)) {
  Show-LauncherError "No se encontro el script de servidor: $StartServerScript"
  exit 1
}

if (-not (Test-WebServerReady)) {
  New-Item -ItemType Directory -Path $LauncherLogDir -Force | Out-Null
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $stdoutPath = Join-Path $LauncherLogDir "$stamp.out.log"
  $stderrPath = Join-Path $LauncherLogDir "$stamp.err.log"

  Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy", "Bypass",
      "-File", $StartServerScript,
      "-HostName", $HostName,
      "-Port", $Port,
      "-LmStudio", $LmStudio,
      "-VenvPath", $VenvPath
    ) `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden

  $deadline = (Get-Date).AddSeconds(45)
  while ((Get-Date) -lt $deadline) {
    if (Test-WebServerReady) {
      break
    }
    Start-Sleep -Milliseconds 750
  }
}

if (Test-WebServerReady) {
  if (-not $NoBrowser) {
    Start-Process $AppUrl
  }
  exit 0
}

Show-LauncherError "No se pudo iniciar el servidor web en $BaseUrl. Revisa que el entorno backend exista y que el puerto $Port no este ocupado por otro proceso."
exit 1
