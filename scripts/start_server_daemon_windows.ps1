param(
  [ValidateSet("Start", "Stop", "Restart", "Status")]
  [string]$Action = "Start",
  [string]$HostName = "127.0.0.1",
  [int]$Port = 4173,
  [string]$LmStudio = "http://127.0.0.1:11434",
  [string]$VenvPath = ".venv",
  [string]$TaskName = "BMOPSO-CD Experiments Portal",
  [switch]$InternalRun
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Server = Join-Path $Root "server.py"
$VenvPython = Join-Path (Join-Path $Root $VenvPath) "Scripts\python.exe"
$Config = Join-Path $Root "baselines\comparator_config.local.json"
$DaemonDir = Join-Path $Root "runs\server-daemon"
$LogFile = Join-Path $DaemonDir "windows-server.log"
$BackendStdoutLog = Join-Path $DaemonDir "windows-backend.out.log"
$BackendStderrLog = Join-Path $DaemonDir "windows-backend.err.log"
$OllamaRuntimeScript = Join-Path $PSScriptRoot "ollama_runtime.ps1"
. $OllamaRuntimeScript

function Fail([string]$Message) {
  throw $Message
}

function Escape-SingleQuoted([string]$Value) {
  return $Value.Replace("'", "''")
}

function Require-Runtime {
  if (-not (Test-Path $VenvPython)) {
    Fail "Missing $VenvPython. Run .\scripts\install_windows.ps1 first."
  }
  if (-not (Test-Path $Server)) {
    Fail "Missing $Server."
  }
  if (-not (Test-Path $Config)) {
    Fail "Missing $Config. Run .\scripts\install_windows.ps1 first."
  }
}

function Stop-PortalServerProcesses {
  $serverPath = [IO.Path]::GetFullPath($Server)
  $processes = Get-CimInstance Win32_Process |
    Where-Object {
      -not [string]::IsNullOrWhiteSpace($_.CommandLine) -and
      $_.CommandLine -like "*$serverPath*"
    }

  foreach ($process in $processes) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
  }

  if ($processes) {
    Start-Sleep -Seconds 2
  }
}

function Wait-PortalBackendHealth {
  param(
    [int]$TimeoutSeconds = 30
  )

  $healthUrl = "http://${HostName}:$Port/api/portal/health"
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  do {
    try {
      Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5 | Out-Null
      return
    } catch {
      Start-Sleep -Seconds 1
    }
  } while ((Get-Date) -lt $deadline)

  Fail "Portal backend did not become healthy at $healthUrl within $TimeoutSeconds seconds. See $BackendStderrLog"
}

function Start-PortalBackend {
  $arguments = @(
    $Server,
    "--host",
    $HostName,
    "--port",
    [string]$Port,
    "--lm-studio",
    $LmStudio,
    "--skip-port-release"
  )

  $process = Start-Process `
    -FilePath $VenvPython `
    -ArgumentList $arguments `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $BackendStdoutLog `
    -RedirectStandardError $BackendStderrLog `
    -PassThru

  Wait-PortalBackendHealth
  Write-Host "Portal backend started with PID $($process.Id)."
}

if ($InternalRun) {
  Require-Runtime
  New-Item -ItemType Directory -Force -Path $DaemonDir | Out-Null
  Set-Location $Root
  $env:COMPARATOR_CONFIG_PATH = $Config
  Ensure-OllamaGpuRuntime
  Write-Host "Starting portal backend at http://${HostName}:$Port"
  Start-PortalBackend
  exit 0
}

function Register-PortalTask {
  Require-Runtime
  New-Item -ItemType Directory -Force -Path $DaemonDir | Out-Null
  $scriptPath = $PSCommandPath
  $escapedScript = Escape-SingleQuoted $scriptPath
  $escapedHost = Escape-SingleQuoted $HostName
  $escapedLmStudio = Escape-SingleQuoted $LmStudio
  $escapedVenv = Escape-SingleQuoted $VenvPath
  $escapedLog = Escape-SingleQuoted $LogFile
  $command = "& '$escapedScript' -InternalRun -HostName '$escapedHost' -Port $Port -LmStudio '$escapedLmStudio' -VenvPath '$escapedVenv' *>> '$escapedLog'"
  $argument = "-NoProfile -ExecutionPolicy Bypass -Command `$ErrorActionPreference='Stop'; $command"
  $taskAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument
  $settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 30) `
    -MultipleInstances IgnoreNew
  Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Settings $settings -Description "BMOPSO-CD experiments portal daemon" -Force | Out-Null
}

function Start-PortalTask {
  Register-PortalTask
  Start-ScheduledTask -TaskName $TaskName
  Start-Sleep -Seconds 2
  Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
  Write-Host "Logs: $LogFile"
}

function Stop-PortalTask {
  $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $task) {
    Write-Host "Task not registered: $TaskName"
    Stop-PortalServerProcesses
    return
  }
  Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  Stop-PortalServerProcesses
  Write-Host "Task stopped: $TaskName"
}

function Show-PortalTaskStatus {
  $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $task) {
    Write-Host "Task not registered: $TaskName"
    exit 3
  }
  $info = Get-ScheduledTaskInfo -TaskName $TaskName
  $task | Select-Object TaskName, State
  $info | Select-Object LastRunTime, LastTaskResult, NextRunTime
  Write-Host "Logs: $LogFile"
}

switch ($Action) {
  "Start" { Start-PortalTask }
  "Stop" { Stop-PortalTask }
  "Restart" {
    Stop-PortalTask
    Start-PortalTask
  }
  "Status" { Show-PortalTaskStatus }
}
