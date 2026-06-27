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

if ($InternalRun) {
  Require-Runtime
  New-Item -ItemType Directory -Force -Path $DaemonDir | Out-Null
  Set-Location $Root
  $env:COMPARATOR_CONFIG_PATH = $Config
  & $VenvPython $Server --host $HostName --port $Port --lm-studio $LmStudio
  exit $LASTEXITCODE
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
    return
  }
  Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
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
