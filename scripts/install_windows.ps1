param(
  [switch]$ForcePpdb,
  [switch]$SkipPpdb,
  [switch]$SkipOllamaInstall,
  [switch]$SkipModelPull,
  [switch]$SkipWarmup
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$SpacyModelWheel = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

function Fail([string]$Message) {
  throw $Message
}

function Ensure-Command([string]$Name, [string]$InstallHint) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    Fail "Missing command '$Name'. $InstallHint"
  }
}

function Ensure-Uv {
  if (Get-Command uv -ErrorAction SilentlyContinue) {
    return
  }
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
  $uvPath = Join-Path $env:USERPROFILE ".local\bin"
  if ($env:Path -notlike "*$uvPath*") {
    $env:Path = "$uvPath;$env:Path"
  }
  if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail "uv installation finished but uv is not in PATH. Add $uvPath to PATH and rerun."
  }
}

function Ensure-Ollama {
  if (Get-Command ollama -ErrorAction SilentlyContinue) {
    return
  }
  if ($SkipOllamaInstall) {
    Fail "Ollama is missing and -SkipOllamaInstall was passed."
  }
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Fail "Ollama is missing and winget is unavailable. Install Ollama from https://ollama.com/download and rerun."
  }
  & winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
  if ($LASTEXITCODE -ne 0) {
    Fail "winget could not install Ollama."
  }
  if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Fail "Ollama installation finished but ollama is not in PATH. Restart PowerShell or add Ollama to PATH and rerun."
  }
}

function Ensure-OllamaReady {
  Ensure-Ollama
  & ollama list *> $null
  if ($LASTEXITCODE -eq 0) {
    return
  }
  Start-Process -FilePath "ollama" -ArgumentList @("serve") -WindowStyle Hidden
  Start-Sleep -Seconds 5
  & ollama list *> $null
  if ($LASTEXITCODE -ne 0) {
    Fail "Ollama is installed but not responding. Start 'ollama serve' and rerun."
  }
}

function Invoke-Checked([string]$FilePath, [string[]]$Arguments) {
  Write-Host ("+ " + $FilePath + " " + ($Arguments -join " "))
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    Fail "Command failed: $FilePath $($Arguments -join ' ')"
  }
}

function Python-InVenv([string]$VenvPath) {
  return (Join-Path $VenvPath "Scripts\python.exe")
}

function Ensure-Venv([string]$VenvPath) {
  $python = Python-InVenv $VenvPath
  if ($env:UV_VENV_CLEAR -eq "1" -or $env:UV_VENV_CLEAR -eq "true") {
    Invoke-Checked "uv" @("venv", "--clear", "--python", "3.13", $VenvPath)
  } elseif (Test-Path $python) {
    Write-Host "Reusing existing virtual environment at: $VenvPath"
  } elseif (Test-Path $VenvPath) {
    Fail "$VenvPath exists but $python is missing. Remove that directory or rerun with UV_VENV_CLEAR=1."
  } else {
    Invoke-Checked "uv" @("venv", "--python", "3.13", $VenvPath)
  }
  Invoke-Checked "uv" @("pip", "install", "--python", $python, "--upgrade", "pip")
}

function Install-Requirements([string]$Python, [string]$Requirements) {
  if (-not (Test-Path $Requirements)) {
    Fail "Requirements file not found: $Requirements"
  }
  Invoke-Checked "uv" @("pip", "install", "--python", $Python, "-r", $Requirements)
}

function Install-SpacyModel([string]$Python) {
  Invoke-Checked "uv" @("pip", "install", "--python", $Python, $SpacyModelWheel)
}

function Sync-Submodules {
  Ensure-Command "git" "Install Git for Windows and rerun."
  Invoke-Checked "git" @("submodule", "sync", "--recursive")
  Invoke-Checked "git" @("submodule", "update", "--init", "--recursive")
}

function Sync-BinaryRepository {
  $binaryDir = Join-Path $Root "baselines\external\binary-mopso-cd"
  if (Test-Path (Join-Path $binaryDir ".git")) {
    Invoke-Checked "git" @("-C", $binaryDir, "fetch", "origin")
    Invoke-Checked "git" @("-C", $binaryDir, "checkout", "dev")
    Invoke-Checked "git" @("-C", $binaryDir, "pull", "--ff-only", "origin", "dev")
    return
  }
  if (Test-Path $binaryDir) {
    Fail "$binaryDir exists but is not a git repository. Move it or remove it before rerunning."
  }
  Invoke-Checked "git" @("clone", "--branch", "dev", "https://github.com/escmHEX/BMOPSO-CD.git", $binaryDir)
}

function Has-KaggleCredentials {
  if (-not [string]::IsNullOrWhiteSpace($env:KAGGLE_API_TOKEN)) {
    return $true
  }
  $accessToken = Join-Path $env:USERPROFILE ".kaggle\access_token"
  if (Test-Path $accessToken) {
    return $true
  }
  $kaggleJson = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
  if (Test-Path $kaggleJson) {
    return $true
  }
  return -not [string]::IsNullOrWhiteSpace($env:KAGGLE_USERNAME) -and -not [string]::IsNullOrWhiteSpace($env:KAGGLE_KEY)
}

function Prepare-Ppdb([string]$PortalPython) {
  if ($SkipPpdb) {
    Write-Host "Skipping PPDB download and index generation."
    return
  }
  $ppdbDir = Join-Path $Root "data\external\ppdb"
  $ppdbSource = Join-Path $ppdbDir "ppdb-2.0-s-all"
  $ppdbIndex = Join-Path $Root "data\turbulence\ppdb_index.json"
  if ($ForcePpdb -or -not (Test-Path $ppdbSource)) {
    if (-not (Has-KaggleCredentials)) {
      Fail "Kaggle credentials are missing. Configure KAGGLE_API_TOKEN, %USERPROFILE%\.kaggle\access_token, %USERPROFILE%\.kaggle\kaggle.json, or KAGGLE_USERNAME/KAGGLE_KEY for cudawarrior/ppdb-2-0-s-all."
    }
    New-Item -ItemType Directory -Force -Path $ppdbDir | Out-Null
    Invoke-Checked $PortalPython @("-m", "kaggle", "datasets", "download", "-d", "cudawarrior/ppdb-2-0-s-all", "-p", $ppdbDir, "--unzip")
    if (-not (Test-Path $ppdbSource)) {
      $candidate = Get-ChildItem -Path $ppdbDir -Recurse -File |
        Where-Object { $_.Name -eq "ppdb-2.0-s-all" -or $_.Name -eq "ppdb-2.0-s-all.gz" } |
        Select-Object -First 1
      if (-not $candidate) {
        Fail "Kaggle download completed but ppdb-2.0-s-all was not found under $ppdbDir."
      }
      if ($candidate.Name.EndsWith(".gz")) {
        Invoke-Checked $PortalPython @("-c", "import gzip, shutil, sys; shutil.copyfileobj(gzip.open(sys.argv[1], 'rb'), open(sys.argv[2], 'wb'))", $candidate.FullName, $ppdbSource)
      } else {
        Copy-Item -LiteralPath $candidate.FullName -Destination $ppdbSource -Force
      }
    }
  }
  if ($ForcePpdb -or -not (Test-Path $ppdbIndex)) {
    Invoke-Checked $PortalPython @("scripts\prepare_ppdb_index.py", "--source", $ppdbSource, "--output", $ppdbIndex)
  }
}

Ensure-Command "git" "Install Git for Windows and rerun."
Ensure-Uv
Invoke-Checked "uv" @("python", "install", "3.13")
Sync-Submodules
Sync-BinaryRepository

Ensure-Venv ".venv"
$portalPython = Python-InVenv ".venv"
Install-Requirements $portalPython "requirements.backend.txt"
Invoke-Checked "uv" @("pip", "install", "--python", $portalPython, "kaggle")

foreach ($proposal in @("evolmd", "evolmd-mo", "mesap")) {
  Ensure-Venv "baselines\venvs\$proposal"
  $proposalPython = Python-InVenv "baselines\venvs\$proposal"
  Install-Requirements $proposalPython "baselines\external\$proposal\requirements.txt"
  if ($proposal -eq "evolmd-mo") {
    Install-SpacyModel $proposalPython
  }
}

Ensure-Venv "baselines\venvs\binary-mopso-cd"
$binaryPython = Python-InVenv "baselines\venvs\binary-mopso-cd"
Invoke-Checked "uv" @("pip", "install", "--python", $binaryPython, "-e", "baselines\external\binary-mopso-cd[models]")
Install-SpacyModel $binaryPython

Invoke-Checked $portalPython @("scripts\setup_runtime.py", "--write-comparator-config", "--platform", "win32")
Prepare-Ppdb $portalPython

Ensure-OllamaReady
if (-not $SkipWarmup) {
  $warmupArgs = @("scripts\warmup_runtime.py")
  if ($SkipModelPull) {
    $warmupArgs += "--skip-ollama"
  }
  Invoke-Checked $portalPython $warmupArgs
}

Write-Host "Setup complete."
Write-Host "Use:"
Write-Host "  .\scripts\start_server_daemon_windows.ps1 -Action Start -HostName 0.0.0.0"
Write-Host "  .\.venv\Scripts\python.exe scripts\verify_install.py --base-url http://127.0.0.1:4173"
