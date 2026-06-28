$OllamaRequiredVersion = "0.30.10"
$OllamaPreferredLibrary = "cuda_v13"
$OllamaFallbackLibrary = "cuda_v12"
$OllamaHost = "127.0.0.1:11434"
$OllamaHostUrl = "http://$OllamaHost"
$OllamaWarmupModel = "llama3.1:8b"
$OllamaInstallerFileName = "OllamaSetup.exe"
$OllamaInstallerUrl = "https://github.com/ollama/ollama/releases/download/v$OllamaRequiredVersion/OllamaSetup.exe"
$OllamaChecksumUrl = "https://github.com/ollama/ollama/releases/download/v$OllamaRequiredVersion/sha256sum.txt"

function Get-OllamaCommandPath {
  $command = Get-Command "ollama" -ErrorAction SilentlyContinue
  if ($command) {
    return $command.Source
  }

  $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
  if (-not [string]::IsNullOrWhiteSpace($localAppData)) {
    $candidate = Join-Path $localAppData "Programs\Ollama\ollama.exe"
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  return $null
}

function Get-OllamaLogPath {
  $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
  if ([string]::IsNullOrWhiteSpace($localAppData)) {
    return "Ollama server log under LOCALAPPDATA"
  }
  return (Join-Path $localAppData "Ollama\server.log")
}

function Invoke-OllamaDownload {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Uri,
    [Parameter(Mandatory = $true)]
    [string]$OutFile
  )

  $previousProgressPreference = $ProgressPreference
  try {
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
  } finally {
    $ProgressPreference = $previousProgressPreference
  }
}

function Get-OllamaExpectedSha256 {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FileName
  )

  $checksumPath = Join-Path ([IO.Path]::GetTempPath()) "ollama-$OllamaRequiredVersion-sha256sum.txt"
  if (-not (Test-Path $checksumPath)) {
    Invoke-OllamaDownload -Uri $OllamaChecksumUrl -OutFile $checksumPath
  }

  $escapedFileName = [regex]::Escape($FileName)
  foreach ($line in Get-Content -Path $checksumPath) {
    if ($line -match "^\s*([A-Fa-f0-9]{64})\s+\*?(?:\./)?$escapedFileName\s*$") {
      return $Matches[1].ToUpperInvariant()
    }
  }

  throw "Checksum entry for $FileName was not found in $OllamaChecksumUrl."
}

function Test-OllamaReleaseAsset {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$FileName
  )

  if (-not (Test-Path $Path)) {
    return $false
  }

  $expectedHash = Get-OllamaExpectedSha256 -FileName $FileName
  $actualHash = (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToUpperInvariant()
  return $actualHash -eq $expectedHash
}

function Get-OllamaPinnedInstallerPath {
  $installerPath = Join-Path ([IO.Path]::GetTempPath()) "OllamaSetup-$OllamaRequiredVersion.exe"
  if (Test-OllamaReleaseAsset -Path $installerPath -FileName $OllamaInstallerFileName) {
    Write-Host "Using verified cached Ollama installer at $installerPath"
    return $installerPath
  }

  if (Test-Path $installerPath) {
    Write-Warning "Removing incomplete or invalid cached Ollama installer at $installerPath"
    Remove-Item -LiteralPath $installerPath -Force
  }

  $downloadPath = "$installerPath.download"
  if (Test-Path $downloadPath) {
    Remove-Item -LiteralPath $downloadPath -Force
  }

  Write-Host "Downloading $OllamaInstallerUrl"
  Invoke-OllamaDownload -Uri $OllamaInstallerUrl -OutFile $downloadPath
  Move-Item -LiteralPath $downloadPath -Destination $installerPath -Force

  if (-not (Test-OllamaReleaseAsset -Path $installerPath -FileName $OllamaInstallerFileName)) {
    throw "Downloaded Ollama installer failed SHA256 validation: $installerPath"
  }

  return $installerPath
}

function Get-OllamaInstalledVersion {
  $ollama = Get-OllamaCommandPath
  if (-not $ollama) {
    return $null
  }

  $output = & $ollama --version 2>&1
  if ($LASTEXITCODE -ne 0) {
    return $null
  }

  $text = ($output | Out-String).Trim()
  if ($text -match "(\d+\.\d+\.\d+)") {
    return $Matches[1]
  }
  return $null
}

function Get-OllamaRegistryProperty {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Entry,
    [Parameter(Mandatory = $true)]
    [string]$Name
  )

  $property = $Entry.PSObject.Properties[$Name]
  if ($property) {
    return $property.Value
  }
  return $null
}

function Get-OllamaUninstallerPath {
  foreach ($registryPath in @(
      "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
      "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
      "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )) {
    $entries = Get-ItemProperty $registryPath -ErrorAction SilentlyContinue |
      Where-Object { (Get-OllamaRegistryProperty -Entry $_ -Name "DisplayName") -like "Ollama*" }

    foreach ($entry in $entries) {
      $installLocation = Get-OllamaRegistryProperty -Entry $entry -Name "InstallLocation"
      if (-not [string]::IsNullOrWhiteSpace($installLocation)) {
        $candidate = Join-Path $installLocation "unins000.exe"
        if (Test-Path $candidate) {
          return $candidate
        }
      }

      $uninstallString = Get-OllamaRegistryProperty -Entry $entry -Name "UninstallString"
      if (-not [string]::IsNullOrWhiteSpace($uninstallString)) {
        if ($uninstallString -match '^"([^"]+)"') {
          return $Matches[1]
        }
        return ($uninstallString -split "\s+", 2)[0]
      }
    }
  }

  $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
  if (-not [string]::IsNullOrWhiteSpace($localAppData)) {
    $candidate = Join-Path $localAppData "Programs\Ollama\unins000.exe"
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  return $null
}

function Stop-OllamaRuntime {
  foreach ($processName in @("ollama", "ollama app", "llama-server", "ollama_llama_server")) {
    Get-Process -Name $processName -ErrorAction SilentlyContinue |
      Stop-Process -Force -ErrorAction SilentlyContinue
  }
  Start-Sleep -Seconds 2
}

function Uninstall-OllamaExistingVersion {
  param(
    [string]$Version
  )

  $uninstaller = Get-OllamaUninstallerPath
  if (-not $uninstaller) {
    Write-Warning "Ollama uninstaller was not found. The pinned installer will run over the current installation."
    return
  }

  Write-Host "Uninstalling existing Ollama $Version before installing pinned version."
  Stop-OllamaRuntime
  $uninstallLog = Join-Path ([IO.Path]::GetTempPath()) "OllamaUninstall-$Version.log"
  $uninstallProcess = Start-Process `
    -FilePath $uninstaller `
    -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/NORESTARTAPPLICATIONS", "/LOG=$uninstallLog") `
    -Wait `
    -PassThru

  if ($uninstallProcess.ExitCode -ne 0) {
    throw "Ollama uninstaller failed with exit code $($uninstallProcess.ExitCode). Log: $uninstallLog"
  }
  Start-Sleep -Seconds 2
}

function Start-OllamaInstallerAndWait {
  param(
    [Parameter(Mandatory = $true)]
    [string]$InstallerPath,
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments
  )

  $process = Start-Process `
    -FilePath $InstallerPath `
    -ArgumentList $Arguments `
    -PassThru
  Wait-Process -Id $process.Id
  $process.Refresh()
  return $process.ExitCode
}

function Install-OllamaPinnedVersion {
  param(
    [switch]$SkipInstall
  )

  $currentVersion = Get-OllamaInstalledVersion
  if ($currentVersion -eq $OllamaRequiredVersion) {
    Write-Host "Ollama $OllamaRequiredVersion already installed."
    return
  }

  if ($SkipInstall) {
    if ($currentVersion) {
      throw "Ollama $currentVersion is installed, but this runtime requires $OllamaRequiredVersion. Rerun without -SkipOllamaInstall to install the pinned version."
    }
    throw "Ollama is missing. Rerun without -SkipOllamaInstall to install $OllamaRequiredVersion."
  }

  if ($currentVersion) {
    Write-Host "Replacing Ollama $currentVersion with pinned version $OllamaRequiredVersion."
  } else {
    Write-Host "Installing pinned Ollama version $OllamaRequiredVersion."
  }

  if ($currentVersion) {
    Uninstall-OllamaExistingVersion -Version $currentVersion
  } else {
    Stop-OllamaRuntime
  }

  $installerPath = Get-OllamaPinnedInstallerPath

  $installLog = Join-Path ([IO.Path]::GetTempPath()) "OllamaInstall-$OllamaRequiredVersion.log"
  Write-Host "Running pinned Ollama installer. Log: $installLog"
  $installExitCode = Start-OllamaInstallerAndWait `
    -InstallerPath $installerPath `
    -Arguments @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/NORESTARTAPPLICATIONS", "/LOG=$installLog")

  if ($installExitCode -ne 0) {
    throw "Ollama installer failed with exit code $installExitCode. Log: $installLog"
  }

  Stop-OllamaRuntime
  Start-Sleep -Seconds 2
  $installedVersion = Get-OllamaInstalledVersion
  if ($installedVersion -ne $OllamaRequiredVersion) {
    throw "Ollama version validation failed. Expected $OllamaRequiredVersion, got '$installedVersion'. Restart PowerShell if PATH was just updated, then rerun."
  }
}

function Set-OllamaGpuEnvironment {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Library
  )

  $env:OLLAMA_LLM_LIBRARY = $Library
  $env:CUDA_VISIBLE_DEVICES = "0"
  $env:OLLAMA_HOST = $OllamaHost
}

function Start-OllamaServe {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Library
  )

  $ollama = Get-OllamaCommandPath
  if (-not $ollama) {
    throw "Ollama executable was not found after installation."
  }

  Set-OllamaGpuEnvironment -Library $Library
  Write-Host "Starting Ollama $OllamaRequiredVersion with OLLAMA_LLM_LIBRARY=$Library and CUDA_VISIBLE_DEVICES=0."
  Start-Process -FilePath $ollama -ArgumentList @("serve") -WindowStyle Hidden | Out-Null
}

function Wait-OllamaHttpReady {
  param(
    [int]$TimeoutSeconds = 60
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  do {
    try {
      Invoke-RestMethod -Method Get -Uri "$OllamaHostUrl/api/tags" -TimeoutSec 5 | Out-Null
      return
    } catch {
      Start-Sleep -Seconds 1
    }
  } while ((Get-Date) -lt $deadline)

  throw "Ollama did not become ready at $OllamaHostUrl within $TimeoutSeconds seconds. Check $(Get-OllamaLogPath)."
}

function Invoke-OllamaWarmup {
  Write-Host "Validating Ollama GPU runtime with $OllamaWarmupModel."
  $payload = @{
    model = $OllamaWarmupModel
    prompt = "responde solo ok"
    stream = $false
    options = @{
      num_predict = 2
    }
  } | ConvertTo-Json -Depth 5

  try {
    Invoke-RestMethod `
      -Method Post `
      -Uri "$OllamaHostUrl/api/generate" `
      -ContentType "application/json" `
      -Body $payload `
      -TimeoutSec 180 | Out-Null
  } catch {
    throw "Ollama warmup API failed for $OllamaWarmupModel. $($_.Exception.Message)"
  }
}

function Get-OllamaObjectProperty {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Object,
    [Parameter(Mandatory = $true)]
    [string]$Name
  )

  $property = $Object.PSObject.Properties[$Name]
  if ($property) {
    return $property.Value
  }
  return $null
}

function Get-OllamaPsText {
  $ollama = Get-OllamaCommandPath
  if (-not $ollama) {
    throw "Ollama executable was not found."
  }

  $output = & $ollama ps 2>&1
  if ($LASTEXITCODE -ne 0) {
    throw "ollama ps failed while validating GPU runtime."
  }
  return ($output | Out-String)
}

function Assert-OllamaGpuRuntime {
  Invoke-OllamaWarmup

  $response = Invoke-RestMethod -Method Get -Uri "$OllamaHostUrl/api/ps" -TimeoutSec 15
  $modelsProperty = $response.PSObject.Properties["models"]
  $models = @()
  if ($modelsProperty) {
    $models = @($modelsProperty.Value)
  }

  $loadedModel = $models |
    Where-Object {
      (Get-OllamaObjectProperty -Object $_ -Name "name") -eq $OllamaWarmupModel -or
      (Get-OllamaObjectProperty -Object $_ -Name "model") -eq $OllamaWarmupModel
    } |
    Select-Object -First 1

  if (-not $loadedModel) {
    throw "Ollama validation failed: $OllamaWarmupModel is not present in /api/ps after warmup."
  }

  $sizeVram = Get-OllamaObjectProperty -Object $loadedModel -Name "size_vram"
  if ($null -eq $sizeVram -or [double]$sizeVram -le 0) {
    throw "Ollama validation failed: /api/ps reports size_vram=$sizeVram for $OllamaWarmupModel."
  }

  $psText = Get-OllamaPsText
  if ($psText -match "100% CPU") {
    throw "Ollama validation failed: ollama ps reports 100% CPU."
  }

  Write-Host "Ollama GPU validation passed for $OllamaWarmupModel with size_vram=$sizeVram."
}

function Ensure-OllamaGpuRuntime {
  param(
    [switch]$SkipInstall
  )

  Install-OllamaPinnedVersion -SkipInstall:$SkipInstall

  $lastError = $null
  $libraries = @($OllamaPreferredLibrary, $OllamaFallbackLibrary) | Select-Object -Unique
  foreach ($library in $libraries) {
    try {
      Stop-OllamaRuntime
      Start-OllamaServe -Library $library
      Wait-OllamaHttpReady
      Assert-OllamaGpuRuntime
      return
    } catch {
      $lastError = $_.Exception.Message
      Write-Warning "Ollama GPU validation failed with OLLAMA_LLM_LIBRARY=$library. $lastError"
      Stop-OllamaRuntime
    }
  }

  throw "Ollama GPU validation failed for $($libraries -join ', '). No CPU fallback was used. Last error: $lastError. Check NVIDIA driver availability and Ollama logs at $(Get-OllamaLogPath)."
}
