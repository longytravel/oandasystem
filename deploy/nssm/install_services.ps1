# Install trading strategies as Windows services using NSSM.
# Run as Administrator.
#
# Prerequisites:
#   1. Download NSSM from https://nssm.cc/download and add to PATH
#   2. Python installed and in PATH
#   3. pip install -r deploy/requirements-live.txt
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deploy\nssm\install_services.ps1

param(
    [string]$BaseDir = "C:\Trading\oandasystem",
    [string]$InstancesDir = "C:\Trading\instances",
    [string]$PythonPath = "python"
)

$strategiesFile = Join-Path $BaseDir "deploy\strategies.json"
if (-not (Test-Path $strategiesFile)) {
    Write-Error "strategies.json not found at $strategiesFile"
    exit 1
}

$config = Get-Content $strategiesFile | ConvertFrom-Json

foreach ($strategy in $config.strategies) {
    $id = $strategy.id
    $serviceName = "Trading_$id"
    $instanceDir = Join-Path $InstancesDir $id

    if (-not $strategy.enabled) {
        Write-Host "Skipping disabled strategy: $id"
        continue
    }

    # Create instance directories
    New-Item -ItemType Directory -Force -Path (Join-Path $instanceDir "state") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $instanceDir "logs") | Out-Null

    # Check if config.json exists
    $configFile = Join-Path $instanceDir "config.json"
    if (-not (Test-Path $configFile)) {
        Write-Warning "No config.json for $id at $configFile - export params first"
        continue
    }

    # Build arguments
    $scriptPath = Join-Path $BaseDir "scripts\run_live.py"
    $args = "--strategy $($strategy.strategy) --pair $($strategy.pair) --timeframe $($strategy.timeframe) --params-file $configFile --instance-dir $instanceDir --instance-id $id --risk $($strategy.risk_pct) --yes"

    # Remove existing service if present
    $existing = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Removing existing service: $serviceName"
        nssm stop $serviceName
        nssm remove $serviceName confirm
    }

    # Install service
    Write-Host "Installing service: $serviceName"
    nssm install $serviceName $PythonPath "$scriptPath $args"
    nssm set $serviceName AppDirectory $BaseDir
    nssm set $serviceName AppStdout (Join-Path $instanceDir "logs\service_stdout.log")
    nssm set $serviceName AppStderr (Join-Path $instanceDir "logs\service_stderr.log")
    nssm set $serviceName AppRotateFiles 1
    nssm set $serviceName AppRotateBytes 10485760  # 10MB
    nssm set $serviceName AppEnvironmentExtra "PYTHONPATH=$BaseDir"

    # Auto-restart on failure
    nssm set $serviceName AppRestartDelay 10000  # 10 seconds

    # Start the service
    nssm start $serviceName
    Write-Host "Started: $serviceName"
}

# Install monitor service
$monitorName = "Trading_Monitor"
$existing = Get-Service -Name $monitorName -ErrorAction SilentlyContinue
if ($existing) {
    nssm stop $monitorName
    nssm remove $monitorName confirm
}

$monitorScript = Join-Path $BaseDir "scripts\run_monitor.py"
$monitorArgs = "--instances-dir $InstancesDir --telegram --interval 60"

nssm install $monitorName $PythonPath "$monitorScript $monitorArgs"
nssm set $monitorName AppDirectory $BaseDir
nssm set $monitorName AppRestartDelay 10000
nssm start $monitorName
Write-Host "Started: $monitorName"

Write-Host ""
Write-Host "All services installed. Check status with:"
Write-Host "  Get-Service Trading_*"
