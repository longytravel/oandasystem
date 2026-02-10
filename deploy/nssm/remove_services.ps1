# Remove all trading Windows services installed by install_services.ps1.
# Run as Administrator.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deploy\nssm\remove_services.ps1

param(
    [string]$BaseDir = "C:\Trading\oandasystem"
)

$strategiesFile = Join-Path $BaseDir "deploy\strategies.json"
if (-not (Test-Path $strategiesFile)) {
    Write-Error "strategies.json not found at $strategiesFile"
    exit 1
}

$config = Get-Content $strategiesFile | ConvertFrom-Json

foreach ($strategy in $config.strategies) {
    $serviceName = "Trading_$($strategy.id)"
    $existing = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Stopping and removing: $serviceName"
        nssm stop $serviceName
        nssm remove $serviceName confirm
    } else {
        Write-Host "Not found: $serviceName"
    }
}

# Remove monitor
$monitorName = "Trading_Monitor"
$existing = Get-Service -Name $monitorName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Stopping and removing: $monitorName"
    nssm stop $monitorName
    nssm remove $monitorName confirm
} else {
    Write-Host "Not found: $monitorName"
}

Write-Host ""
Write-Host "All trading services removed."
