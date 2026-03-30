param(
    [string]$Date = "2018-08-04",
    [string]$BaseUrl = "http://localhost:5000"
)

$ErrorActionPreference = "Stop"

Write-Host "Testing API at $BaseUrl"
Write-Host "Requested date: $Date"

try {
    $health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get
    Write-Host "[OK] /health -> status=$($health.status)"
}
catch {
    Write-Error "Health check failed: $($_.Exception.Message)"
    exit 1
}

try {
    $payload = @{ date = $Date } | ConvertTo-Json
    $result = Invoke-RestMethod -Uri "$BaseUrl/predict-day" -Method Post -ContentType "application/json" -Body $payload

    Write-Host "[OK] /predict-day"
    Write-Host "date: $($result.date)"
    Write-Host "daily_total: $([math]::Round([double]$result.daily_total, 2))"
    Write-Host "hour_count: $($result.hourly.Count)"
    Write-Host "used_targets: $($result.used_targets -join ', ')"

    if ($result.skipped_targets) {
        Write-Host "skipped_targets:"
        $result.skipped_targets.PSObject.Properties | ForEach-Object {
            Write-Host "  - $($_.Name): $($_.Value)"
        }
    }

    Write-Host "Results"
    $result.hourly | Select-Object | ForEach-Object {
        Write-Host "  $($_.timestamp) -> $([math]::Round([double]$_.aggregate_usage, 2))"
    }
}
catch {
    Write-Error "Prediction request failed: $($_.Exception.Message)"
    exit 1
}
