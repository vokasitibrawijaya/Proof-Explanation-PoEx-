# Quick monitoring script for PoEx experiment
Write-Host "`n=== PoEx Experiment Monitor ===" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'HH:mm:ss')"

Write-Host "`n[Containers]"
docker ps --filter "name=poex" --format "  {{.Names}}: {{.Status}}" 2>$null

Write-Host "`n[Clients]"
docker ps -a --filter "name=client" --format "  {{.Names}}: {{.Status}}" 2>$null

Write-Host "`n[Activity]"
$submits = (docker logs poex-aggregator 2>&1 | Select-String "POST /submit_update").Count
Write-Host "  Submits received: $submits"

Write-Host "`n[Results]"
if (Test-Path "results/poex_results.csv") {
    Write-Host "  ✅ poex_results.csv EXISTS" -ForegroundColor Green
    $lines = (Get-Content "results/poex_results.csv").Count
    Write-Host "  Lines: $lines"
    Write-Host "`n--- Content ---"
    Get-Content "results/poex_results.csv"
} else {
    Write-Host "  ⏳ No results yet" -ForegroundColor Yellow
}

Write-Host "`n[Recent Aggregator Logs]"
docker logs --tail 10 poex-aggregator 2>&1 | Select-Object -Last 10
