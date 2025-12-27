# Check experiment results
Write-Host "`n=== Docker Container Status ===" -ForegroundColor Cyan
docker ps --filter "name=poex" --format "{{.Names}}: {{.Status}}"

Write-Host "`n=== Results Files ===" -ForegroundColor Cyan
if (Test-Path "results/poex_results.csv") {
    Write-Host "✅ poex_results.csv found" -ForegroundColor Green
    Get-Content "results/poex_results.csv" | Select-Object -First 5
} else {
    Write-Host "❌ poex_results.csv not found" -ForegroundColor Red
}

Write-Host "`n=== Aggregator Logs (last 20 lines) ===" -ForegroundColor Cyan
docker logs --tail 20 poex-aggregator 2>&1

Write-Host "`n=== Client Logs ===" -ForegroundColor Cyan
@("poex-client-0", "poex-client-1", "poex-client-2") | ForEach-Object {
    Write-Host "`n[$_]" -ForegroundColor Yellow
    docker logs --tail 10 $_ 2>&1
}
