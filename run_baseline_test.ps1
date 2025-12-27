# Quick baseline FL test (no PoEx, just FedAvg)
param(
    [int]$NClients = 3,
    [int]$Rounds = 1
)

Write-Host "=== Baseline FL Test (No Blockchain) ===" -ForegroundColor Cyan
Write-Host "Clients: $NClients | Rounds: $Rounds" -ForegroundColor Yellow

# Start aggregator only (no Fabric network)
$env:POEX_ENABLED = "0"
$env:N_CLIENTS = [string]$NClients
$env:MAX_ROUNDS = [string]$Rounds
$env:RUN_ID = "baseline_test"
$env:ATTACK_TYPE = "none"
$env:CLEAR_RESULTS = "1"

docker compose -f docker-compose-poex.yml up -d --build poex-aggregator

Write-Host "Waiting for aggregator..." -ForegroundColor Yellow
for ($i=0; $i -lt 30; $i++) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://localhost:5000/health"
        if ($r.StatusCode -eq 200) {
            Write-Host "✓ Aggregator ready" -ForegroundColor Green
            break
        }
    } catch {
        # Aggregator not ready yet
    }
    Start-Sleep -Seconds 2
}

Write-Host "Starting $NClients clients..." -ForegroundColor Yellow
$jobs = @()
for ($i=0; $i -lt $NClients; $i++) {
    $job = Start-Job -ScriptBlock {
        param($ClientId, $AggrUrl)
        docker run --rm --name "baseline-client-$ClientId" --network poex -e CLIENT_ID=$ClientId -e AGGREGATOR_URL=$AggrUrl -e POEX_ENABLED=0 consensus_flblockchain-poex-client python /app/run_poex_distributed_client.py
    } -ArgumentList $i, "http://poex-aggregator:5000"
    $jobs += $job
}

Write-Host "Waiting for clients to complete..." -ForegroundColor Yellow
$jobs | Wait-Job | Receive-Job | Out-Host

Write-Host "`n=== Results ===" -ForegroundColor Cyan
if (Test-Path "results/poex_results.csv") {
    Get-Content "results/poex_results.csv" | Out-Host
    Write-Host "`n✓ Test completed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ No results file generated" -ForegroundColor Red
}

docker compose -f docker-compose-poex.yml logs poex-aggregator --tail 20
