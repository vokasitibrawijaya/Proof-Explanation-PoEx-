#!/usr/bin/env powershell
# Baseline FL test without blockchain
param([int]$NClients=3, [int]$Rounds=1)

Write-Host "=== Baseline FL Test ===" -ForegroundColor Cyan

$env:POEX_ENABLED="0"
$env:N_CLIENTS=[string]$NClients
$env:MAX_ROUNDS=[string]$Rounds
$env:RUN_ID="baseline_test"
$env:CLEAR_RESULTS="1"

docker compose -f docker-compose-poex.yml up -d --build poex-aggregator
Start-Sleep -Seconds 10

Write-Host "Starting clients..." -ForegroundColor Yellow
docker run --rm --network poex -e CLIENT_ID=0 -e AGGREGATOR_URL="http://poex-aggregator:5000" -e POEX_ENABLED=0 consensus_flblockchain-poex-client python /app/run_poex_distributed_client.py &
docker run --rm --network poex -e CLIENT_ID=1 -e AGGREGATOR_URL="http://poex-aggregator:5000" -e POEX_ENABLED=0 consensus_flblockchain-poex-client python /app/run_poex_distributed_client.py &
docker run --rm --network poex -e CLIENT_ID=2 -e AGGREGATOR_URL="http://poex-aggregator:5000" -e POEX_ENABLED=0 consensus_flblockchain-poex-client python /app/run_poex_distributed_client.py

Write-Host "Checking results..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
if (Test-Path "results/poex_results.csv") {
    Get-Content "results/poex_results.csv"
    Write-Host "SUCCESS!" -ForegroundColor Green
}
