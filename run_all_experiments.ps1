#!/usr/bin/env pwsh
# PoEx Experiment Runner - Run all 3 required experiments

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  PoEx Complete Experiment Suite" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

$ErrorActionPreference = "Continue"

function Run-Experiment {
    param(
        [string]$Name,
        [string]$AttackType,
        [string]$MaliciousClients,
        [int]$MaxRounds,
        [string]$PoExEnabled = "true"
    )
    
    Write-Host "`n===============================================" -ForegroundColor Yellow
    Write-Host "EXPERIMENT: $Name" -ForegroundColor Cyan
    Write-Host "===============================================" -ForegroundColor Yellow
    Write-Host "  Attack: $AttackType" -ForegroundColor White
    Write-Host "  Malicious: $MaliciousClients" -ForegroundColor White
    Write-Host "  Rounds: $MaxRounds" -ForegroundColor White
    Write-Host "  PoEx: $PoExEnabled`n" -ForegroundColor White
    
    Write-Host "[1/5] Stopping previous containers..." -ForegroundColor Gray
    docker compose -f docker-compose-poex.yml down 2>&1 | Out-Null
    Start-Sleep -Seconds 3
    
    $env:ATTACK_TYPE = $AttackType
    $env:MALICIOUS_RATIO = "0.33"
    $env:MALICIOUS_CLIENTS = $MaliciousClients
    $env:MAX_ROUNDS = $MaxRounds
    $env:POEX_ENABLED = $PoExEnabled
    $env:CLEAR_RESULTS = "0"
    
    Write-Host "[2/5] Starting Fabric infrastructure..." -ForegroundColor Gray
    docker compose -f docker-compose-poex.yml up -d 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to start infrastructure" -ForegroundColor Red
        return $false
    }
    
    Write-Host "[3/5] Waiting 20s for orderer..." -ForegroundColor Gray
    Start-Sleep -Seconds 20
    
    Write-Host "[4/5] Starting FL clients..." -ForegroundColor Gray
    docker compose -f docker-compose-poex.yml --profile clients up -d 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to start clients" -ForegroundColor Red
        return $false
    }
    
    $WaitTime = ($MaxRounds * 30) + 30
    Write-Host "[5/5] Running experiment (~$WaitTime seconds)..." -ForegroundColor Gray
    
    $Elapsed = 0
    while ($Elapsed -lt $WaitTime) {
        Start-Sleep -Seconds 15
        $Elapsed += 15
        
        $Rounds = docker logs poex-aggregator 2>&1 | Select-String "\[Round" | Measure-Object
        if ($Rounds.Count -gt 0) {
            Write-Host "  Progress: ~$($Rounds.Count) rounds completed..." -ForegroundColor DarkGray
        }
        
        $RunningClients = docker ps --filter "name=poex-client" --format "{{.Names}}" | Measure-Object
        if ($RunningClients.Count -eq 0 -and $Elapsed -gt 30) {
            Write-Host "  All clients finished" -ForegroundColor Green
            break
        }
    }
    
    Write-Host "`nResults:" -ForegroundColor Cyan
    docker logs poex-aggregator 2>&1 | Select-String "\[Round" | Select-Object -Last 5 | ForEach-Object {
        Write-Host "  $_" -ForegroundColor White
    }
    
    Write-Host "`nCompleted: $Name`n" -ForegroundColor Green
    return $true
}

# EXPERIMENT 1: Multi-round sign_flip
$Result1 = Run-Experiment -Name "Multi-round sign_flip (3 rounds)" -AttackType "sign_flip" -MaliciousClients "1" -MaxRounds 3

if (-not $Result1) {
    Write-Host "Experiment 1 had issues, continuing...`n" -ForegroundColor Yellow
}

# EXPERIMENT 2: Label flip attack
$Result2 = Run-Experiment -Name "Label flip attack" -AttackType "label_flip" -MaliciousClients "2" -MaxRounds 2

if (-not $Result2) {
    Write-Host "Experiment 2 had issues, continuing...`n" -ForegroundColor Yellow
}

# EXPERIMENT 3: Baseline (no PoEx)
$Result3 = Run-Experiment -Name "Baseline (no PoEx)" -AttackType "sign_flip" -MaliciousClients "1" -MaxRounds 3 -PoExEnabled "false"

if (-not $Result3) {
    Write-Host "Experiment 3 had issues, continuing...`n" -ForegroundColor Yellow
}

# CLEANUP
Write-Host "`n===============================================" -ForegroundColor Yellow
Write-Host "Cleanup" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Yellow
docker compose -f docker-compose-poex.yml down 2>&1 | Out-Null
Write-Host "All containers stopped`n" -ForegroundColor Green

# RESULTS SUMMARY
Write-Host "`n================================================" -ForegroundColor Green
Write-Host "  EXPERIMENT RESULTS SUMMARY" -ForegroundColor Green
Write-Host "================================================`n" -ForegroundColor Green

if (Test-Path "results/poex_results.csv") {
    $Lines = Get-Content results/poex_results.csv | Measure-Object -Line
    Write-Host "Results file: results/poex_results.csv" -ForegroundColor Green
    Write-Host "Total rows: $($Lines.Lines)`n" -ForegroundColor White
    
    Write-Host "Last 10 results:" -ForegroundColor Cyan
    Get-Content results/poex_results.csv -Tail 11 | ForEach-Object {
        Write-Host "   $_" -ForegroundColor White
    }
    
    Write-Host "`n===============================================" -ForegroundColor Cyan
    Write-Host "NEXT STEP: Generate visualizations" -ForegroundColor Yellow
    Write-Host "===============================================" -ForegroundColor Cyan
    Write-Host "`nRun: python scripts/visualize_poex_results.py`n" -ForegroundColor White
    
} else {
    Write-Host "No results file generated" -ForegroundColor Yellow
    Write-Host "Check logs: docker logs poex-aggregator`n" -ForegroundColor Gray
}

Write-Host "================================================" -ForegroundColor Green
Write-Host "  ALL EXPERIMENTS COMPLETED" -ForegroundColor Green
Write-Host "================================================`n" -ForegroundColor Green
