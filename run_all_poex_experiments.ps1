#!/usr/bin/env pwsh
# Script untuk menjalankan semua skenario eksperimen PoEx
# Sesuai dengan eksperimen_pox.md: Baseline vs Proposed × 3 jenis serangan

$ErrorActionPreference = "Stop"

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  PoEx Complete Experiment Suite" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Skenario yang akan dijalankan:" -ForegroundColor Yellow
Write-Host "  [1] Baseline (PoEx OFF) + No Attack" -ForegroundColor Gray
Write-Host "  [2] Baseline (PoEx OFF) + Label Flipping" -ForegroundColor Gray
Write-Host "  [3] Baseline (PoEx OFF) + Gaussian Noise" -ForegroundColor Gray
Write-Host "  [4] Proposed (PoEx ON) + No Attack" -ForegroundColor Gray
Write-Host "  [5] Proposed (PoEx ON) + Label Flipping" -ForegroundColor Gray
Write-Host "  [6] Proposed (PoEx ON) + Gaussian Noise" -ForegroundColor Gray
Write-Host ""

$MaxRounds = 3
$Threshold = 0.5
$MaliciousClients = "0"  # Client 0 akan jadi malicious di attack scenarios

# Clear previous results
if (Test-Path "results/poex_results.csv") {
    Write-Host "Clearing previous results..." -ForegroundColor Yellow
    Remove-Item "results/poex_results.csv" -Force
}

# Skenario 1: Baseline + No Attack
Write-Host "[1/6] Running: Baseline (PoEx OFF) + No Attack..." -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File .\run_poex_experiment.ps1 `
    -RunId "baseline_no_attack" `
    -MaxRounds $MaxRounds `
    -AggMethod "fedavg" `
    -PoExEnabled 0 `
    -PoExThreshold $Threshold `
    -AttackType "none" `
    -MaliciousRatio 0.0 `
    -Reset 1

Write-Host "✓ Skenario 1 selesai" -ForegroundColor Green
Start-Sleep -Seconds 5

# Skenario 2: Baseline + Label Flipping
Write-Host "[2/6] Running: Baseline (PoEx OFF) + Label Flipping..." -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File .\run_poex_experiment.ps1 `
    -RunId "baseline_label_flip" `
    -MaxRounds $MaxRounds `
    -AggMethod "fedavg" `
    -PoExEnabled 0 `
    -PoExThreshold $Threshold `
    -AttackType "label_flip" `
    -MaliciousRatio 0.33 `
    -Reset 1

Write-Host "✓ Skenario 2 selesai" -ForegroundColor Green
Start-Sleep -Seconds 5

# Skenario 3: Baseline + Gaussian Noise
Write-Host "[3/6] Running: Baseline (PoEx OFF) + Gaussian Noise..." -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File .\run_poex_experiment.ps1 `
    -RunId "baseline_gaussian" `
    -MaxRounds $MaxRounds `
    -AggMethod "fedavg" `
    -PoExEnabled 0 `
    -PoExThreshold $Threshold `
    -AttackType "gaussian_noise" `
    -MaliciousRatio 0.33 `
    -AttackSigma 0.1 `
    -Reset 1

Write-Host "✓ Skenario 3 selesai" -ForegroundColor Green
Start-Sleep -Seconds 5

# Skenario 4: Proposed + No Attack
Write-Host "[4/6] Running: Proposed (PoEx ON) + No Attack..." -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File .\run_poex_experiment.ps1 `
    -RunId "proposed_no_attack" `
    -MaxRounds $MaxRounds `
    -AggMethod "fedavg" `
    -PoExEnabled 1 `
    -PoExThreshold $Threshold `
    -AttackType "none" `
    -MaliciousRatio 0.0 `
    -Reset 1

Write-Host "✓ Skenario 4 selesai" -ForegroundColor Green
Start-Sleep -Seconds 5

# Skenario 5: Proposed + Label Flipping
Write-Host "[5/6] Running: Proposed (PoEx ON) + Label Flipping..." -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File .\run_poex_experiment.ps1 `
    -RunId "proposed_label_flip" `
    -MaxRounds $MaxRounds `
    -AggMethod "fedavg" `
    -PoExEnabled 1 `
    -PoExThreshold $Threshold `
    -AttackType "label_flip" `
    -MaliciousRatio 0.33 `
    -Reset 1

Write-Host "✓ Skenario 5 selesai" -ForegroundColor Green
Start-Sleep -Seconds 5

# Skenario 6: Proposed + Gaussian Noise
Write-Host "[6/6] Running: Proposed (PoEx ON) + Gaussian Noise..." -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File .\run_poex_experiment.ps1 `
    -RunId "proposed_gaussian" `
    -MaxRounds $MaxRounds `
    -AggMethod "fedavg" `
    -PoExEnabled 1 `
    -PoExThreshold $Threshold `
    -AttackType "gaussian_noise" `
    -MaliciousRatio 0.33 `
    -AttackSigma 0.1 `
    -Reset 1

Write-Host "✓ Skenario 6 selesai" -ForegroundColor Green
Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  Semua Eksperimen Selesai!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Hasil tersimpan di: results/poex_results.csv" -ForegroundColor Yellow
Write-Host ""
Write-Host "Untuk visualisasi, jalankan:" -ForegroundColor Yellow
Write-Host "  python scripts/visualize_poex_results.py" -ForegroundColor Gray
