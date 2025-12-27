# Quick status check
Write-Host "=== POEX EXPERIMENT STATUS ===" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'HH:mm:ss')`n"

# Check if result exists
if (Test-Path "results/poex_results.csv") {
    Write-Host "✅ HASIL EKSPERIMEN TERSEDIA!" -ForegroundColor Green
    Write-Host "`n--- poex_results.csv ---"
    Get-Content "results/poex_results.csv"
    Write-Host "`n✅ Eksperimen SELESAI dan BERHASIL!" -ForegroundColor Green
} else {
    Write-Host "⏳ Hasil belum ada, masih berjalan..." -ForegroundColor Yellow
    
    # Count submits
    try {
        $submits = (docker logs poex-aggregator 2>&1 | Select-String "POST /submit_update").Count
        Write-Host "`nProgress: $submits/6 updates submitted (need 6 for 2 rounds × 3 clients)"
    } catch {}
    
    # Client status
    Write-Host "`nClient Containers:"
    docker ps --filter "name=client" --format "  {{.Names}}: {{.Status}}" 2>$null
}
