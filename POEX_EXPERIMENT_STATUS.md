# PoEx Experiment Status dan Panduan Lengkap

## Status Persiapan Eksperimen

### ✅ Yang Sudah Selesai:

#### 1. **Perbaikan Infrastruktur**
- ✅ Fixed channel creation issue di `run_poex_experiment.ps1`:
  - Meningkatkan wait time dari 5s ke 10s
  - Memperbaiki regex detection untuk channel yang sudah exist
  - Menambahkan check "Received block: 0" untuk success indicator
  
- ✅ Hyperledger Fabric network berfungsi:
  - Orderer, Peer, CLI containers running successfully
  - Channel creation dan join berhasil
  - Chaincode deployment via CCAAS working

#### 2. **Penambahan Metrik Lengkap**
- ✅ Modified `run_poex_distributed_aggregator.py`:
  - Menambahkan import: `precision_score`, `recall_score`, `f1_score`
  - Mengubah `evaluate_global()` return dict dengan 4 metrik:
    - `accuracy`
    - `precision`
    - `recall`
    - `f1`
  - Updated CSV header dan penulisan untuk menyertakan semua metrik
  
- ✅ CSV Output Structure (Updated):
  ```
  run_id, method, poex_enabled, poex_threshold, attack_type, malicious_ratio, 
  malicious_clients, round, global_accuracy, global_precision, global_recall, 
  global_f1, avg_local_accuracy, avg_nsds, accepted_updates, rejected_updates, 
  avg_poex_latency_ms
  ```

#### 3. **Script Eksperimen Lengkap**
- ✅ Created `run_all_poex_experiments.ps1`:
  - Menjalankan 6 skenario secara berurutan
  - 3 baseline (PoEx OFF): no_attack, label_flip, gaussian_noise
  - 3 proposed (PoEx ON): no_attack, label_flip, gaussian_noise
  - Setiap skenario: 3 rounds, 3 clients
  - Automatic cleanup between runs
  
#### 4. **Visualisasi dan Analisis**
- ✅ Created `scripts/visualize_poex_results.py`:
  - Grafik accuracy comparison (baseline vs proposed)
  - Grafik precision/recall/F1 untuk semua skenario
  - Security metrics (accepted vs rejected updates)
  - PoEx latency overhead
  - Summary statistics table
  
- ✅ Created `scripts/visualize_shap_integrity.py`:
  - SHAP value comparison: normal vs malicious nodes
  - Heatmap of SHAP patterns
  - KL divergence calculation dan visualization
  - Demonstrasi bahwa malicious nodes punya pattern berbeda

### 📋 Skenario Eksperimen (Sesuai eksperimen_pox.md)

| No | Method | PoEx | Attack Type | Malicious Ratio | Expected Outcome |
|----|--------|------|-------------|-----------------|------------------|
| 1  | Baseline | OFF | none | 0.0 | High accuracy, no protection |
| 2  | Baseline | OFF | label_flip | 0.33 | **Degraded accuracy** (poisoning succeeds) |
| 3  | Baseline | OFF | gaussian_noise | 0.33 | **Degraded accuracy** (noise affects model) |
| 4  | Proposed | ON  | none | 0.0 | High accuracy, low overhead |
| 5  | Proposed | ON  | label_flip | 0.33 | **Maintained accuracy** (PoEx rejects malicious) |
| 6  | Proposed | ON  | gaussian_noise | 0.33 | **Maintained accuracy** (PoEx detects noise) |

### 🎯 Metrik Evaluasi (Untuk Paper IEEE)

#### Model Performance:
- ✅ **Accuracy**: Global model accuracy on test set
- ✅ **Precision**: True positives / (True positives + False positives)
- ✅ **Recall**: True positives / (True positives + False negatives)
- ✅ **F1-Score**: Harmonic mean of precision and recall

#### Security:
- ✅ **Accepted Updates**: Count of updates yang lolos validasi PoEx
- ✅ **Rejected Updates**: Count of updates yang ditolak PoEx
- ✅ **Success Rate**: Percentage of attack attempts detected and blocked

#### Efficiency:
- ✅ **PoEx Latency**: Average time (ms) untuk validasi SHAP + KL divergence
- ✅ **Overhead**: Comparison of total training time dengan/tanpa PoEx

#### XAI Integrity:
- ✅ **SHAP Visualizations**: Bar charts dan heatmaps
- ✅ **KL Divergence (NSDS)**: Quantitative measure of explanation difference
- ✅ **Pattern Detection**: Visual proof bahwa malicious nodes berbeda

### 📊 Expected Results Summary

**Hipotesis (Sesuai Paper):**
1. **Baseline + Attack** → Accuracy turun signifikan (e.g., 95% → 60%)
2. **Proposed + Attack** → Accuracy tetap tinggi (e.g., 95% → 92%)
3. **PoEx Overhead** → Minimal (~50ms per update)
4. **Detection Rate** → High (>90% malicious updates rejected)

### 🚀 Cara Menjalankan

#### Opsi 1: Run All Experiments (Recommended)
```powershell
# Run semua 6 skenario (estimated 30-60 minutes total)
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all_poex_experiments.ps1

# Generate visualizations
python scripts/visualize_poex_results.py
python scripts/visualize_shap_integrity.py
```

#### Opsi 2: Run Individual Experiment
```powershell
# Contoh: Proposed dengan label flipping attack
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_poex_experiment.ps1 `
    -RunId "proposed_label_flip" `
    -MaxRounds 3 `
    -AggMethod "fedavg" `
    -PoExEnabled 1 `
    -PoExThreshold 0.5 `
    -AttackType "label_flip" `
    -MaliciousRatio 0.33 `
    -Reset 1
```

### 📁 Output Files

After running experiments:
```
results/
├── poex_results.csv              # Raw experiment data
└── visualizations/
    ├── accuracy_comparison.png    # Baseline vs Proposed accuracy
    ├── precision_recall_f1.png    # Detailed metrics per scenario
    ├── security_metrics.png       # Accepted vs rejected updates
    ├── poex_latency.png          # Validation overhead
    ├── shap_comparison_bar.png   # SHAP values comparison
    ├── shap_heatmap.png          # Feature importance heatmap
    ├── kl_divergence.png         # NSDS metric visualization
    └── summary_statistics.csv    # Aggregated results table
```

### 🔍 Troubleshooting

**Issue: Channel creation fails**
- Solution: Sudah fixed di `run_poex_experiment.ps1` dengan increased wait time

**Issue: Aggregator exits with "PoEx gateway not reachable"**
- Check: `docker logs poex-gateway`
- Solution: Ensure chaincode is deployed and gateway can connect to Fabric

**Issue: Clients hang during training**
- Check: `docker logs poex-aggregator`
- Solution: Verify aggregator is running and accessible at port 5001

**Issue: No results file generated**
- Check: Experiment completed all rounds?
- Check: `docker logs poex-aggregator` for any errors during result writing

### 📝 Next Steps for Paper

1. **Run Complete Experiments**:
   - Execute `run_all_poex_experiments.ps1`
   - Verify all 6 scenarios complete successfully
   - Generate `poex_results.csv` with ~18 rows (6 scenarios × 3 rounds)

2. **Generate Visualizations**:
   - Run `visualize_poex_results.py` for performance graphs
   - Run `visualize_shap_integrity.py` for XAI validation
   - Include all PNGs in paper as figures

3. **Write Results Section**:
   - Use `summary_statistics.csv` for results table
   - Compare accuracy drop: Baseline (large) vs Proposed (small)
   - Highlight PoEx detection rate and latency overhead
   - Show SHAP visualizations prove XAI-based detection works

4. **Discussion Points**:
   - PoEx successfully detects and rejects malicious updates **before ledger**
   - Minimal overhead (~50ms) compared to security benefit
   - SHAP explanations provide interpretable security mechanism
   - Blockchain ensures immutability of accepted updates only

### ✅ Validation Checklist

- [x] PoEx chaincode implements KL divergence validation
- [x] Sequential client execution (eksperimen_pox.md requirement)
- [x] Precision, Recall, F1 metrics added
- [x] 6 experiment scenarios prepared
- [x] Visualization scripts ready
- [x] SHAP integrity demonstration included
- [ ] **TODO**: Run all experiments and verify results
- [ ] **TODO**: Generate paper-ready figures
- [ ] **TODO**: Write results section with data

---

**Status**: Infrastruktur dan skrip siap. Tinggal menjalankan eksperimen lengkap dan menganalisis hasil.
