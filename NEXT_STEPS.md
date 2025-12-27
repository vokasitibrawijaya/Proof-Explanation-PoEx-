# Next Steps - Complete PoEx Experiments

## Current Status ✅

- **Critical bug fixed**: Rejected PoEx updates now properly excluded from aggregation
- **Infrastructure validated**: All Fabric components working (orderer, peer, gateway, chaincode)
- **PoEx detection confirmed**: Round 0 successfully detected sign_flip attack (NSDS=0.964 > 0.5)

## What's Missing (3 experiments + visualizations)

### ✅ **Recommended: Run automated script**

```powershell
.\run_all_experiments.ps1
```

This script automatically runs **all 3 remaining experiments**:
1. Multi-round sign_flip (3 rounds) - trust evolution
2. Label flip attack (2 rounds) - training data poisoning
3. Baseline comparison (3 rounds, no PoEx) - shows improvement

**Time:** ~8-10 minutes total  
**Output:** `results/poex_results.csv` with complete dataset

---

### 🔧 **Alternative: Run experiments manually**

#### Experiment 1: Multi-round trust evolution
```powershell
# Clean start
docker compose -f docker-compose-poex.yml down
Start-Sleep -Seconds 3

# Configure
$env:ATTACK_TYPE="sign_flip"
$env:MALICIOUS_CLIENTS="1"
$env:MAX_ROUNDS="3"
$env:CLEAR_RESULTS="1"

# Run
docker compose -f docker-compose-poex.yml up -d
Start-Sleep -Seconds 20
docker compose -f docker-compose-poex.yml --profile clients up -d

# Wait 2-3 minutes, then check
docker logs poex-aggregator | Select-String "\[Round"
Get-Content results/poex_results.csv
```

#### Experiment 2: Label flip attack
```powershell
docker compose -f docker-compose-poex.yml down
Start-Sleep -Seconds 3

$env:ATTACK_TYPE="label_flip"
$env:MALICIOUS_CLIENTS="2"
$env:MAX_ROUNDS="2"
$env:CLEAR_RESULTS="0"  # Append results

docker compose -f docker-compose-poex.yml up -d
Start-Sleep -Seconds 20
docker compose -f docker-compose-poex.yml --profile clients up -d
```

#### Experiment 3: Baseline (no PoEx)
```powershell
docker compose -f docker-compose-poex.yml down
Start-Sleep -Seconds 3

$env:ATTACK_TYPE="sign_flip"
$env:MALICIOUS_CLIENTS="1"
$env:MAX_ROUNDS="3"
$env:POEX_ENABLED="false"
$env:CLEAR_RESULTS="0"

docker compose -f docker-compose-poex.yml up -d
Start-Sleep -Seconds 20
docker compose -f docker-compose-poex.yml --profile clients up -d
```

---

## Generate Visualizations (Step 4)

After collecting data from all experiments:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Generate plots
python scripts/visualize_poex_results.py
python scripts/visualize_shap_integrity.py

# Check figures
ls results/*.png
ls results/*.pdf
```

**Expected outputs:**
- NSDS distribution histogram
- Trust evolution line plot
- SHAP pattern heatmaps
- Acceptance rate comparison

---

## Monitoring Tips

### Check if experiment is running:
```powershell
docker compose -f docker-compose-poex.yml ps
```

### Watch progress:
```powershell
# See round completions
docker logs poex-aggregator --follow | Select-String "\[Round"

# Check client activity
docker logs poex-client-0 --tail 20
docker logs poex-client-1 --tail 20  # malicious
docker logs poex-client-2 --tail 20
```

### Verify results:
```powershell
# Check if results file exists
Test-Path results/poex_results.csv

# View results
Get-Content results/poex_results.csv

# Count experiments
(Get-Content results/poex_results.csv | Measure-Object -Line).Lines
```

---

## Expected Timeline

| Task | Time | Status |
|------|------|--------|
| Experiment 1 (multi-round) | 3 min | ⏳ Pending |
| Experiment 2 (label_flip) | 2 min | ⏳ Pending |
| Experiment 3 (baseline) | 3 min | ⏳ Pending |
| Generate visualizations | 2 min | ⏳ Pending |
| **Total** | **~10 min** | |

---

## Troubleshooting

### Clients crash with "Connection refused":
- Aggregator not ready yet
- **Fix:** Increase wait time to 25-30 seconds before starting clients

### No results file generated:
- Experiment didn't complete all rounds
- **Check:** `docker logs poex-aggregator | Select-String "\[Round"`
- **Fix:** Ensure clients are running: `docker ps | Select-String client`

### Containers exit immediately:
- Network issue or configuration error
- **Fix:** Clean restart with `docker compose down; docker compose up -d`

---

## Files Created

- ✅ `run_all_experiments.ps1` - Automated script for all 3 experiments
- ✅ `EXPERIMENT_STATUS.md` - Detailed status and requirements mapping
- ✅ `NEXT_STEPS.md` - This file

## Success Criteria

After completing all experiments, you should have:

1. ✅ `results/poex_results.csv` with 8-10 rows (3+2+3 rounds)
2. ✅ NSDS values showing: honest < 0.5, malicious > 0.5
3. ✅ Trust evolution: malicious client trust decreasing over rounds
4. ✅ Baseline showing: all updates accepted (no PoEx protection)
5. ✅ Visualization plots ready for IEEE Access paper

---

## Ready to Start?

**Recommended command:**
```powershell
.\run_all_experiments.ps1
```

This will complete 100% of eksperimen_pox.md requirements automatically.
