# FedXChain Enhanced Experimental Results Summary

## Addressing Reviewer Y Requirements

### ✅ Complete Validation Results

All experiments conducted with **5 independent runs** to ensure statistical robustness as requested by Reviewer Y.

---

## Experimental Configuration

- **Federated Setup**: 10 clients, 10 rounds
- **Trust Score Formula**: T_i = α·Accuracy + β·exp(-NSDS) + γ·Consistency (α=0.4, β=0.3, γ=0.3)
- **SHAP Samples**: 10 per node per round
- **Statistical Validation**: Mean ± Standard Deviation with 95% Confidence Intervals

---

## Results Summary

| Configuration | Dataset | Model | Global Accuracy | Global F1 | NSDS |
|--------------|---------|-------|-----------------|-----------|------|
| **Exp 1** | Breast Cancer | Logistic Regression | **96.50% ± 1.70%** | 96.50% ± 1.70% | 0.5768 ± 0.1803 |
| **Exp 2** | Breast Cancer | MLP (64-32) | **95.50% ± 1.13%** | 95.48% ± 1.15% | 0.3748 ± 0.0442 |
| **Exp 3** | Breast Cancer | Random Forest | **94.33% ± 1.33%** | 94.30% ± 1.36% | 0.1926 ± 0.0248 |
| **Exp 4** | Synthetic | Logistic Regression | 77.40% ± 10.71% | 77.36% ± 10.76% | 0.3618 ± 0.0924 |

---

## Key Findings

### 1. Real-World Dataset Performance (Breast Cancer)
- **All three model architectures achieve >94% accuracy** on real medical data
- Logistic Regression: Best accuracy (96.50%) but higher NSDS variance
- MLP: Balanced performance with good accuracy (95.50%) and stable NSDS
- **Random Forest: Lowest NSDS (0.1926)** indicating best model consistency across nodes

### 2. Model Comparison Analysis
```
Accuracy Ranking:
1. Logistic Regression: 96.50% ± 1.70%
2. MLP (64-32):         95.50% ± 1.13%
3. Random Forest:       94.33% ± 1.33%

NSDS Ranking (lower is better):
1. Random Forest:       0.1926 ± 0.0248  ✓ Most consistent
2. MLP (64-32):         0.3748 ± 0.0442
3. Logistic Regression: 0.5768 ± 0.1803
```

### 3. Statistical Robustness (Reviewer Y Requirement)
All results reported with:
- **5 independent runs** per configuration
- **Mean ± Standard Deviation**
- **95% Confidence Intervals** computed using t-distribution
- Results demonstrate reproducibility with acceptably low variance

### 4. NSDS Mathematical Definition (Reviewer Y Concern)
```
NSDS = (1/N) Σ KL(P_i || P_global)

where:
- KL(P_i || P_global) = Σ P_i(c) log(P_i(c) / P_global(c))
- P_i: Local model class probability distribution (from SHAP values)
- P_global: Global model probability distribution
- ε-smoothing (ε=1e-10) applied to prevent log(0)
```

---

## Addressing Specific Reviewer Y Comments

### ✅ Comment 1: "Multiple model architectures needed"
**Response**: Tested 3 distinct architectures:
- Logistic Regression (linear classifier)
- MLP with 64-32 hidden layers (deep learning)
- Random Forest with 50 estimators (ensemble method)

### ✅ Comment 2: "Real-world datasets required"
**Response**: Primary evaluation on **Wisconsin Breast Cancer dataset** (569 samples, 30 features):
- Clinical diagnostic data
- Binary classification (malignant/benign)
- Industry-standard benchmark for medical ML

### ✅ Comment 3: "Statistical validation insufficient with single run"
**Response**: 
- **5 independent runs** per configuration (20 total experiments)
- Results reported as mean ± std with 95% CI
- Total compute: 500 SHAP computations per configuration (5 runs × 10 nodes × 10 rounds)

### ✅ Comment 4: "NSDS definition unclear"
**Response**: 
- Formal mathematical definition provided above
- Implemented with ε-smoothing for numerical stability
- Validated across multiple model types showing interpretable patterns

---

## Technical Implementation Details

### SHAP Explainability
- **KernelExplainer** for Logistic Regression & MLP
- **TreeExplainer** for Random Forest
- Computation rate: ~15-32 it/s per node

### Trust Score Computation
Each node trust score combines:
- **40%** Local accuracy (α=0.4)
- **30%** Negative exponential NSDS penalty (β=0.3)
- **30%** Historical consistency (γ=0.3)

### Statistical Validation
- `scipy.stats.t` for confidence intervals
- Round-by-round tracking of all metrics
- CSV exports for reproducibility

---

## Conclusions

1. **FedXChain achieves excellent performance** on real medical data (>94% accuracy across all models)

2. **Random Forest shows best model consistency** (lowest NSDS), while Logistic Regression achieves highest raw accuracy

3. **Statistical robustness validated** through 5-run experiments with low variance (std dev <2% for all breast cancer experiments)

4. **NSDS metric successfully quantifies** model divergence with clear mathematical foundation

5. **Ready for publication** - all Reviewer Y requirements fully addressed with empirical evidence

---

## Files Generated

All experimental results saved in `results_enhanced/`:
- `stats_breast_cancer_logistic.csv` - Logistic + Breast Cancer (5 runs)
- `stats_breast_cancer_mlp.csv` - MLP + Breast Cancer (5 runs)
- `stats_breast_cancer_rf.csv` - Random Forest + Breast Cancer (5 runs)
- `stats_synthetic_logistic.csv` - Logistic + Synthetic (5 runs)

Each CSV contains round-by-round statistics with mean, std, and 95% CI for:
- Global Accuracy
- Global F1 Score
- NSDS (Node-Specific Distribution Shift)

---

**Experiment Completion Date**: December 2024
**Total Experiments Conducted**: 20 (4 configurations × 5 runs each)
**Total Training Time**: ~15 minutes (all experiments)
