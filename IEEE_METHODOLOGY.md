# Metodologi Eksperimen Ilmiah untuk IEEE Access
# FedXChain: Federated Explainable Blockchain with Proof of Explanation (PoEx)

## 1. Executive Summary

Dokumen ini menjelaskan metodologi eksperimen yang dirancang sesuai standar IEEE Access untuk evaluasi sistem FedXChain dengan konsensus Proof of Explanation (PoEx). Eksperimen mengikuti prinsip rigor ilmiah dengan:

- **Baseline comparison**: FedAvg vs FedXChain
- **Multiple attack scenarios**: Label Flipping, Gaussian Noise, Sign Flipping
- **Statistical validation**: 10 independent runs per configuration
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1-Score, ASR
- **Hypothesis testing**: Paired t-test, ANOVA, Mann-Whitney U

---

## 2. Research Questions

**RQ1**: Does FedXChain with PoEx consensus improve model accuracy compared to standard FedAvg?

**RQ2**: Is FedXChain more resilient against model poisoning attacks (label flipping, Gaussian noise)?

**RQ3**: What is the computational overhead of SHAP-based explainability in federated learning?

**RQ4**: How do trust mechanism parameters (α, β, γ) affect system performance?

---

## 3. Experimental Design

### 3.1 Independent Variables

| Variable | Type | Values |
|----------|------|--------|
| **Method** | Categorical | FedAvg (baseline), FedXChain (proposed) |
| **Attack Type** | Categorical | None, Label Flip, Gaussian Noise, Sign Flip |
| **Attack Intensity** | Continuous | 0.0, 0.3, 0.5, 1.0 |
| **Malicious Ratio** | Continuous | 0%, 10%, 20%, 30%, 40% |
| **Trust Weights** | Continuous | (α, β, γ) combinations |

### 3.2 Dependent Variables

#### Model Performance
- **Accuracy**: Proportion of correct predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall

#### Security Metrics
- **Attack Success Rate (ASR)**: (Baseline_Acc - Attack_Acc) / Baseline_Acc × 100%
- **Accuracy Degradation**: Absolute decrease in accuracy under attack

#### Explainability Metrics
- **NSDS**: Node-Specific Divergence Score (cosine distance from mean SHAP)
- **SHAP Consistency**: Variance of SHAP values across nodes
- **Trust Score Distribution**: Min, Max, Mean, Std of trust scores

#### Efficiency Metrics
- **Training Time**: Time per round (seconds)
- **Communication Cost**: Size of transmitted data
- **Computational Overhead**: Additional computation for SHAP

### 3.3 Control Variables

- Dataset: Fixed (Synthetic or Breast Cancer)
- Number of clients: 10
- Number of rounds: 20
- Local epochs: 1
- Non-IID distribution: Dirichlet α = 0.5
- Random seeds: [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]

---

## 4. Experimental Procedure

### 4.1 Data Preparation

```
1. Load dataset (Synthetic: n=1000, f=20; or Breast Cancer: n=569, f=30)
2. Standardize features using StandardScaler
3. Split into train (80%) and test (20%)
4. Partition training data into 10 clients using Dirichlet distribution (α=0.5)
5. Assign malicious nodes based on attack scenario
```

### 4.2 Training Protocol

```
For each experimental configuration:
  For run = 1 to 10:
    1. Initialize global model (SGD Classifier)
    2. For round = 1 to 20:
       a. Distribute global model to all clients
       b. Each client trains locally for 1 epoch
       c. Malicious clients apply attacks (if applicable)
       d. Clients compute SHAP values (if FedXChain)
       e. Aggregation:
          - FedAvg: Weighted average by sample size
          - FedXChain: Trust-weighted aggregation using ETASR
       f. Evaluate global model on test set
       g. Record metrics
    3. Save results for this run
  Aggregate statistics across 10 runs
```

### 4.3 Attack Implementation

#### Label Flipping Attack
```python
# Flip attack_intensity% of labels
flip_indices = random.choice(len(y_train), int(len(y_train) * attack_intensity))
y_train[flip_indices] = 1 - y_train[flip_indices]
```

#### Gaussian Noise Attack
```python
# Add Gaussian noise to model weights
noise = normal(0, attack_intensity, shape=weights.shape)
corrupted_weights = weights + noise
```

#### Sign Flipping Attack
```python
# Reverse signs of all weights
corrupted_weights = -weights
```

### 4.4 Trust Mechanism (FedXChain)

```
For each node i:
  1. Compute accuracy: a_i = accuracy on global test set
  2. Compute SHAP values: S_i = mean absolute SHAP
  3. Compute NSDS: d_i = 1 - cosine_similarity(S_i, mean(S))
  4. Update trust: T_i = α·a_i + β·(1-d_i) + γ·T_i^(prev)
  5. Normalize: T_i = T_i / sum(T_all)

Aggregate weights: W_global = Σ(T_i · W_i)
```

---

## 5. Statistical Analysis

### 5.1 Descriptive Statistics

For each configuration (method × attack_type):
- **Mean ± Std**: Average performance across 10 runs
- **95% CI**: Confidence interval using t-distribution
- **Min/Max**: Range of observed values

### 5.2 Hypothesis Testing

#### Test 1: Paired t-test
- **H0**: μ(FedAvg) = μ(FedXChain)
- **H1**: μ(FedAvg) ≠ μ(FedXChain)
- **α**: 0.05
- **Effect size**: Cohen's d

#### Test 2: One-way ANOVA
- **H0**: Performance is equal across all attack types
- **H1**: At least one attack type differs
- **Post-hoc**: Tukey HSD for pairwise comparisons

#### Test 3: Mann-Whitney U (non-parametric)
- Alternative to t-test when normality assumption violated
- Compares median performance

### 5.3 Bonferroni Correction

When performing multiple comparisons:
- Adjusted α = 0.05 / number_of_comparisons
- Example: 6 attack scenarios → α_adj = 0.0083

---

## 6. Experimental Configurations

### 6.1 Baseline Scenarios (No Attack)

| Config | Method | Malicious Nodes | Attack Type |
|--------|--------|----------------|-------------|
| B1     | FedAvg | None          | None        |
| B2     | FedXChain | None       | None        |

### 6.2 Attack Scenarios

| Config | Method | Malicious Nodes | Attack Type | Intensity |
|--------|--------|----------------|-------------|-----------|
| A1     | FedAvg | [0, 1] (20%)  | Label Flip  | 0.3       |
| A2     | FedXChain | [0, 1]     | Label Flip  | 0.3       |
| A3     | FedAvg | [0, 1, 2] (30%) | Label Flip | 0.5      |
| A4     | FedXChain | [0, 1, 2] | Label Flip  | 0.5       |
| A5     | FedAvg | [0, 1]        | Gaussian    | 0.3       |
| A6     | FedXChain | [0, 1]     | Gaussian    | 0.3       |
| A7     | FedAvg | [0, 1, 2]     | Gaussian    | 0.5       |
| A8     | FedXChain | [0, 1, 2]  | Gaussian    | 0.5       |
| A9     | FedAvg | [0, 1]        | Sign Flip   | 1.0       |
| A10    | FedXChain | [0, 1]     | Sign Flip   | 1.0       |

**Total experiments**: 10 configurations × 10 runs = **100 experiments**

### 6.3 Ablation Study

#### Study A: Trust Weight Analysis
| Config | α (Accuracy) | β (XAI) | γ (Consistency) |
|--------|-------------|---------|-----------------|
| AB1    | 1.0         | 0.0     | 0.0             |
| AB2    | 0.0         | 1.0     | 0.0             |
| AB3    | 0.0         | 0.0     | 1.0             |
| AB4    | 0.4         | 0.3     | 0.3 (default)   |

#### Study B: Malicious Node Ratio
| Config | Malicious Ratio | Nodes       |
|--------|----------------|-------------|
| AB5    | 10%            | [0]         |
| AB6    | 20%            | [0, 1]      |
| AB7    | 30%            | [0, 1, 2]   |
| AB8    | 40%            | [0, 1, 2, 3] |

---

## 7. Reproducibility Guidelines

### 7.1 Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Running Experiments

```bash
# Run full experimental suite
python scripts/run_ieee_experiment.py \
    --dataset synthetic \
    --n_clients 10 \
    --rounds 20 \
    --n_runs 10 \
    --output results_ieee

# Analyze results
python scripts/analyze_ieee_results.py \
    --input results_ieee/ieee_results_synthetic_20231225_120000.csv \
    --output analysis_ieee
```

### 7.3 Random Seed Management

```python
# Fixed seeds for reproducibility
SEEDS = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]

for run_id, seed in enumerate(SEEDS):
    np.random.seed(seed)
    # Run experiment
```

---

## 8. Expected Outcomes

### 8.1 Hypothesized Results

**H1**: FedXChain will achieve comparable accuracy to FedAvg in non-attack scenarios (difference < 2%)

**H2**: FedXChain will show significantly higher resilience to attacks (ASR reduction > 30%)

**H3**: SHAP computation will add 10-20% computational overhead

**H4**: Trust mechanism will identify malicious nodes (NSDS > 0.5 for attackers)

### 8.2 Publication-Ready Outputs

1. **Tables** (LaTeX format):
   - Performance comparison table
   - Statistical test results
   - Ablation study results
   - Attack success rate table

2. **Figures** (PDF/PNG, 300 DPI):
   - Accuracy vs rounds convergence plot
   - Attack scenario comparison bar chart
   - Trust score evolution heatmap
   - NSDS distribution box plot
   - Ablation study results

3. **Supplementary Materials**:
   - Complete experimental results (CSV)
   - Statistical analysis summary (JSON)
   - Source code (GitHub repository)

---

## 9. Validation Checklist

- [ ] Multiple runs (≥5) per configuration for statistical significance
- [ ] Baseline comparison included
- [ ] Multiple attack scenarios tested
- [ ] Statistical tests performed (t-test, ANOVA)
- [ ] Effect sizes reported (Cohen's d)
- [ ] Confidence intervals computed
- [ ] Random seeds documented
- [ ] Computational environment documented
- [ ] Code made available for reproducibility
- [ ] Results reported with mean ± std
- [ ] Limitations discussed
- [ ] Ethical considerations addressed

---

## 10. Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| **Phase 1** | Environment setup, code implementation | 1 week |
| **Phase 2** | Pilot experiments, parameter tuning | 3 days |
| **Phase 3** | Full experimental suite execution | 2-3 days |
| **Phase 4** | Statistical analysis, visualization | 2 days |
| **Phase 5** | Paper writing, table/figure generation | 1 week |

**Total estimated time**: ~2.5 weeks

---

## 11. Computational Resources

### 11.1 Hardware Requirements

- **CPU**: 4+ cores (for parallel runs)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: Optional (NVIDIA with CUDA for faster SHAP computation)
- **Storage**: 10GB for results and logs

### 11.2 Estimated Runtime

- **Single experiment**: ~2-5 minutes (20 rounds × 10 clients)
- **Full suite**: ~100 experiments × 3 minutes = **5-8 hours**
- **Parallelization**: Can reduce to 2-3 hours with 4 parallel workers

---

## 12. Quality Assurance

### 12.1 Sanity Checks

```python
# Check 1: Data partition validity
assert sum(len(client_data) for client in clients) == len(train_data)

# Check 2: Model convergence
assert global_accuracy[-1] > global_accuracy[0]  # Should improve

# Check 3: Trust scores validity
assert abs(sum(trust_scores) - 1.0) < 1e-6  # Should sum to 1

# Check 4: Attack application
if is_malicious:
    assert model_weights_after != model_weights_before
```

### 12.2 Logging and Monitoring

```python
# Log essential information
logger.info(f"Round {r}: Accuracy={acc:.4f}, NSDS={nsds:.4f}")
logger.warning(f"Node {i} has high NSDS={nsds:.4f} > threshold")
logger.error(f"Experiment failed: {error_message}")
```

---

## 13. References

1. McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.

2. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.

3. Fang, M., et al. (2020). "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning." USENIX Security.

4. IEEE Access Author Guidelines: https://ieeeaccess.ieee.org/

---

## 14. Contact and Support

**Primary Investigator**: [Your Name]  
**Institution**: [Your University]  
**Email**: [your.email@university.edu]  
**GitHub**: https://github.com/yourusername/fedxchain

For questions about methodology or implementation, please open an issue on GitHub or contact via email.

---

**Document Version**: 1.0  
**Last Updated**: December 25, 2025  
**Status**: Ready for Execution
