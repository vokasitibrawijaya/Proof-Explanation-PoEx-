#!/bin/bash
# Run enhanced experiments for reviewer response
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════════════════════"
echo "  FedXChain Enhanced Experiments - Addressing Reviewer Comments"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

source venv/bin/activate

# Create output directory
mkdir -p results_enhanced

echo "Running experiments with multiple models and datasets..."
echo "Each configuration will run 5 times for statistical robustness"
echo ""

# Experiment 1: Logistic Regression + Synthetic
echo "═══════════════════════════════════════════════════════════"
echo "Experiment 1: Logistic Regression + Synthetic Dataset"
echo "═══════════════════════════════════════════════════════════"
python scripts/run_enhanced_experiment.py \
    --config <(echo "dataset: synthetic
model_type: logistic
n_samples: 1000
n_features: 20
n_clients: 10
rounds: 10
local_epochs: 1
shap_samples: 10
trust_alpha: 0.4
trust_beta: 0.3
trust_gamma: 0.3") \
    --output results_enhanced \
    --runs 5

# Experiment 2: Logistic Regression + Breast Cancer
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Experiment 2: Logistic Regression + Breast Cancer Dataset"
echo "═══════════════════════════════════════════════════════════"
python scripts/run_enhanced_experiment.py \
    --config <(echo "dataset: breast_cancer
model_type: logistic
n_clients: 10
rounds: 10
local_epochs: 1
shap_samples: 10
trust_alpha: 0.4
trust_beta: 0.3
trust_gamma: 0.3") \
    --output results_enhanced \
    --runs 5

# Experiment 3: MLP + Breast Cancer
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Experiment 3: Multi-Layer Perceptron + Breast Cancer"
echo "═══════════════════════════════════════════════════════════"
python scripts/run_enhanced_experiment.py \
    --config <(echo "dataset: breast_cancer
model_type: mlp
n_clients: 10
rounds: 10
local_epochs: 1
shap_samples: 10
trust_alpha: 0.4
trust_beta: 0.3
trust_gamma: 0.3") \
    --output results_enhanced \
    --runs 5

# Experiment 4: Random Forest + Breast Cancer
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Experiment 4: Random Forest + Breast Cancer"
echo "═══════════════════════════════════════════════════════════"
python scripts/run_enhanced_experiment.py \
    --config <(echo "dataset: breast_cancer
model_type: rf
n_clients: 10
rounds: 10
local_epochs: 1
shap_samples: 10
trust_alpha: 0.4
trust_beta: 0.3
trust_gamma: 0.3") \
    --output results_enhanced \
    --runs 5

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  All Enhanced Experiments Complete!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved in: results_enhanced/"
echo ""
echo "Generated files:"
ls -lh results_enhanced/stats_*.csv 2>/dev/null || echo "  (files will be generated during run)"
echo ""
