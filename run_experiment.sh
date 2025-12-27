#!/bin/bash
# Run FedXChain experiment without blockchain
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════════════════════"
echo "  FedXChain: Federated Explainable Blockchain Experiment"
echo "  Mode: Simple (without blockchain)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import numpy, pandas, sklearn, shap" 2>/dev/null; then
    echo "❌ Dependencies not installed. Run ./setup.sh first"
    exit 1
fi

# Create output directories
mkdir -p results logs

# Run experiment
echo "Starting experiment..."
echo ""

python scripts/run_fedxchain.py \
    --config configs/experiment_config.yaml \
    --output results

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  Experiment Complete!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: results/"
echo "  - fedxchain_results.csv: Round-by-round metrics"
echo "  - trust_scores.json: Trust scores for each node per round"
echo ""
