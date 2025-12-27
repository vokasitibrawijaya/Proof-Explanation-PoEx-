# FedXChain: Federated Explainable Blockchain

Implementation of the FedXChain framework as described in the ETASR paper "FedXChain: Federated Explainable Blockchain with Node-Specific Adaptive Trust".

## Overview

FedXChain addresses three key challenges in federated learning:
1. **Explainability**: Privacy-preserving SHAP aggregation for model interpretability
2. **Trust**: Adaptive trust-based aggregation using Node-Specific Divergence Scores (NSDS)
3. **Auditability**: Blockchain-verified audit trails for transparent model aggregation

## Features

- **Federated-SHAP Aggregation**: Privacy-preserving feature importance synthesis
- **Node-Specific Divergence Scores (NSDS)**: Quantify local explanation fidelity
- **Adaptive Trust-Based Aggregation**: Dynamic weighting based on accuracy, explainability, and consistency
- **Blockchain Integration**: Immutable logging of model updates and aggregation decisions
- **Non-IID Data Support**: Dirichlet-based label skew for realistic heterogeneous scenarios

## Directory Structure

```
fedXchain-etasr/
├── configs/
│   └── experiment_config.yaml    # Experiment configuration
├── scripts/
│   └── run_fedxchain.py         # Main experiment implementation
├── hardhat/                      # Blockchain smart contracts (optional)
│   ├── contracts/
│   │   └── FedXChain.sol        # FedXChain smart contract
│   ├── scripts/
│   │   └── deploy.js            # Deployment script
│   └── hardhat.config.js        # Hardhat configuration
├── results/                      # Experiment results (generated)
├── logs/                         # Logs (generated)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container definition
├── docker-compose.yml            # Docker Compose setup
├── setup.sh                      # Setup script
├── run_experiment.sh             # Run experiment (simple mode)
└── README.md                     # This file
```

## Quick Start

### Option 1: Simple Mode (Without Blockchain)

This is the fastest way to run the experiment:

```bash
# 1. Run setup
./setup.sh

# 2. Run experiment
./run_experiment.sh
```

### Option 2: Docker Mode (With Blockchain)

Run the complete setup with blockchain logging:

```bash
# Build and run with Docker Compose
docker-compose up
```

### Option 3: Local Blockchain Mode

Run with a local blockchain (requires Node.js and Hardhat):

```bash
# 1. Setup
./setup.sh

# 2. Install Hardhat dependencies
cd hardhat
npm install

# 3. Start local blockchain (in separate terminal)
npx hardhat node

# 4. Deploy contract (in another terminal)
npx hardhat run scripts/deploy.js --network localhost

# 5. Update config with contract address
# Edit configs/experiment_config.yaml:
#   use_blockchain: true
#   contract_address: "<address from deployment>"

# 6. Run experiment
cd ..
source venv/bin/activate
python scripts/run_fedxchain.py --config configs/experiment_config.yaml --output results
```

## Configuration

Edit `configs/experiment_config.yaml` to customize:

```yaml
# Dataset
n_samples: 1000
n_features: 20

# Federated learning
n_clients: 10
rounds: 10
local_epochs: 1

# Non-IID distribution
dirichlet_alpha: 0.5  # Lower = more non-IID

# Trust score weights
trust_alpha: 0.4      # Accuracy
trust_beta: 0.3       # XAI fidelity
trust_gamma: 0.3      # Consistency

# Blockchain (optional)
use_blockchain: false
blockchain_rpc: "http://localhost:8545"
```

## Experiment Details

Based on the paper specifications:

- **Dataset**: Synthetic classification (scikit-learn)
- **Clients**: 10 nodes with 100% participation
- **Rounds**: 10 federated rounds
- **Model**: Logistic Regression (SGDClassifier with log_loss)
- **Explainability**: SHAP (KernelExplainer)
- **Trust Computation**: 
  - Accuracy component (α = 0.4)
  - XAI fidelity via NSDS (β = 0.3)
  - Consistency across rounds (γ = 0.3)

## Results

After running the experiment, check the `results/` directory:

- `fedxchain_results.csv`: Round-by-round metrics
  - Global accuracy
  - Average local accuracy  
  - Average NSDS (Node-Specific Divergence Score)
  
- `trust_scores.json`: Trust scores for each node per round

## Key Metrics

1. **Global Accuracy**: Performance of aggregated model on all test data
2. **Avg Local Accuracy**: Average performance of individual nodes
3. **Avg NSDS**: Average divergence between local and global SHAP explanations (lower is better)
4. **Trust Scores**: Adaptive weights combining accuracy, fidelity, and consistency

## Dependencies

- Python 3.10+
- NumPy, Pandas, Scikit-learn
- SHAP (explainability)
- Web3.py (blockchain interface)
- PyYAML (configuration)

Optional (for blockchain):
- Node.js 18+
- Hardhat
- Foundry (for Docker mode)

## Paper Reference

```
Atmoko, R. A., Rohmatillah, M., Avian, C., Pramono, S. H., Purnomo, F. E., & Mudjirahardjo, P.
FedXChain: Federated Explainable Blockchain with Node-Specific Adaptive Trust.
Engineering, Technology & Applied Science Research.
```

## Comparison with Baselines

FedXChain demonstrates:
- **vs FedAvg**: Better balance between global performance and local interpretability
- **vs FedProx**: Improved explanation fidelity under non-IID conditions
- **Security**: Privacy-preserving SHAP aggregation via secure aggregation
- **Transparency**: Blockchain-verified audit trails for reproducibility

## Troubleshooting

**Issue**: Import errors
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue**: Blockchain connection failed
```bash
# Solution: Run in simple mode (blockchain disabled by default)
./run_experiment.sh
```

**Issue**: Out of memory
```bash
# Solution: Reduce n_samples or n_clients in config
# Edit configs/experiment_config.yaml
```

## License

This implementation is provided for research and educational purposes.

## Contact

For questions about the implementation, please refer to the original paper.
