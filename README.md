# FedXChain: Proof of Explanation (PoEx) Consensus for Byzantine-Robust Federated Learning

Implementation of the FedXChain framework as described in the IEEE Access paper:

> **"FedXChain: Proof of Explanation (PoEx) Consensus for Byzantine-Robust Federated Learning Using SHAP-Based Model Validation on Blockchain"**
>
> Rachmad Andri Atmoko, Sholeh Hadi Pramono, M. Fauzan Edy Purnomo, Panca Mudjirahardjo, Mahdin Rohmatillah, and Cries Avian
>
> IEEE Access, 2025

---

## Reproducibility Statement

This repository provides all necessary code, configurations, and documentation to reproduce the experimental results presented in the paper. Key reproducibility features:

- **Fixed random seeds** for deterministic results
- **Docker containerization** for consistent environment
- **Comprehensive configuration files** documenting all hyperparameters
- **Automated experiment scripts** with logging

**GitHub Repository**: [https://github.com/vokasitibrawijaya/Proof-Explanation-PoEx-](https://github.com/vokasitibrawijaya/Proof-Explanation-PoEx-)

---

## Overview

FedXChain addresses three key challenges in federated learning:
1. **Byzantine-Robust Aggregation**: Defense against malicious clients using SHAP-based validation
2. **Explainability**: NSDS (Normalized Symmetric Divergence Score) for interpretable anomaly detection
3. **Auditability**: Hyperledger Fabric blockchain for immutable audit trails

## Key Contributions

- **Proof of Explanation (PoEx)**: Novel consensus mechanism using SHAP values
- **NSDS Metric**: Quantifies explanation divergence using Jensen-Shannon divergence
- **Comprehensive Evaluation**: 8 aggregation methods, 4 attack types, IID/Non-IID scenarios

## Directory Structure

```
consensus_flblockchain/
├── configs/                      # Experiment configurations
│   ├── ieee_experiment_config.yaml   # Main experiment config
│   └── experiment_config.yaml        # Basic config
├── scripts/                      # Python experiment scripts
│   ├── run_ieee_experiment.py        # Main IEEE experiment
│   ├── run_cifar10_cnn_experiment.py # CIFAR-10 CNN experiment
│   └── visualize_results.py          # Result visualization
├── hardhat/                      # Ethereum smart contracts
│   ├── contracts/FedXChain.sol       # PoEx smart contract
│   └── scripts/deploy.js             # Deployment script
├── hlf/                          # Hyperledger Fabric chaincode
│   └── chaincode/poex/               # PoEx chaincode (Go)
├── results/                      # Experiment results (generated)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container
├── docker-compose.yml            # Multi-container setup
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

## Citation

If you use this code in your research, please cite:

```bibtex
@article{atmoko2025fedxchain,
  title={FedXChain: Proof of Explanation (PoEx) Consensus for Byzantine-Robust Federated Learning Using SHAP-Based Model Validation on Blockchain},
  author={Atmoko, Rachmad Andri and Pramono, Sholeh Hadi and Purnomo, M. Fauzan Edy and Mudjirahardjo, Panca and Rohmatillah, Mahdin and Avian, Cries},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE},
  doi={10.1109/ACCESS.2025.XXXXXXX}
}
```

## License

This implementation is provided for research and educational purposes under the MIT License.

## Contact

- **Corresponding Author**: Sholeh Hadi Pramono (sholehpramono@ub.ac.id)
- **Institution**: Electrical Engineering Department, Universitas Brawijaya, Malang, Indonesia
- **GitHub**: [https://github.com/vokasitibrawijaya/Proof-Explanation-PoEx-](https://github.com/vokasitibrawijaya/Proof-Explanation-PoEx-)
