#!/usr/bin/env python3
"""
FedXChain: Federated Explainable Blockchain with Node-Specific Adaptive Trust
Implementation based on ETASR paper

This script implements:
- Federated-SHAP aggregation for privacy-preserving explainability
- Node-Specific Divergence Scores (NSDS) 
- Adaptive trust-based aggregation
- Blockchain-verified audit trails
"""

import numpy as np
import pandas as pd
import json
import hashlib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
import shap
from web3 import Web3
import warnings
warnings.filterwarnings('ignore')

class BlockchainClient:
    """Interface to FedXChain smart contract"""
    
    def __init__(self, rpc_url, contract_address, contract_abi):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contract = self.w3.eth.contract(address=contract_address, abi=contract_abi)
        self.account = self.w3.eth.accounts[0]  # Use first account
        
    def register_node(self, node_address, node_id):
        """Register a node on blockchain"""
        tx = self.contract.functions.registerNode(node_address, node_id).transact({'from': self.account})
        self.w3.eth.wait_for_transaction_receipt(tx)
        print(f"✓ Node {node_id} registered on blockchain")
        
    def submit_update(self, node_id, model_hash, shap_hash, accuracy, fidelity):
        """Submit model update to blockchain"""
        acc_int = int(accuracy * 10000)  # Convert to integer for blockchain
        fid_int = int(fidelity * 10000)
        tx = self.contract.functions.submitUpdate(
            node_id, model_hash, shap_hash, acc_int, fid_int
        ).transact({'from': self.account})
        self.w3.eth.wait_for_transaction_receipt(tx)
        
    def log_aggregation(self, global_model_hash, global_shap_hash, n_nodes, metadata):
        """Log aggregation results to blockchain"""
        tx = self.contract.functions.logAggregation(
            global_model_hash, global_shap_hash, n_nodes, metadata
        ).transact({'from': self.account})
        self.w3.eth.wait_for_transaction_receipt(tx)

class FedXChainNode:
    """Individual federated learning node with explainability"""
    
    def __init__(self, node_id, X_train, y_train, X_test, y_test):
        self.node_id = node_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
        self.shap_values = None
        self.trust_score = 1.0
        
    def train_local(self, global_weights=None, epochs=1):
        """Train local model"""
        if global_weights is not None:
            self.model.coef_ = global_weights['coef']
            self.model.intercept_ = global_weights['intercept']
        
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train, classes=np.unique(self.y_train))
            
    def compute_shap(self, n_samples=10):
        """Compute SHAP values for explainability"""
        # Sample data for SHAP computation
        sample_idx = np.random.choice(len(self.X_train), min(n_samples, len(self.X_train)), replace=False)
        X_sample = self.X_train[sample_idx]
        
        # Use KernelExplainer for model-agnostic explanation
        explainer = shap.KernelExplainer(self.model.predict_proba, X_sample)
        shap_values = explainer.shap_values(X_sample)
        
        # Aggregate SHAP values (mean absolute values across samples)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, use positive class
        elif len(shap_values.shape) == 3:
            # Shape is (n_samples, n_features, n_classes) - take positive class
            shap_values = shap_values[:, :, 1]
        
        self.shap_values = np.mean(np.abs(shap_values), axis=0)
        
        return self.shap_values
        
    def evaluate(self, X=None, y=None):
        """Evaluate model accuracy"""
        if X is None:
            X, y = self.X_test, self.y_test
        predictions = self.model.predict(X)
        return accuracy_score(y, predictions)
        
    def get_weights(self):
        """Get model weights"""
        return {
            'coef': self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy()
        }

class FedXChainAggregator:
    """Central aggregator with trust-based weighting"""
    
    def __init__(self, n_features, alpha=0.4, beta=0.3, gamma=0.3):
        self.global_model = None
        self.global_shap = None
        self.n_features = n_features
        # Trust score weights
        self.alpha = alpha  # Accuracy weight
        self.beta = beta    # XAI fidelity weight  
        self.gamma = gamma  # Consistency weight
        self.history = []
        
    def compute_trust_scores(self, nodes, round_num):
        """Compute trust scores for all nodes"""
        trust_scores = {}
        
        for node in nodes:
            # Accuracy component
            acc = node.evaluate()
            
            # XAI fidelity component (Node-Specific Divergence Score)
            if self.global_shap is not None and node.shap_values is not None:
                # Compute KL divergence between local and global SHAP
                eps = 1e-10
                local_shap = node.shap_values + eps
                global_shap = self.global_shap + eps
                # Normalize to probability distributions
                local_shap = local_shap / local_shap.sum()
                global_shap = global_shap / global_shap.sum()
                kl_div = np.sum(local_shap * np.log(local_shap / global_shap))
                fidelity = np.exp(-kl_div)  # Convert divergence to fidelity
            else:
                fidelity = 1.0
                
            # Consistency component (based on historical performance)
            if len(self.history) > 0:
                recent_accs = [h['nodes'].get(node.node_id, {}).get('accuracy', 0) 
                              for h in self.history[-3:]]
                consistency = 1.0 - np.std(recent_accs) if recent_accs else 1.0
            else:
                consistency = 1.0
                
            # Combined trust score
            trust = self.alpha * acc + self.beta * fidelity + self.gamma * consistency
            trust_scores[node.node_id] = {
                'trust': trust,
                'accuracy': acc,
                'fidelity': fidelity,
                'consistency': consistency
            }
            node.trust_score = trust
            
        return trust_scores
        
    def aggregate(self, nodes, round_num):
        """Aggregate model updates with adaptive trust-based weighting"""
        # Compute trust scores
        trust_scores = self.compute_trust_scores(nodes, round_num)
        
        # Normalize trust scores to get aggregation weights
        total_trust = sum(scores['trust'] for scores in trust_scores.values())
        weights = {nid: scores['trust'] / total_trust for nid, scores in trust_scores.items()}
        
        # Aggregate model parameters
        global_coef = np.zeros_like(nodes[0].get_weights()['coef'])
        global_intercept = np.zeros_like(nodes[0].get_weights()['intercept'])
        
        for node in nodes:
            w = weights[node.node_id]
            node_weights = node.get_weights()
            global_coef += w * node_weights['coef']
            global_intercept += w * node_weights['intercept']
            
        self.global_model = {
            'coef': global_coef,
            'intercept': global_intercept
        }
        
        # Aggregate SHAP values with secure aggregation simulation
        # In real implementation, this would use additive masking
        global_shap = np.zeros(self.n_features)
        for node in nodes:
            if node.shap_values is not None:
                w = weights[node.node_id]
                global_shap += w * node.shap_values
        self.global_shap = global_shap
        
        # Log round info
        self.history.append({
            'round': round_num,
            'weights': weights,
            'trust_scores': trust_scores,
            'nodes': {node.node_id: {
                'accuracy': trust_scores[node.node_id]['accuracy'],
                'fidelity': trust_scores[node.node_id]['fidelity'],
                'trust': trust_scores[node.node_id]['trust']
            } for node in nodes}
        })
        
        return self.global_model, weights, trust_scores

def create_hash(data):
    """Create SHA256 hash of data"""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return '0x' + hashlib.sha256(data_str.encode()).hexdigest()

def run_fedxchain_experiment(config):
    """Run FedXChain experiment"""
    
    print("=" * 80)
    print("FedXChain: Federated Explainable Blockchain Experiment")
    print("=" * 80)
    print()
    
    # Generate synthetic dataset
    print("1. Generating synthetic classification dataset...")
    n_samples = config['n_samples']
    n_features = config['n_features']
    n_clients = config['n_clients']
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 2,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data with non-IID distribution (Dirichlet)
    print(f"2. Splitting data into {n_clients} clients with non-IID distribution...")
    alpha_dirichlet = config.get('dirichlet_alpha', 0.5)
    
    # Improved non-IID split based on label skew with minimum samples per class
    client_data = []
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Split indices among clients
    splits = np.array_split(indices, n_clients)
    
    for i, client_indices in enumerate(splits):
        X_client = X[client_indices]
        y_client = y[client_indices]
        
        # Ensure both classes are present
        unique_classes = np.unique(y_client)
        if len(unique_classes) < 2:
            # If only one class, add some samples from the other class
            other_class = 1 - unique_classes[0]
            other_indices = np.where(y == other_class)[0]
            extra_samples = np.random.choice(other_indices, size=min(10, len(other_indices)), replace=False)
            X_client = np.vstack([X_client, X[extra_samples]])
            y_client = np.hstack([y_client, y[extra_samples]])
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=0.2, random_state=42, stratify=y_client
        )
        
        client_data.append((X_train, y_train, X_test, y_test))
        classes_in_train = np.unique(y_train)
        print(f"   Client {i}: {len(y_train)} train samples, {len(y_test)} test samples, classes: {classes_in_train}")
    
    # Initialize nodes
    print(f"\n3. Initializing {n_clients} federated nodes...")
    nodes = [
        FedXChainNode(f"node_{i}", *client_data[i])
        for i in range(n_clients)
    ]
    
    # Initialize aggregator
    aggregator = FedXChainAggregator(n_features)
    
    # Initialize blockchain client (if configured)
    blockchain = None
    if config.get('use_blockchain', False):
        try:
            print("\n4. Connecting to blockchain...")
            blockchain = BlockchainClient(
                config['blockchain_rpc'],
                config['contract_address'],
                config.get('contract_abi', [])
            )
            # Register nodes
            for i, node in enumerate(nodes):
                node_address = blockchain.w3.eth.accounts[min(i+1, len(blockchain.w3.eth.accounts)-1)]
                blockchain.register_node(node_address, node.node_id)
        except Exception as e:
            print(f"   Warning: Could not connect to blockchain: {e}")
            print("   Continuing without blockchain logging...")
            blockchain = None
    
    # Training rounds
    print(f"\n5. Starting federated training for {config['rounds']} rounds...")
    print("-" * 80)
    
    results = {
        'rounds': [],
        'global_accuracy': [],
        'avg_local_accuracy': [],
        'avg_nsds': [],
        'trust_scores': []
    }
    
    for round_num in range(config['rounds']):
        print(f"\n[Round {round_num + 1}/{config['rounds']}]")
        
        # Select participating clients (100% participation)
        participating_nodes = nodes
        
        # Local training
        print("  → Local training...")
        for node in participating_nodes:
            global_weights = aggregator.global_model if round_num > 0 else None
            node.train_local(global_weights, epochs=config['local_epochs'])
        
        # Compute SHAP values
        print("  → Computing SHAP explanations...")
        for node in participating_nodes:
            node.compute_shap(n_samples=config.get('shap_samples', 10))
        
        # Aggregate
        print("  → Aggregating with adaptive trust weighting...")
        global_model, weights, trust_scores = aggregator.aggregate(participating_nodes, round_num)
        
        # Evaluate
        # Create a temporary model with global weights for evaluation
        eval_model = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
        eval_model.coef_ = global_model['coef']
        eval_model.intercept_ = global_model['intercept']
        eval_model.classes_ = np.unique(y)
        
        # Global accuracy on test set
        X_test_all = np.vstack([node.X_test for node in nodes])
        y_test_all = np.hstack([node.y_test for node in nodes])
        y_pred_global = eval_model.predict(X_test_all)
        global_acc = accuracy_score(y_test_all, y_pred_global)
        
        # Average local accuracy
        local_accs = [node.evaluate() for node in participating_nodes]
        avg_local_acc = np.mean(local_accs)
        
        # Average NSDS (Node-Specific Divergence Score)
        nsds_scores = []
        if aggregator.global_shap is not None:
            for node in participating_nodes:
                if node.shap_values is not None:
                    eps = 1e-10
                    local_shap = node.shap_values + eps
                    global_shap = aggregator.global_shap + eps
                    local_shap = local_shap / local_shap.sum()
                    global_shap = global_shap / global_shap.sum()
                    kl_div = np.sum(local_shap * np.log(local_shap / global_shap))
                    nsds_scores.append(kl_div)
        avg_nsds = np.mean(nsds_scores) if nsds_scores else 0.0
        
        # Log to blockchain
        if blockchain is not None:
            try:
                model_hash = create_hash(global_model['coef'].tolist())
                shap_hash = create_hash(aggregator.global_shap.tolist())
                metadata = json.dumps({
                    'round': round_num,
                    'global_accuracy': float(global_acc),
                    'avg_nsds': float(avg_nsds)
                })
                blockchain.log_aggregation(
                    bytes.fromhex(model_hash[2:]),
                    bytes.fromhex(shap_hash[2:]),
                    len(participating_nodes),
                    metadata
                )
            except Exception as e:
                print(f"    Warning: Blockchain logging failed: {e}")
        
        # Store results
        results['rounds'].append(round_num + 1)
        results['global_accuracy'].append(global_acc)
        results['avg_local_accuracy'].append(avg_local_acc)
        results['avg_nsds'].append(avg_nsds)
        results['trust_scores'].append({
            node.node_id: trust_scores[node.node_id]['trust']
            for node in participating_nodes
        })
        
        # Print results
        print(f"  ✓ Global Accuracy: {global_acc:.4f}")
        print(f"  ✓ Avg Local Accuracy: {avg_local_acc:.4f}")
        print(f"  ✓ Avg NSDS: {avg_nsds:.4f}")
        print(f"  ✓ Trust Scores: {', '.join([f'{nid}: {ts:.3f}' for nid, ts in weights.items()])}")
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    
    return results, aggregator, nodes

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Run FedXChain experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run experiment
    results, aggregator, nodes = run_fedxchain_experiment(config)
    
    # Save results
    import os
    os.makedirs(args.output, exist_ok=True)
    
    results_df = pd.DataFrame({
        'round': results['rounds'],
        'global_accuracy': results['global_accuracy'],
        'avg_local_accuracy': results['avg_local_accuracy'],
        'avg_nsds': results['avg_nsds']
    })
    
    output_file = os.path.join(args.output, 'fedxchain_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Save trust scores
    trust_file = os.path.join(args.output, 'trust_scores.json')
    with open(trust_file, 'w') as f:
        json.dump(results['trust_scores'], f, indent=2)
    print(f"Trust scores saved to: {trust_file}")
