#!/usr/bin/env python3
"""
Distributed FL Aggregator Server
Coordinates federated learning rounds across Docker containers
Integrates with blockchain for PoEx consensus
"""

import os
import json
import time
import numpy as np
from flask import Flask, request, jsonify
from web3 import Web3
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import pickle
import base64
from pathlib import Path
import csv

app = Flask(__name__)

class DistributedAggregator:
    def __init__(self):
        self.config = self._load_config()
        self.blockchain_rpc = os.getenv('BLOCKCHAIN_RPC', 'http://blockchain:8545')
        self.n_clients = int(os.getenv('N_CLIENTS', self.config.get('n_clients', 5)))
        self.agg_method = os.getenv('AGG_METHOD', self.config.get('agg_method', 'fedxchain')).strip().lower()
        self.run_id = os.getenv('RUN_ID', str(self.config.get('run_id', '1')))
        self.attack_type = os.getenv('ATTACK_TYPE', self.config.get('attack_type', 'none')).strip().lower()
        self.malicious_ratio = float(os.getenv('MALICIOUS_RATIO', self.config.get('malicious_ratio', 0.0)))
        self.malicious_clients = os.getenv('MALICIOUS_CLIENTS', self.config.get('malicious_clients', '')).strip()
        self.clear_results = os.getenv('CLEAR_RESULTS', str(self.config.get('clear_results', '0'))).strip().lower() in {'1', 'true', 'yes', 'y'}
        self.current_round = 0
        self.max_rounds = int(os.getenv('MAX_ROUNDS', self.config.get('rounds', 10)))
        self.clients_ready = {}
        self.client_updates = {}
        self.global_model = None
        self.X_test = None
        self.y_test = None

        self.output_dir = Path(os.getenv('OUTPUT_DIR', '/app/results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results_file = self.output_dir / 'fedxchain_results.csv'
        self._trust_file = self.output_dir / 'trust_scores.json'
        self._trust_scores_last = {}

        if self.clear_results and self._results_file.exists():
            try:
                self._results_file.unlink()
            except Exception as e:
                print(f"! Failed to clear results file: {e}")
        
        # Initialize blockchain connection
        self.init_blockchain()
        
        # Create dataset
        self.init_dataset()
        
        print(f"✓ Aggregator initialized")
        print(f"  - Blockchain: {self.blockchain_rpc}")
        print(f"  - Contract: {self.contract_address}")
        print(f"  - Clients: {self.n_clients}")
        print(f"  - Rounds: {self.max_rounds}")
        print(f"  - Method: {self.agg_method}")
        print(f"  - Run ID: {self.run_id}")
        print(f"  - Attack: {self.attack_type} | Malicious ratio: {self.malicious_ratio} | Malicious clients: {self.malicious_clients or '-'}")

    def _load_config(self) -> dict:
        config_path = os.getenv('CONFIG_PATH', '/app/configs/experiment_config.yaml')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"! Failed to load config {config_path}: {e}")
        return {}

    def _append_round_result(self, round_num: int, global_accuracy: float, avg_local_accuracy: float, avg_nsds: float) -> None:
        write_header = not self._results_file.exists()
        with open(self._results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'run_id',
                    'method',
                    'attack_type',
                    'malicious_ratio',
                    'malicious_clients',
                    'round',
                    'global_accuracy',
                    'avg_local_accuracy',
                    'avg_nsds'
                ])
            writer.writerow([
                self.run_id,
                self.agg_method,
                self.attack_type,
                float(self.malicious_ratio),
                self.malicious_clients,
                round_num,
                global_accuracy,
                avg_local_accuracy,
                avg_nsds
            ])

    def _save_trust_scores(self) -> None:
        try:
            with open(self._trust_file, 'w', encoding='utf-8') as f:
                json.dump(self._trust_scores_last, f, indent=2)
        except Exception as e:
            print(f"! Failed to save trust_scores.json: {e}")
    
    def init_blockchain(self):
        """Initialize blockchain connection"""
        print("Connecting to blockchain...")
        time.sleep(5)  # Wait for blockchain to be ready
        
        self.w3 = Web3(Web3.HTTPProvider(self.blockchain_rpc))
        
        # Wait for connection
        for i in range(30):
            if self.w3.is_connected():
                print("✓ Connected to blockchain")
                break
            time.sleep(1)
        else:
            raise Exception("Failed to connect to blockchain")
        
        # Load contract
        contract_file = '/app/deployments/contract_address.json'
        
        # Wait for contract deployment
        for i in range(60):
            if os.path.exists(contract_file):
                with open(contract_file, 'r') as f:
                    deployment = json.load(f)
                break
            time.sleep(1)
        else:
            raise Exception("Contract deployment file not found")
        
        self.contract_address = deployment['contractAddress']
        self.contract_abi = deployment['abi']
        
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        
        print(f"✓ Contract loaded: {self.contract_address}")
    
    def init_dataset(self):
        """Create and split dataset"""
        print("Creating dataset...")

        seed = int(os.getenv('SEED', self.config.get('seed', 42)))
        n_samples = int(os.getenv('N_SAMPLES', self.config.get('n_samples', 500)))
        n_features = int(os.getenv('N_FEATURES', self.config.get('n_features', 20)))
        test_size = float(os.getenv('TEST_SIZE', self.config.get('test_size', 0.2)))
        n_informative = int(os.getenv('N_INFORMATIVE', self.config.get('n_informative', max(2, int(n_features * 0.7)))))
        n_redundant = int(os.getenv('N_REDUNDANT', self.config.get('n_redundant', max(0, int(n_features * 0.2)))))
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=2,
            n_informative=n_informative,
            n_redundant=n_redundant,
            random_state=seed
        )
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        
        self.X_test = X_test
        self.y_test = y_test
        
        # Split training data for clients using Dirichlet
        self.client_data = self.split_data_non_iid(X_train, y_train, seed=seed)
        
        # Initialize global model
        self.global_model = SGDClassifier(loss='log_loss', max_iter=100, random_state=seed)
        self.global_model.fit(X_train[:100], y_train[:100])
        
        print(f"✓ Dataset created: {len(X_train)} train, {len(X_test)} test")
    
    def split_data_non_iid(self, X, y, seed: int = 42):
        """Split data into non-IID partitions"""
        rng = np.random.default_rng(seed)
        min_size = 0
        K = len(np.unique(y))
        N = len(y)
        alpha = float(os.getenv('DIRICHLET_ALPHA', self.config.get('dirichlet_alpha', 0.5)))
        min_required = int(os.getenv('MIN_SAMPLES_PER_CLIENT', self.config.get('min_samples_per_client', 10)))
        
        while min_size < min_required:
            idx_batch = [[] for _ in range(self.n_clients)]
            for k in range(K):
                idx_k = np.where(y == k)[0]
                rng.shuffle(idx_k)
                proportions = rng.dirichlet(np.repeat(alpha, self.n_clients))
                proportions = np.array([p * (len(idx_j) < N / self.n_clients) 
                                       for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() 
                           for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        
        client_data = {}
        for i, indices in enumerate(idx_batch):
            client_data[i] = {
                'X_train': X[indices],
                'y_train': y[indices]
            }
        
        return client_data
    
    def serialize_model(self, model):
        """Serialize model to base64"""
        model_bytes = pickle.dumps({
            'coef': model.coef_,
            'intercept': model.intercept_
        })
        return base64.b64encode(model_bytes).decode('utf-8')
    
    def deserialize_model(self, model_str):
        """Deserialize model from base64"""
        model_bytes = base64.b64decode(model_str.encode('utf-8'))
        return pickle.loads(model_bytes)
    
    def aggregate_updates(self):
        """Aggregate client updates with PoEx consensus"""
        print(f"\n[Round {self.current_round}] Aggregating {len(self.client_updates)} updates...")
        
        # Extract updates
        weights_list = []
        shap_values_list = []
        trust_scores = []
        accuracies = []

        for _client_id, update in self.client_updates.items():
            weights = self.deserialize_model(update['model'])
            weights_list.append(weights)
            if update.get('shap_values') is not None:
                shap_values_list.append(np.array(update['shap_values'], dtype=float))
            accuracies.append(float(update.get('accuracy', 0.0)))
            trust_scores.append(float(update.get('trust_score', 1.0)))

        # Compute NSDS (only if SHAP provided)
        nsds_list = [0.0 for _ in range(len(weights_list))]
        if len(shap_values_list) == len(weights_list) and len(shap_values_list) > 0:
            shap_mean = np.mean(shap_values_list, axis=0)
            nsds_list = []
            for shap_values in shap_values_list:
                cos_sim = np.dot(shap_values, shap_mean) / (
                    np.linalg.norm(shap_values) * np.linalg.norm(shap_mean) + 1e-10
                )
                nsds = 1 - cos_sim
                nsds_list.append(float(nsds))

        # Determine aggregation weights
        if self.agg_method == 'fedavg':
            new_trust_scores = (np.ones(len(weights_list), dtype=float) / max(1, len(weights_list))).tolist()
        else:
            new_trust_scores = self.update_trust_scores(accuracies, nsds_list, trust_scores)

        # Weighted aggregation
        global_coef = np.zeros_like(weights_list[0]['coef'])
        global_intercept = np.zeros_like(weights_list[0]['intercept'])

        for weights, weight in zip(weights_list, new_trust_scores):
            global_coef += weight * weights['coef']
            global_intercept += weight * weights['intercept']
        
        # Update global model
        self.global_model.coef_ = global_coef
        self.global_model.intercept_ = global_intercept
        
        # Evaluate
        y_pred = self.global_model.predict(self.X_test)
        global_accuracy = np.mean(y_pred == self.y_test)

        avg_local_accuracy = float(np.mean(accuracies))
        
        print(f"  ✓ Global Accuracy: {global_accuracy:.4f}")
        avg_nsds = float(np.mean(nsds_list)) if len(nsds_list) else 0.0
        print(f"  ✓ Avg NSDS: {avg_nsds:.4f}")
        print(f"  ✓ Trust range: [{min(new_trust_scores):.3f}, {max(new_trust_scores):.3f}]")

        # Persist results
        try:
            self._append_round_result(self.current_round + 1, float(global_accuracy), avg_local_accuracy, avg_nsds)
            self._trust_scores_last = {str(i): float(t) for i, t in enumerate(new_trust_scores)}
            self._save_trust_scores()
        except Exception as e:
            print(f"! Failed to persist round results: {e}")
        
        return new_trust_scores
    
    def update_trust_scores(self, accuracies, nsds_list, prev_trust):
        """Update trust scores using ETASR formula"""
        alpha, beta, gamma = 0.4, 0.3, 0.3
        
        acc_norm = np.array(accuracies)
        nsds_norm = 1 - np.array(nsds_list)
        prev_trust = np.array(prev_trust)
        
        trust_scores = alpha * acc_norm + beta * nsds_norm + gamma * prev_trust
        trust_scores = trust_scores / trust_scores.sum()
        
        return trust_scores.tolist()

# Global aggregator instance
aggregator = DistributedAggregator()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'round': aggregator.current_round})

@app.route('/register', methods=['POST'])
def register_client():
    """Register a new client"""
    data = request.json
    client_id = data['client_id']
    
    aggregator.clients_ready[client_id] = False
    
    # Send dataset partition to client
    client_data = aggregator.client_data[client_id]
    
    return jsonify({
        'status': 'registered',
        'client_id': client_id,
        'n_samples': len(client_data['X_train']),
        'current_round': aggregator.current_round
    })

@app.route('/get_data/<int:client_id>', methods=['GET'])
def get_client_data(client_id):
    """Get training data for client"""
    client_data = aggregator.client_data[client_id]
    
    return jsonify({
        'X_train': client_data['X_train'].tolist(),
        'y_train': client_data['y_train'].tolist(),
        'X_test': aggregator.X_test.tolist(),
        'y_test': aggregator.y_test.tolist()
    })

@app.route('/get_model', methods=['GET'])
def get_global_model():
    """Get current global model"""
    model_str = aggregator.serialize_model(aggregator.global_model)
    
    return jsonify({
        'model': model_str,
        'round': aggregator.current_round,
        'continue': aggregator.current_round < aggregator.max_rounds
    })

@app.route('/submit_update', methods=['POST'])
def submit_update():
    """Receive update from client"""
    data = request.json
    client_id = data['client_id']

    # If rounds finished, don't accept new work (prevents clients looping forever)
    if aggregator.current_round >= aggregator.max_rounds:
        return jsonify({
            'status': 'done',
            'round': aggregator.current_round,
            'continue': False
        })
    
    aggregator.client_updates[client_id] = {
        'model': data['model'],
        'shap_values': data['shap_values'],
        'accuracy': data['accuracy'],
        'trust_score': data.get('trust_score', 1.0)
    }
    
    print(f"  ✓ Received update from client {client_id}")
    
    # Check if all clients submitted
    if len(aggregator.client_updates) == aggregator.n_clients:
        # Aggregate
        new_trust_scores = aggregator.aggregate_updates()
        
        # Clear updates
        aggregator.client_updates = {}
        
        # Increment round
        aggregator.current_round += 1
        
        return jsonify({
            'status': 'aggregated',
            'trust_score': new_trust_scores[client_id],
            'round': aggregator.current_round,
            'continue': aggregator.current_round < aggregator.max_rounds
        })
    else:
        return jsonify({
            'status': 'waiting',
            'waiting_for': aggregator.n_clients - len(aggregator.client_updates),
            'round': aggregator.current_round,
            'continue': aggregator.current_round < aggregator.max_rounds
        })

@app.route('/status', methods=['GET'])
def get_status():
    """Get aggregator status"""
    return jsonify({
        'current_round': aggregator.current_round,
        'max_rounds': aggregator.max_rounds,
        'n_clients': aggregator.n_clients,
        'updates_received': len(aggregator.client_updates),
        'blockchain_connected': aggregator.w3.is_connected()
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FedXChain Distributed Aggregator Server")
    print("="*70)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
