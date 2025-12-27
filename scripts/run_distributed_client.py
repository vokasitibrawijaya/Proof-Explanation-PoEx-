#!/usr/bin/env python3
"""
Distributed FL Client
Runs in Docker container, communicates with aggregator
"""

import os
import time
import numpy as np
import requests
from sklearn.linear_model import SGDClassifier
import shap
import pickle
import base64
import warnings
warnings.filterwarnings('ignore')

class DistributedClient:
    def __init__(self):
        self.node_id = int(os.getenv('NODE_ID', 0))
        self.aggregator_url = os.getenv('AGGREGATOR_URL', 'http://aggregator:5000')
        self.blockchain_rpc = os.getenv('BLOCKCHAIN_RPC', 'http://blockchain:8545')
        self.local_epochs = int(os.getenv('LOCAL_EPOCHS', 1))
        self.shap_samples = int(os.getenv('SHAP_SAMPLES', 10))
        self.seed = int(os.getenv('SEED', 42))

        self.attack_type = os.getenv('ATTACK_TYPE', 'none').strip().lower()
        self.attack_sigma = float(os.getenv('ATTACK_SIGMA', 0.1))
        self.malicious_ratio = float(os.getenv('MALICIOUS_RATIO', 0.0))
        self.malicious_clients = self._parse_malicious_clients(os.getenv('MALICIOUS_CLIENTS', ''))
        self.is_malicious = self._determine_is_malicious(os.getenv('MALICIOUS', '0'))
        
        np.random.seed(self.seed + self.node_id)
        self.model = SGDClassifier(loss='log_loss', max_iter=100, random_state=self.seed)
        self.trust_score = 1.0
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        print(f"\n{'='*70}")
        print(f"FL Client {self.node_id}")
        print(f"{'='*70}")
        print(f"  Aggregator: {self.aggregator_url}")
        print(f"  Blockchain: {self.blockchain_rpc}")
        print(f"  Malicious: {self.is_malicious} | Attack: {self.attack_type}")

    def _parse_malicious_clients(self, value: str):
        try:
            parts = [p.strip() for p in value.split(',') if p.strip()]
            return {int(p) for p in parts}
        except Exception:
            return set()

    def _determine_is_malicious(self, malicious_flag: str) -> bool:
        if str(malicious_flag).strip() in {'1', 'true', 'yes', 'y'}:
            return True
        if self.node_id in self.malicious_clients:
            return True
        if self.malicious_ratio > 0.0:
            return float(np.random.RandomState(self.seed + self.node_id).rand()) < self.malicious_ratio
        return False
    
    def wait_for_aggregator(self):
        """Wait for aggregator to be ready"""
        print("\nWaiting for aggregator...")
        
        for i in range(60):
            try:
                response = requests.get(f'{self.aggregator_url}/health', timeout=5)
                if response.status_code == 200:
                    print("✓ Aggregator is ready")
                    return True
            except:
                pass
            time.sleep(2)
        
        raise Exception("Aggregator not available")
    
    def register(self):
        """Register with aggregator"""
        print("\nRegistering with aggregator...")
        
        response = requests.post(
            f'{self.aggregator_url}/register',
            json={'client_id': self.node_id}
        )
        
        data = response.json()
        print(f"✓ Registered: {data['n_samples']} training samples")
        
        return data
    
    def get_training_data(self):
        """Get training data from aggregator"""
        print("\nFetching training data...")
        
        response = requests.get(f'{self.aggregator_url}/get_data/{self.node_id}')
        data = response.json()
        
        self.X_train = np.array(data['X_train'])
        self.y_train = np.array(data['y_train'])
        self.X_test = np.array(data['X_test'])
        self.y_test = np.array(data['y_test'])

        if self.is_malicious and self.attack_type == 'label_flip':
            self.y_train = 1 - self.y_train
            print("! Applied label_flip attack (malicious client)")
        
        print(f"✓ Received: {len(self.X_train)} train, {len(self.X_test)} test samples")
    
    def get_global_model(self):
        """Get global model from aggregator"""
        response = requests.get(f'{self.aggregator_url}/get_model')
        data = response.json()

        if not data.get('continue', True):
            return data['round'], False
        
        # Deserialize model
        model_bytes = base64.b64decode(data['model'].encode('utf-8'))
        weights = pickle.loads(model_bytes)
        
        self.model.coef_ = weights['coef']
        self.model.intercept_ = weights['intercept']

        return data['round'], True
    
    def train_local(self, epochs=1):
        """Train local model"""
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train, classes=np.array([0, 1]))
    
    def compute_shap(self, n_samples=10):
        """Compute SHAP values - FAST version using LinearExplainer"""
        sample_idx = np.random.choice(
            len(self.X_train), 
            min(n_samples, len(self.X_train)), 
            replace=False
        )
        X_sample = self.X_train[sample_idx]
        
        # Use LinearExplainer for SGDClassifier - 1000x faster than KernelExplainer!
        explainer = shap.LinearExplainer(self.model, X_sample, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        return np.mean(np.abs(shap_values), axis=0)
    
    def evaluate(self):
        """Evaluate model"""
        y_pred = self.model.predict(self.X_test)
        accuracy = np.mean(y_pred == self.y_test)
        return accuracy
    
    def serialize_model(self):
        """Serialize model to base64"""
        model_bytes = pickle.dumps({
            'coef': self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy()
        })
        return base64.b64encode(model_bytes).decode('utf-8')

    def _apply_weight_attack_to_model_str(self, model_str: str) -> str:
        if not self.is_malicious or self.attack_type in {'none', ''}:
            return model_str
        if self.attack_type not in {'gaussian_noise', 'sign_flip'}:
            return model_str

        model_bytes = base64.b64decode(model_str.encode('utf-8'))
        weights = pickle.loads(model_bytes)
        coef = np.array(weights['coef'], dtype=float)
        intercept = np.array(weights['intercept'], dtype=float)

        if self.attack_type == 'sign_flip':
            coef = -coef
            intercept = -intercept
        elif self.attack_type == 'gaussian_noise':
            coef = coef + np.random.normal(0.0, self.attack_sigma, size=coef.shape)
            intercept = intercept + np.random.normal(0.0, self.attack_sigma, size=intercept.shape)

        attacked = pickle.dumps({'coef': coef, 'intercept': intercept})
        return base64.b64encode(attacked).decode('utf-8')
    
    def submit_update(self, model_str, shap_values, accuracy):
        """Submit update to aggregator"""
        response = requests.post(
            f'{self.aggregator_url}/submit_update',
            json={
                'client_id': self.node_id,
                'model': model_str,
                'shap_values': shap_values.tolist(),
                'accuracy': float(accuracy),
                'trust_score': self.trust_score
            }
        )
        
        return response.json()
    
    def run_training(self):
        """Main training loop"""
        # Wait and register
        self.wait_for_aggregator()
        time.sleep(self.node_id * 5)  # Stagger registrations (5s to avoid MVCC conflicts)
        
        self.register()
        self.get_training_data()
        
        last_round_submitted = -1
        while True:
            current_round, can_continue = self.get_global_model()
            if not can_continue:
                print("\n" + "="*70)
                print("Training complete! (aggregator stopped)")
                print("="*70)
                break

            if current_round == last_round_submitted:
                time.sleep(2)
                continue

            print(f"\n{'='*70}")
            print(f"[Round {current_round + 1}] Client {self.node_id}")
            print(f"{'='*70}")

            # Local training
            print("  → Training locally...")
            self.train_local(epochs=self.local_epochs)

            # Compute SHAP
            print("  → Computing SHAP values...")
            shap_values = self.compute_shap(n_samples=self.shap_samples)

            # Evaluate
            accuracy = self.evaluate()
            print(f"  → Local accuracy: {accuracy:.4f}")

            # Serialize model (+ optional attack)
            model_str = self.serialize_model()
            model_str = self._apply_weight_attack_to_model_str(model_str)
            if self.is_malicious and self.attack_type in {'gaussian_noise', 'sign_flip'}:
                print(f"! Applied {self.attack_type} attack (malicious client)")

            # Submit update
            print("  → Submitting update...")
            result = self.submit_update(model_str, shap_values, accuracy)
            last_round_submitted = current_round

            if result.get('status') == 'aggregated':
                if 'trust_score' in result and result['trust_score'] is not None:
                    self.trust_score = result['trust_score']
                trust_str = f"{self.trust_score:.4f}" if self.trust_score is not None else "N/A"
                print(f"  ✓ Update aggregated | Trust: {trust_str}")
                if not result.get('continue', True):
                    print("\n" + "="*70)
                    print("Training complete!")
                    print("="*70)
                    break
            elif result.get('status') == 'done':
                print("\n" + "="*70)
                print("Training complete! (server returned done)")
                print("="*70)
                break
            else:
                print(f"  ⏳ Waiting for {result.get('waiting_for', '?')} more clients...")

            time.sleep(2)

if __name__ == '__main__':
    client = DistributedClient()
    
    try:
        client.run_training()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
