#!/usr/bin/env python3
"""
FAST TEST CLIENT - No SHAP for speed testing
"""

import os
import time
import numpy as np
import requests
from sklearn.linear_model import SGDClassifier
import pickle
import base64

class FastTestClient:
    def __init__(self):
        self.node_id = int(os.getenv('NODE_ID', 0))
        self.aggregator_url = os.getenv('AGGREGATOR_URL', 'http://aggregator:5000')
        
        np.random.seed(42 + self.node_id)
        self.model = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
        
        print(f"\n{'='*70}")
        print(f"FAST TEST Client {self.node_id}")
        print(f"{'='*70}")
        print(f"  Aggregator: {self.aggregator_url}")
        print(f"  SHAP: DISABLED (using dummy values for speed)")
    
    def wait_for_aggregator(self):
        print("\nWaiting for aggregator...")
        for i in range(30):
            try:
                response = requests.get(f'{self.aggregator_url}/health', timeout=2)
                if response.status_code == 200:
                    print("✓ Aggregator is ready")
                    return True
            except:
                time.sleep(1)
        print("✗ Aggregator not available")
        return False
    
    def register(self):
        print("\nRegistering with aggregator...")
        response = requests.post(f'{self.aggregator_url}/register', json={'client_id': self.node_id})
        data = response.json()
        print(f"✓ Registered: {data['n_train']} training samples")
        return data
    
    def get_data(self):
        print("\nFetching training data...")
        response = requests.get(f'{self.aggregator_url}/get_data')
        data = response.json()
        
        self.X_train = np.array(data['X_train'])
        self.y_train = np.array(data['y_train'])
        self.X_test = np.array(data['X_test'])
        self.y_test = np.array(data['y_test'])
        print(f"✓ Received: {len(self.X_train)} train, {len(self.X_test)} test samples")
    
    def get_global_model(self):
        response = requests.get(f'{self.aggregator_url}/get_model')
        data = response.json()

        if not data.get('continue', True):
            return data['round'], False
        
        model_bytes = base64.b64decode(data['model'].encode('utf-8'))
        weights = pickle.loads(model_bytes)
        
        self.model.coef_ = weights['coef']
        self.model.intercept_ = weights['intercept']

        return data['round'], True
    
    def train_local(self):
        for _ in range(1):
            self.model.partial_fit(self.X_train, self.y_train, classes=np.array([0, 1]))
    
    def submit_update(self, round_num):
        # Serialize model
        model_bytes = pickle.dumps({'coef': self.model.coef_, 'intercept': self.model.intercept_})
        model_str = base64.b64encode(model_bytes).decode('utf-8')
        
        # Use DUMMY SHAP values (all zeros) for speed test
        n_features = self.X_train.shape[1]
        dummy_shap = np.zeros(n_features)
        
        # Calculate accuracy
        accuracy = self.model.score(self.X_test, self.y_test)
        
        response = requests.post(
            f'{self.aggregator_url}/submit_update',
            json={
                'client_id': self.node_id,
                'model': model_str,
                'shap_values': dummy_shap.tolist(),
                'accuracy': float(accuracy),
            },
            timeout=30,
        )
        return response.json()
    
    def run(self):
        if not self.wait_for_aggregator():
            return
        
        self.register()
        self.get_data()
        
        while True:
            round_num, should_continue = self.get_global_model()
            if not should_continue:
                print(f"\n✓ Training completed after {round_num} rounds")
                break
            
            print(f"\n{'='*70}")
            print(f"[Round {round_num + 1}] Client {self.node_id}")
            print(f"{'='*70}")
            print("  → Training locally...")
            self.train_local()
            
            print("  → Submitting update (with dummy SHAP)...")
            response = self.submit_update(round_num)
            
            if response.get('status') == 'rejected':
                print(f"  ✗ Update REJECTED")
            else:
                print(f"  ✓ Update accepted - Status: {response.get('status')}")
            
            time.sleep(0.5)

if __name__ == "__main__":
    client = FastTestClient()
    client.run()
