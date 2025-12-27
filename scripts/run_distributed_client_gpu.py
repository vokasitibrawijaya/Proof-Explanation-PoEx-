#!/usr/bin/env python3
"""
Distributed FL Client with GPU Support (PyTorch)
Runs in Docker container with CUDA, communicates with aggregator
"""

import os
import time
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import shap
import pickle
import base64
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class SimpleNN(nn.Module):
    """Simple Neural Network for binary classification"""
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class DistributedClientGPU:
    def __init__(self):
        self.node_id = int(os.getenv('NODE_ID', 0))
        self.aggregator_url = os.getenv('AGGREGATOR_URL', 'http://aggregator:5000')
        self.blockchain_rpc = os.getenv('BLOCKCHAIN_RPC', 'http://blockchain:8545')
        
        # Setup device (GPU if available, fallback to CPU)
        try:
            if torch.cuda.is_available():
                # Test CUDA before using
                torch.cuda.init()
                test_tensor = torch.zeros(1).cuda()
                self.device = torch.device('cuda')
                gpu_available = True
            else:
                self.device = torch.device('cpu')
                gpu_available = False
        except Exception as e:
            print(f"  ⚠ CUDA error: {e}")
            print(f"  → Falling back to CPU")
            self.device = torch.device('cpu')
            gpu_available = False
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
        self.trust_score = 1.0
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        print(f"\n{'='*70}")
        print(f"FL Client {self.node_id} (PyTorch)")
        print(f"{'='*70}")
        print(f"  Aggregator: {self.aggregator_url}")
        print(f"  Blockchain: {self.blockchain_rpc}")
        print(f"  Device: {self.device}")
        if gpu_available:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print(f"  Running on CPU (GPU not available or incompatible)")
    
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
        
        # Initialize model with correct input dimension
        if self.model is None:
            input_dim = self.X_train.shape[1]
            self.model = SimpleNN(input_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"✓ Received: {len(self.X_train)} train, {len(self.X_test)} test samples")
    
    def get_global_model(self):
        """Get global model from aggregator"""
        response = requests.get(f'{self.aggregator_url}/get_model')
        data = response.json()
        
        # For first round with PyTorch model
        if 'pytorch_model' in data:
            model_bytes = base64.b64decode(data['pytorch_model'].encode('utf-8'))
            state_dict = pickle.loads(model_bytes)
            self.model.load_state_dict(state_dict)
        
        return data['round']
    
    def train_local(self, epochs=5):
        """Train local model on GPU"""
        self.model.train()
        
        # Convert to tensors and move to device
        X_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_tensor = torch.FloatTensor(self.y_train).reshape(-1, 1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
    
    def compute_accuracy(self):
        """Compute accuracy on test set"""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X_test).to(self.device)
            outputs = self.model(X_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
            accuracy = (predictions == self.y_test).mean()
        
        return accuracy
    
    def compute_shap(self, n_samples=10):
        """Compute SHAP values for explainability"""
        print("  → Computing SHAP values...")
        
        # Use CPU for SHAP computation
        self.model.eval()
        
        def model_predict(x):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                outputs = self.model(x_tensor)
                return outputs.cpu().numpy()
        
        # Sample background data
        background = shap.sample(self.X_train, min(100, len(self.X_train)))
        
        # Use KernelExplainer (model-agnostic)
        explainer = shap.KernelExplainer(model_predict, background)
        
        # Compute SHAP for sample
        sample_X = self.X_test[:n_samples]
        shap_values = explainer.shap_values(sample_X, nsamples=100)
        
        # Aggregate
        avg_shap = np.abs(shap_values).mean(axis=0)
        
        return avg_shap
    
    def get_model_weights(self):
        """Get PyTorch model state dict"""
        state_dict = self.model.state_dict()
        # Move to CPU for serialization
        state_dict = {k: v.cpu() for k, v in state_dict.items()}
        
        model_bytes = pickle.dumps(state_dict)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        return {
            'pytorch_model': model_b64,
            'model_type': 'pytorch'
        }
    
    def submit_update(self, round_num, shap_values, accuracy):
        """Submit model update to aggregator"""
        print("  → Submitting update...")
        
        weights = self.get_model_weights()
        
        response = requests.post(
            f'{self.aggregator_url}/submit_update',
            json={
                'client_id': self.node_id,
                'round': round_num,
                'model_weights': weights,
                'shap_values': shap_values.tolist(),
                'accuracy': float(accuracy)
            },
            timeout=30
        )
        
        data = response.json()
        
        if data['status'] == 'accepted':
            print(f"  ✓ Update aggregated | Trust: {data['trust_score']:.4f}")
        elif data['status'] == 'waiting':
            print(f"  ⏳ Waiting for {data['waiting_for']} more clients...")
        
        return data
    
    def run(self, n_rounds=10):
        """Main training loop"""
        try:
            # Wait for aggregator
            self.wait_for_aggregator()
            
            # Register
            self.register()
            
            # Get data
            self.get_training_data()
            
            # Training rounds
            for round_num in range(1, n_rounds + 1):
                print(f"\n{'='*70}")
                print(f"[Round {round_num}] Client {self.node_id}")
                print(f"{'='*70}")
                
                # Get global model
                self.get_global_model()
                
                # Train locally
                print("  → Training locally...")
                self.train_local(epochs=5)
                
                # Compute SHAP
                shap_values = self.compute_shap(n_samples=10)
                
                # Evaluate
                accuracy = self.compute_accuracy()
                print(f"  → Local accuracy: {accuracy:.4f}")
                
                # Submit update
                result = self.submit_update(round_num, shap_values, accuracy)
                
                # Check if training is complete
                if result.get('complete'):
                    print("\n" + "="*70)
                    print("Training complete!")
                    print("="*70)
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    client = DistributedClientGPU()
    client.run()
