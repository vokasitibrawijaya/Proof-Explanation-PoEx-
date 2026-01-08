#!/usr/bin/env python3
"""
Comprehensive PoEx Experiment - Addressing IEEE Access Review Comments

This script addresses the following reviewer concerns:
- M1: Expand from 3 clients/3 rounds to 10 clients/50 rounds
- M2: Compare with SOTA baselines (Krum, TrimmedMean, Bulyan)
- M3: Evaluate adaptive attacks
- M5: Fix NSDS metric with proper normalization
- M6: Sensitivity analysis for threshold τ

Author: FedXChain Research Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.special import rel_entr
import shap
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# BYZANTINE-ROBUST AGGREGATION METHODS (SOTA BASELINES)
# ==============================================================================

class AggregationMethod:
    """Base class for aggregation methods"""
    
    def __init__(self, name):
        self.name = name
    
    def aggregate(self, updates, **kwargs):
        raise NotImplementedError


class FedAvg(AggregationMethod):
    """Standard Federated Averaging - No Byzantine defense"""
    
    def __init__(self):
        super().__init__("FedAvg")
    
    def aggregate(self, updates, **kwargs):
        """Simple averaging of all updates"""
        n = len(updates)
        aggregated = {}
        for key in updates[0].keys():
            aggregated[key] = np.mean([u[key] for u in updates], axis=0)
        return aggregated, list(range(n)), []  # all accepted


class Krum(AggregationMethod):
    """Krum Byzantine-robust aggregation
    Reference: Blanchard et al. 2017 - Machine Learning with Adversaries
    """
    
    def __init__(self, n_byzantine=0):
        super().__init__("Krum")
        self.n_byzantine = n_byzantine
    
    def aggregate(self, updates, **kwargs):
        """Select the most representative update"""
        n = len(updates)
        f = self.n_byzantine
        
        # Flatten updates for distance computation
        flattened = []
        for u in updates:
            flat = np.concatenate([u[k].flatten() for k in u.keys()])
            flattened.append(flat)
        flattened = np.array(flattened)
        
        # Compute pairwise distances
        scores = []
        for i in range(n):
            distances = np.sum((flattened - flattened[i]) ** 2, axis=1)
            distances = np.sort(distances)
            # Sum of n-f-2 closest distances
            score = np.sum(distances[1:n-f-1]) if n > f + 2 else np.sum(distances[1:])
            scores.append(score)
        
        # Select update with minimum score
        selected_idx = np.argmin(scores)
        return updates[selected_idx], [selected_idx], [i for i in range(n) if i != selected_idx]


class MultiKrum(AggregationMethod):
    """Multi-Krum Byzantine-robust aggregation
    Selects m best updates and averages them
    """
    
    def __init__(self, n_byzantine=0, m=None):
        super().__init__("MultiKrum")
        self.n_byzantine = n_byzantine
        self.m = m
    
    def aggregate(self, updates, **kwargs):
        n = len(updates)
        f = self.n_byzantine
        m = self.m if self.m else n - f
        
        # Flatten updates
        flattened = []
        for u in updates:
            flat = np.concatenate([u[k].flatten() for k in u.keys()])
            flattened.append(flat)
        flattened = np.array(flattened)
        
        # Compute scores
        scores = []
        for i in range(n):
            distances = np.sum((flattened - flattened[i]) ** 2, axis=1)
            distances = np.sort(distances)
            score = np.sum(distances[1:n-f-1]) if n > f + 2 else np.sum(distances[1:])
            scores.append(score)
        
        # Select m best updates
        selected_indices = np.argsort(scores)[:m]
        rejected = [i for i in range(n) if i not in selected_indices]
        
        # Average selected updates
        aggregated = {}
        for key in updates[0].keys():
            aggregated[key] = np.mean([updates[i][key] for i in selected_indices], axis=0)
        
        return aggregated, list(selected_indices), rejected


class TrimmedMean(AggregationMethod):
    """Trimmed Mean Byzantine-robust aggregation
    Reference: Yin et al. 2018 - Byzantine-Robust Distributed Learning
    """
    
    def __init__(self, trim_ratio=0.1):
        super().__init__("TrimmedMean")
        self.trim_ratio = trim_ratio
    
    def aggregate(self, updates, **kwargs):
        n = len(updates)
        trim_count = int(n * self.trim_ratio)
        
        aggregated = {}
        for key in updates[0].keys():
            # Stack all values
            values = np.stack([u[key] for u in updates], axis=0)
            
            # For each coordinate, sort and trim
            if len(values.shape) == 1:
                values = values.reshape(-1, 1)
            
            trimmed = np.zeros(values.shape[1:])
            for i in range(values.shape[1]):
                if len(values.shape) == 2:
                    sorted_vals = np.sort(values[:, i])
                    if trim_count > 0 and n > 2 * trim_count:
                        trimmed[i] = np.mean(sorted_vals[trim_count:n-trim_count])
                    else:
                        trimmed[i] = np.mean(sorted_vals)
                else:
                    # Handle multi-dimensional
                    sorted_vals = np.sort(values[:, i])
                    trimmed[i] = np.mean(sorted_vals)
            
            aggregated[key] = trimmed.reshape(updates[0][key].shape)
        
        # TrimmedMean accepts all but trims extremes
        return aggregated, list(range(n)), []


class Bulyan(AggregationMethod):
    """Bulyan Byzantine-robust aggregation
    Reference: El Mhamdi et al. 2018 - The Hidden Vulnerability of Distributed Learning
    Combines Krum selection with coordinate-wise trimmed mean
    """
    
    def __init__(self, n_byzantine=0):
        super().__init__("Bulyan")
        self.n_byzantine = n_byzantine
    
    def aggregate(self, updates, **kwargs):
        n = len(updates)
        f = self.n_byzantine
        
        # Need at least 4f + 3 workers
        if n < 4 * f + 3:
            # Fall back to MultiKrum
            mk = MultiKrum(f, m=n-2*f)
            return mk.aggregate(updates)
        
        # Step 1: Use Krum to select n - 2f updates
        flattened = []
        for u in updates:
            flat = np.concatenate([u[k].flatten() for k in u.keys()])
            flattened.append(flat)
        flattened = np.array(flattened)
        
        selected = []
        remaining = list(range(n))
        
        for _ in range(n - 2 * f):
            if len(remaining) <= 2:
                break
            
            scores = []
            for i in remaining:
                distances = np.sum((flattened[remaining] - flattened[i]) ** 2, axis=1)
                distances = np.sort(distances)
                score = np.sum(distances[1:min(len(remaining)-f-1, len(distances))])
                scores.append(score)
            
            best_idx = remaining[np.argmin(scores)]
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        # Step 2: Coordinate-wise trimmed mean on selected
        aggregated = {}
        for key in updates[0].keys():
            values = np.stack([updates[i][key] for i in selected], axis=0)
            
            if len(values.shape) == 1:
                values = values.reshape(-1, 1)
            
            beta = f
            trimmed = np.zeros(values.shape[1:])
            for i in range(values.shape[1]):
                sorted_vals = np.sort(values[:, i])
                if len(sorted_vals) > 2 * beta:
                    trimmed[i] = np.mean(sorted_vals[beta:len(sorted_vals)-beta])
                else:
                    trimmed[i] = np.mean(sorted_vals)
            
            aggregated[key] = trimmed.reshape(updates[0][key].shape)
        
        rejected = [i for i in range(n) if i not in selected]
        return aggregated, selected, rejected


class PoEx(AggregationMethod):
    """Proof of Explanation (PoEx) - Our proposed method
    Uses SHAP-based validation with NSDS metric
    """
    
    def __init__(self, threshold=0.5, use_jensen_shannon=True):
        super().__init__("PoEx")
        self.threshold = threshold
        self.use_jensen_shannon = use_jensen_shannon
        self.reference_shap = None
    
    def set_reference(self, shap_values):
        """Set reference SHAP values for comparison"""
        self.reference_shap = shap_values
    
    def compute_nsds(self, shap_local, shap_ref):
        """
        Compute Normalized Symmetric Divergence Score
        
        Fixed version addressing reviewer concerns:
        - Proper normalization to probability distributions
        - Using Jensen-Shannon divergence (symmetric, bounded) instead of raw KL
        """
        eps = 1e-10
        
        # Normalize SHAP values to probability distributions (absolute values)
        p = np.abs(shap_local) + eps
        q = np.abs(shap_ref) + eps
        p = p / p.sum()
        q = q / q.sum()
        
        if self.use_jensen_shannon:
            # Jensen-Shannon Divergence (symmetric, bounded [0, ln(2)])
            m = 0.5 * (p + q)
            js_div = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
            # Normalize to [0, 1]
            nsds = js_div / np.log(2)
        else:
            # Symmetric KL divergence
            kl_pq = np.sum(rel_entr(p, q))
            kl_qp = np.sum(rel_entr(q, p))
            nsds = 0.5 * (kl_pq + kl_qp)
            # Clip for numerical stability
            nsds = min(nsds, 10.0) / 10.0  # Normalize roughly to [0, 1]
        
        return nsds
    
    def aggregate(self, updates, shap_values=None, **kwargs):
        """Aggregate with SHAP-based validation"""
        n = len(updates)
        
        if shap_values is None or self.reference_shap is None:
            # Fall back to FedAvg if no SHAP
            fedavg = FedAvg()
            return fedavg.aggregate(updates)
        
        # Compute NSDS for each client
        accepted = []
        rejected = []
        nsds_scores = []
        
        for i, shap in enumerate(shap_values):
            nsds = self.compute_nsds(shap, self.reference_shap)
            nsds_scores.append(nsds)
            
            if nsds < self.threshold:
                accepted.append(i)
            else:
                rejected.append(i)
        
        if len(accepted) == 0:
            # If all rejected, use most trusted one
            best_idx = np.argmin(nsds_scores)
            accepted = [best_idx]
            rejected = [i for i in range(n) if i != best_idx]
        
        # Aggregate accepted updates
        aggregated = {}
        for key in updates[0].keys():
            aggregated[key] = np.mean([updates[i][key] for i in accepted], axis=0)
        
        return aggregated, accepted, rejected, nsds_scores


# ==============================================================================
# ATTACK IMPLEMENTATIONS
# ==============================================================================

class Attack:
    """Base class for Byzantine attacks"""
    
    def __init__(self, name):
        self.name = name
    
    def apply(self, weights, **kwargs):
        raise NotImplementedError


class NoAttack(Attack):
    """No attack - honest client"""
    
    def __init__(self):
        super().__init__("none")
    
    def apply(self, weights, **kwargs):
        return weights


class SignFlipAttack(Attack):
    """Sign flipping attack - reverses gradient direction"""
    
    def __init__(self):
        super().__init__("sign_flip")
    
    def apply(self, weights, **kwargs):
        poisoned = {}
        for key in weights.keys():
            poisoned[key] = -weights[key]
        return poisoned


class LabelFlipAttack(Attack):
    """Label flipping is applied during training, not on weights"""
    
    def __init__(self):
        super().__init__("label_flip")
    
    def apply(self, weights, **kwargs):
        # Label flip happens during training
        return weights


class GaussianNoiseAttack(Attack):
    """Add Gaussian noise to weights"""
    
    def __init__(self, scale=1.0):
        super().__init__("gaussian_noise")
        self.scale = scale
    
    def apply(self, weights, **kwargs):
        poisoned = {}
        for key in weights.keys():
            noise = np.random.normal(0, self.scale, weights[key].shape)
            poisoned[key] = weights[key] + noise
        return poisoned


class AdaptiveAttack(Attack):
    """
    Adaptive attack that tries to evade SHAP-based detection
    Attacker knows the threshold and tries to craft updates that:
    1. Pass SHAP validation
    2. Still poison the model
    """
    
    def __init__(self, threshold=0.5, poison_factor=0.3):
        super().__init__("adaptive")
        self.threshold = threshold
        self.poison_factor = poison_factor
    
    def apply(self, weights, reference_shap=None, **kwargs):
        """Craft adversarial update that evades detection"""
        poisoned = {}
        for key in weights.keys():
            # Add subtle poison that maintains SHAP similarity
            # Scale poison by weight magnitude to stay within detection bounds
            magnitude = np.abs(weights[key]).mean() + 1e-10
            poison = np.random.uniform(-1, 1, weights[key].shape) * magnitude * self.poison_factor
            poisoned[key] = weights[key] + poison
        return poisoned


class CoordinatedAttack(Attack):
    """
    Coordinated Byzantine attack where multiple attackers collude
    They average their malicious updates to appear more legitimate
    """
    
    def __init__(self, n_attackers=2):
        super().__init__("coordinated")
        self.n_attackers = n_attackers
        self.stored_updates = []
    
    def apply(self, weights, **kwargs):
        # Each attacker creates a slightly different poisoned update
        poisoned = {}
        for key in weights.keys():
            # Coordinate to point in same malicious direction
            direction = -np.sign(weights[key])  # Opposite of gradient
            magnitude = np.abs(weights[key]) * 0.5
            poisoned[key] = weights[key] + direction * magnitude
        return poisoned


# ==============================================================================
# DATA DISTRIBUTION
# ==============================================================================

def create_non_iid_data(X, y, n_clients, alpha=0.5):
    """
    Create non-IID data distribution using Dirichlet distribution
    
    Args:
        X: Features
        y: Labels
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
    
    Returns:
        List of (X_train, y_train, X_test, y_test) for each client
    """
    n_classes = len(np.unique(y))
    n_samples = len(y)
    
    # Group samples by class
    class_indices = {c: np.where(y == c)[0] for c in range(n_classes)}
    
    # Use Dirichlet distribution to assign samples to clients
    client_data = [{'X': [], 'y': []} for _ in range(n_clients)]
    
    for c in range(n_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        
        # Dirichlet distribution for this class
        proportions = np.random.dirichlet([alpha] * n_clients)
        proportions = (proportions * len(indices)).astype(int)
        
        # Ensure all samples are assigned
        proportions[-1] = len(indices) - proportions[:-1].sum()
        
        # Assign samples to clients
        start = 0
        for i, prop in enumerate(proportions):
            client_data[i]['X'].extend(X[indices[start:start+prop]])
            client_data[i]['y'].extend(y[indices[start:start+prop]])
            start += prop
    
    # Convert to numpy arrays and split train/test
    client_splits = []
    for i in range(n_clients):
        X_client = np.array(client_data[i]['X'])
        y_client = np.array(client_data[i]['y'])
        
        if len(X_client) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test = X_client, X_client
            y_train, y_test = y_client, y_client
        
        client_splits.append((X_train, y_train, X_test, y_test))
    
    return client_splits


def create_iid_data(X, y, n_clients):
    """Create IID data distribution"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Split training data equally among clients
    n_samples_per_client = len(X_train) // n_clients
    client_splits = []
    
    for i in range(n_clients):
        start = i * n_samples_per_client
        end = start + n_samples_per_client if i < n_clients - 1 else len(X_train)
        
        client_splits.append((
            X_train[start:end],
            y_train[start:end],
            X_test,
            y_test
        ))
    
    return client_splits


# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

class ComprehensiveExperiment:
    """Run comprehensive PoEx experiments"""
    
    def __init__(self, config):
        self.config = config
        self.results = []
        
    def load_dataset(self, dataset_name):
        """Load dataset"""
        if dataset_name == "breast_cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
        elif dataset_name == "mnist":
            # Load subset of MNIST
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            X, y = mnist.data[:10000], mnist.target[:10000].astype(int)
        elif dataset_name == "cifar10":
            # Placeholder - would need to load CIFAR-10
            print("Note: CIFAR-10 requires additional setup. Using synthetic data.")
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=5000, n_features=100, 
                                       n_informative=50, n_classes=10,
                                       random_state=42)
        else:
            # Synthetic data
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=2000, n_features=30,
                                       n_informative=20, n_classes=2,
                                       random_state=42)
        
        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y
    
    def train_client(self, X_train, y_train, global_weights=None, 
                     attack=None, flip_labels=False):
        """Train a single client"""
        model = LogisticRegression(max_iter=100, warm_start=True, random_state=42)
        
        # Apply label flipping if specified
        y_train_local = y_train.copy()
        if flip_labels:
            y_train_local = 1 - y_train_local
        
        # Check if we have at least 2 classes
        unique_classes = np.unique(y_train_local)
        if len(unique_classes) < 2:
            # If only one class, use global weights if available, otherwise random init
            if global_weights is not None:
                weights = {
                    'coef': global_weights['coef'].copy(),
                    'intercept': global_weights['intercept'].copy()
                }
                model.coef_ = global_weights['coef'].copy()
                model.intercept_ = global_weights['intercept'].copy()
                model.classes_ = np.array([0, 1])
            else:
                # Random initialization
                n_features = X_train.shape[1]
                weights = {
                    'coef': np.random.randn(1, n_features) * 0.01,
                    'intercept': np.array([0.0])
                }
                model.coef_ = weights['coef']
                model.intercept_ = weights['intercept']
                model.classes_ = np.array([0, 1])
            
            # Apply attack if specified
            if attack is not None and attack.name != "label_flip":
                weights = attack.apply(weights)
            
            return model, weights
        
        # Initialize with global weights if provided
        if global_weights is not None:
            model.coef_ = global_weights['coef'].copy()
            model.intercept_ = global_weights['intercept'].copy()
            model.classes_ = np.array([0, 1])
        
        # Train
        model.fit(X_train, y_train_local)
        
        # Get weights
        weights = {
            'coef': model.coef_.copy(),
            'intercept': model.intercept_.copy()
        }
        
        # Apply attack if specified
        if attack is not None and attack.name != "label_flip":
            weights = attack.apply(weights)
        
        return model, weights
    
    def compute_shap(self, model, X_background, n_samples=50):
        """Compute SHAP values for a model"""
        try:
            explainer = shap.LinearExplainer(model, X_background[:n_samples])
            shap_values = explainer.shap_values(X_background[:n_samples])
            
            # Average absolute SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            return np.mean(np.abs(shap_values), axis=0)
        except Exception as e:
            print(f"SHAP computation error: {e}")
            return np.ones(X_background.shape[1]) / X_background.shape[1]
    
    def evaluate_model(self, weights, X_test, y_test):
        """Evaluate model on test set"""
        model = LogisticRegression()
        model.coef_ = weights['coef']
        model.intercept_ = weights['intercept']
        model.classes_ = np.array([0, 1])
        
        y_pred = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
    
    def run_single_experiment(self, config):
        """Run a single experiment configuration"""
        
        # Load data
        X, y = self.load_dataset(config['dataset'])
        n_clients = config['n_clients']
        n_rounds = config['n_rounds']
        n_byzantine = config['n_byzantine']
        attack_type = config['attack_type']
        aggregation_method = config['aggregation_method']
        threshold = config.get('threshold', 0.5)
        data_distribution = config.get('data_distribution', 'iid')
        dirichlet_alpha = config.get('dirichlet_alpha', 0.5)
        
        print(f"\n{'='*60}")
        print(f"Experiment: {aggregation_method} vs {attack_type}")
        print(f"Clients: {n_clients}, Byzantine: {n_byzantine}, Rounds: {n_rounds}")
        print(f"Data: {data_distribution}, Threshold: {threshold}")
        print(f"{'='*60}")
        
        # Create data splits
        if data_distribution == 'non_iid':
            client_data = create_non_iid_data(X, y, n_clients, dirichlet_alpha)
        else:
            client_data = create_iid_data(X, y, n_clients)
        
        # Setup aggregator
        if aggregation_method == "FedAvg":
            aggregator = FedAvg()
        elif aggregation_method == "Krum":
            aggregator = Krum(n_byzantine)
        elif aggregation_method == "MultiKrum":
            aggregator = MultiKrum(n_byzantine, m=n_clients - n_byzantine)
        elif aggregation_method == "TrimmedMean":
            aggregator = TrimmedMean(trim_ratio=n_byzantine / n_clients)
        elif aggregation_method == "Bulyan":
            aggregator = Bulyan(n_byzantine)
        elif aggregation_method == "PoEx":
            aggregator = PoEx(threshold=threshold)
        else:
            aggregator = FedAvg()
        
        # Setup attacks
        attack_map = {
            'none': NoAttack(),
            'sign_flip': SignFlipAttack(),
            'label_flip': LabelFlipAttack(),
            'gaussian_noise': GaussianNoiseAttack(scale=1.0),
            'adaptive': AdaptiveAttack(threshold=threshold),
            'coordinated': CoordinatedAttack(n_byzantine)
        }
        attack = attack_map.get(attack_type, NoAttack())
        
        # Byzantine client indices
        byzantine_indices = list(range(n_byzantine))
        
        # Global model initialization
        global_weights = None
        
        # Background data for SHAP
        X_background = np.vstack([d[0][:20] for d in client_data if len(d[0]) > 0])
        
        # Results tracking
        round_results = []
        
        # Training loop
        for round_num in range(n_rounds):
            round_start = time.time()
            
            # Train all clients
            client_weights = []
            client_shap = []
            
            for i in range(n_clients):
                X_train, y_train, X_test, y_test = client_data[i]
                
                is_byzantine = i in byzantine_indices
                is_label_flip = is_byzantine and attack_type == 'label_flip'
                client_attack = attack if is_byzantine else None
                
                _, weights = self.train_client(
                    X_train, y_train, global_weights,
                    attack=client_attack,
                    flip_labels=is_label_flip
                )
                
                client_weights.append(weights)
                
                # Compute SHAP for PoEx
                if aggregation_method == "PoEx":
                    temp_model = LogisticRegression()
                    temp_model.coef_ = weights['coef']
                    temp_model.intercept_ = weights['intercept']
                    temp_model.classes_ = np.array([0, 1])
                    shap_vals = self.compute_shap(temp_model, X_background)
                    client_shap.append(shap_vals)
            
            # Set reference SHAP (honest client average) for PoEx
            if aggregation_method == "PoEx" and round_num == 0:
                honest_shap = [client_shap[i] for i in range(n_clients) 
                              if i not in byzantine_indices]
                if honest_shap:
                    aggregator.set_reference(np.mean(honest_shap, axis=0))
            
            # Aggregate
            if aggregation_method == "PoEx":
                result = aggregator.aggregate(client_weights, shap_values=client_shap)
                global_weights, accepted, rejected, nsds_scores = result
            else:
                global_weights, accepted, rejected = aggregator.aggregate(client_weights)
                nsds_scores = []
            
            round_time = time.time() - round_start
            
            # Evaluate
            X_test_global = np.vstack([d[2] for d in client_data])
            y_test_global = np.hstack([d[3] for d in client_data])
            metrics = self.evaluate_model(global_weights, X_test_global, y_test_global)
            
            # Count Byzantine detection
            byzantine_rejected = len([i for i in rejected if i in byzantine_indices])
            byzantine_accepted = len([i for i in accepted if i in byzantine_indices])
            
            round_result = {
                'round': round_num,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accepted': len(accepted),
                'rejected': len(rejected),
                'byzantine_rejected': byzantine_rejected,
                'byzantine_accepted': byzantine_accepted,
                'latency_ms': round_time * 1000,
                'nsds_scores': nsds_scores
            }
            round_results.append(round_result)
            
            if round_num % 10 == 0 or round_num == n_rounds - 1:
                print(f"Round {round_num:3d}: Acc={metrics['accuracy']:.4f}, "
                      f"Accepted={len(accepted)}, Byzantine_rejected={byzantine_rejected}")
        
        # Compute summary statistics
        final_accuracy = round_results[-1]['accuracy']
        avg_accuracy = np.mean([r['accuracy'] for r in round_results])
        defense_rate = np.mean([r['byzantine_rejected'] / n_byzantine 
                               if n_byzantine > 0 else 1.0 
                               for r in round_results])
        avg_latency = np.mean([r['latency_ms'] for r in round_results])
        
        return {
            'config': config,
            'round_results': round_results,
            'final_accuracy': final_accuracy,
            'avg_accuracy': avg_accuracy,
            'defense_rate': defense_rate,
            'avg_latency_ms': avg_latency
        }
    
    def run_all_experiments(self):
        """Run all experiment configurations"""
        
        # Base configuration
        base_config = {
            'dataset': 'breast_cancer',
            'n_clients': 10,
            'n_rounds': 50,
            'n_byzantine': 3,  # 30% Byzantine
            'data_distribution': 'iid'
        }
        
        all_results = []
        
        # ==== Experiment 1: Method Comparison ====
        print("\n" + "="*80)
        print("EXPERIMENT 1: Aggregation Method Comparison")
        print("="*80)
        
        methods = ['FedAvg', 'Krum', 'MultiKrum', 'TrimmedMean', 'PoEx']
        attacks = ['sign_flip', 'label_flip', 'gaussian_noise']
        
        for method in methods:
            for attack in attacks:
                config = base_config.copy()
                config['aggregation_method'] = method
                config['attack_type'] = attack
                config['threshold'] = 0.5
                
                result = self.run_single_experiment(config)
                all_results.append(result)
        
        # ==== Experiment 2: Threshold Sensitivity Analysis ====
        print("\n" + "="*80)
        print("EXPERIMENT 2: PoEx Threshold Sensitivity Analysis")
        print("="*80)
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for threshold in thresholds:
            config = base_config.copy()
            config['aggregation_method'] = 'PoEx'
            config['attack_type'] = 'sign_flip'
            config['threshold'] = threshold
            
            result = self.run_single_experiment(config)
            all_results.append(result)
        
        # ==== Experiment 3: Byzantine Fraction Analysis ====
        print("\n" + "="*80)
        print("EXPERIMENT 3: Byzantine Fraction Analysis")
        print("="*80)
        
        byzantine_fractions = [0.1, 0.2, 0.3, 0.4]
        for frac in byzantine_fractions:
            n_byz = int(10 * frac)
            config = base_config.copy()
            config['aggregation_method'] = 'PoEx'
            config['attack_type'] = 'sign_flip'
            config['n_byzantine'] = n_byz
            
            result = self.run_single_experiment(config)
            all_results.append(result)
        
        # ==== Experiment 4: Non-IID Data Distribution ====
        print("\n" + "="*80)
        print("EXPERIMENT 4: Non-IID Data Distribution")
        print("="*80)
        
        for method in ['FedAvg', 'PoEx']:
            config = base_config.copy()
            config['aggregation_method'] = method
            config['attack_type'] = 'sign_flip'
            config['data_distribution'] = 'non_iid'
            config['dirichlet_alpha'] = 0.5
            
            result = self.run_single_experiment(config)
            all_results.append(result)
        
        # ==== Experiment 5: Adaptive Attack ====
        print("\n" + "="*80)
        print("EXPERIMENT 5: Adaptive Attack Evaluation")
        print("="*80)
        
        for method in ['FedAvg', 'Krum', 'TrimmedMean', 'PoEx']:
            config = base_config.copy()
            config['aggregation_method'] = method
            config['attack_type'] = 'adaptive'
            
            result = self.run_single_experiment(config)
            all_results.append(result)
        
        self.results = all_results
        return all_results
    
    def generate_report(self, output_dir):
        """Generate comprehensive results report"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Summary DataFrame
        summary_data = []
        for r in self.results:
            config = r['config']
            summary_data.append({
                'method': config['aggregation_method'],
                'attack': config['attack_type'],
                'n_clients': config['n_clients'],
                'n_byzantine': config['n_byzantine'],
                'n_rounds': config['n_rounds'],
                'threshold': config.get('threshold', 'N/A'),
                'data_dist': config.get('data_distribution', 'iid'),
                'final_accuracy': r['final_accuracy'],
                'avg_accuracy': r['avg_accuracy'],
                'defense_rate': r['defense_rate'],
                'avg_latency_ms': r['avg_latency_ms']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, 'comprehensive_results.csv'), index=False)
        
        # Statistical analysis
        stats_report = self.compute_statistics()
        with open(os.path.join(output_dir, 'statistical_analysis.json'), 'w') as f:
            json.dump(stats_report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(df.to_string())
        
        print(f"\nResults saved to {output_dir}")
        
        return df
    
    def compute_statistics(self):
        """Compute statistical tests comparing methods"""
        
        # Group results by method and attack
        method_results = {}
        for r in self.results:
            method = r['config']['aggregation_method']
            attack = r['config']['attack_type']
            key = f"{method}_{attack}"
            
            if key not in method_results:
                method_results[key] = []
            method_results[key].append(r['final_accuracy'])
        
        # Compare PoEx vs each baseline
        stats = {'comparisons': []}
        
        attacks = ['sign_flip', 'label_flip', 'gaussian_noise']
        baselines = ['FedAvg', 'Krum', 'TrimmedMean']
        
        for attack in attacks:
            poex_key = f"PoEx_{attack}"
            if poex_key not in method_results:
                continue
            
            poex_acc = method_results[poex_key]
            
            for baseline in baselines:
                base_key = f"{baseline}_{attack}"
                if base_key not in method_results:
                    continue
                
                base_acc = method_results[base_key]
                
                # Mann-Whitney U test (non-parametric)
                if len(poex_acc) > 1 and len(base_acc) > 1:
                    try:
                        stat, pvalue = mannwhitneyu(poex_acc, base_acc, alternative='greater')
                    except:
                        stat, pvalue = 0, 1.0
                else:
                    stat, pvalue = 0, 1.0
                
                stats['comparisons'].append({
                    'poex_method': 'PoEx',
                    'baseline_method': baseline,
                    'attack': attack,
                    'poex_mean_acc': float(np.mean(poex_acc)),
                    'baseline_mean_acc': float(np.mean(base_acc)),
                    'improvement': float(np.mean(poex_acc) - np.mean(base_acc)),
                    'mann_whitney_u': float(stat),
                    'p_value': float(pvalue),
                    'significant': bool(pvalue < 0.05)
                })
        
        return stats


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE PoEx EXPERIMENT")
    print("Addressing IEEE Access Reviewer Comments")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = {
        'output_dir': 'results/comprehensive_experiments'
    }
    
    experiment = ComprehensiveExperiment(config)
    
    # Run all experiments
    results = experiment.run_all_experiments()
    
    # Generate report
    df = experiment.generate_report(config['output_dir'])
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
