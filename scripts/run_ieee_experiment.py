#!/usr/bin/env python3
"""
IEEE Access Compliant Experimental Framework for FedXChain
Implements rigorous experimental methodology with:
- Baseline vs Proposed comparison
- Attack scenarios (Label Flipping, Gaussian Noise)
- Multiple runs for statistical significance
- Comprehensive metrics and statistical analysis
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import platform
import sys
import yaml
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge updates into base and return base."""
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_ieee_config(config_path: Path) -> dict:
    """Load IEEE config YAML and normalize to a flat runtime config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    dataset_key = raw.get('active_dataset', 'synthetic')
    dataset_cfg = raw.get('datasets', {}).get(dataset_key, {})
    fl_cfg = raw.get('federated_learning', {})
    dist_cfg = fl_cfg.get('data_distribution', {})
    stat_cfg = raw.get('statistical_analysis', {})
    repro_cfg = raw.get('reproducibility', {})
    output_cfg = raw.get('output', {})

    cfg = {
        'dataset': dataset_key,
        'n_samples': dataset_cfg.get('n_samples', 1000),
        'n_features': dataset_cfg.get('n_features', 20),
        'test_size': dataset_cfg.get('test_size', 0.2),
        'n_clients': int(fl_cfg.get('n_clients', 10)),
        'rounds': int(fl_cfg.get('rounds', 20)),
        'local_epochs': int(fl_cfg.get('local_epochs', 1)),
        'batch_size': int(fl_cfg.get('batch_size', 32)),
        'dirichlet_alpha': float(dist_cfg.get('alpha', 0.5)),
        'min_samples_per_client': int(dist_cfg.get('min_samples_per_client', 10)),
        'methods': raw.get('methods', {}),
        'attack_scenarios': raw.get('attack_scenarios', {}),
        'ablation_study': raw.get('ablation_study', {}),
        'n_runs': int(stat_cfg.get('n_runs', 10)),
        'random_seeds': stat_cfg.get('random_seeds', [42]),
        'confidence_level': float(stat_cfg.get('confidence_level', 0.95)),
        'output': {
            'directory': output_cfg.get('directory', 'results_ieee'),
            'save_format': output_cfg.get('save_format', ['csv']),
        },
        'reproducibility': {
            'set_seed': bool(repro_cfg.get('set_seed', True)),
            'seed': int(repro_cfg.get('seed', 42)),
            'deterministic': bool(repro_cfg.get('deterministic', True)),
            'log_environment': bool(repro_cfg.get('log_environment', True)),
            'save_config': bool(repro_cfg.get('save_config', True)),
        },
    }

    return cfg


def log_environment(output_dir: Path) -> None:
    """Persist minimal environment metadata for reproducibility."""
    try:
        from importlib import metadata as importlib_metadata
    except Exception:
        import importlib_metadata  # type: ignore

    packages = [
        'numpy', 'pandas', 'scikit-learn', 'shap', 'scipy', 'matplotlib', 'seaborn', 'pyyaml'
    ]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib_metadata.version(name)
        except Exception:
            versions[name] = None

    env = {
        'timestamp': datetime.now().isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'packages': versions,
    }
    with open(output_dir / 'environment.json', 'w', encoding='utf-8') as f:
        json.dump(env, f, indent=2)

class FedLearningNode:
    """Federated Learning Node with optional attack capability"""
    
    def __init__(self, node_id, X_train, y_train, X_test, y_test, is_malicious=False, attack_type=None, attack_intensity=0.3):
        self.node_id = node_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.attack_intensity = attack_intensity
        self.model = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
        self.trust_score = 1.0
        
    def apply_attack(self, weights):
        """Apply attack to model weights"""
        if not self.is_malicious or self.attack_type is None:
            return weights
            
        if self.attack_type == 'label_flip':
            # Label flipping attack - already applied during training
            return weights
            
        elif self.attack_type == 'gaussian_noise':
            # Add Gaussian noise to weights
            noise_coef = np.random.normal(0, self.attack_intensity, weights['coef'].shape)
            noise_intercept = np.random.normal(0, self.attack_intensity, weights['intercept'].shape)
            
            return {
                'coef': weights['coef'] + noise_coef,
                'intercept': weights['intercept'] + noise_intercept
            }
            
        elif self.attack_type == 'sign_flip':
            # Sign flipping attack
            return {
                'coef': -weights['coef'],
                'intercept': -weights['intercept']
            }
            
        return weights
    
    def train_local(self, global_weights=None, epochs=1):
        """Train local model with potential attacks"""
        if global_weights is not None:
            self.model.coef_ = global_weights['coef'].copy()
            self.model.intercept_ = global_weights['intercept'].copy()
        
        # Apply label flipping attack if malicious
        y_train = self.y_train.copy()
        if self.is_malicious and self.attack_type == 'label_flip':
            flip_indices = np.random.choice(
                len(y_train), 
                int(len(y_train) * self.attack_intensity), 
                replace=False
            )
            y_train[flip_indices] = 1 - y_train[flip_indices]
        
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, y_train, classes=np.array([0, 1]))
        
        weights = {
            'coef': self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy()
        }
        
        # Apply weight-based attacks and persist them to the node model
        attacked = self.apply_attack(weights)
        self.model.coef_ = attacked['coef'].copy()
        self.model.intercept_ = attacked['intercept'].copy()
        return attacked
    
    def compute_shap(self, n_samples=10):
        """Compute SHAP values for explainability"""
        sample_idx = np.random.choice(len(self.X_train), min(n_samples, len(self.X_train)), replace=False)
        X_sample = self.X_train[sample_idx]
        
        explainer = shap.KernelExplainer(self.model.predict_proba, X_sample)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        return np.mean(np.abs(shap_values), axis=0)
    
    def evaluate(self, X=None, y=None):
        """Evaluate model performance"""
        if X is None:
            X, y = self.X_test, self.y_test
        
        y_pred = self.model.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }

class FedXChainExperiment:
    """Main experimental framework for IEEE Access paper"""
    
    def __init__(self, config):
        self.config = config
        self.results = []
        
    def create_dataset(self):
        """Create or load dataset"""
        if self.config['dataset'] == 'synthetic':
            X, y = make_classification(
                n_samples=self.config['n_samples'],
                n_features=self.config['n_features'],
                n_classes=2,
                n_informative=int(self.config['n_features'] * 0.7),
                n_redundant=int(self.config['n_features'] * 0.2),
                random_state=self.config.get('seed', 42)
            )
        elif self.config['dataset'] == 'breast_cancer':
            data = load_breast_cancer()
            X, y = data.data, data.target
        else:
            raise ValueError(f"Unknown dataset: {self.config['dataset']}")
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return train_test_split(
            X,
            y,
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('seed', 42),
            stratify=y if len(np.unique(y)) > 1 else None,
        )
    
    def split_data_non_iid(self, X, y, n_clients):
        """Split data into non-IID partitions using Dirichlet distribution"""
        min_size = 0
        K = len(np.unique(y))
        N = len(y)
        
        min_required = int(self.config.get('min_samples_per_client', 10))
        while min_size < min_required:
            idx_batch = [[] for _ in range(n_clients)]
            for k in range(K):
                idx_k = np.where(y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(self.config.get('dirichlet_alpha', 0.5), n_clients)
                )
                proportions = np.array([p * (len(idx_j) < N / n_clients) 
                                       for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() 
                           for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        
        return idx_batch
    
    def create_nodes(self, X_train, y_train, X_test, y_test, malicious_config):
        """Create federated learning nodes with malicious nodes"""
        idx_batch = self.split_data_non_iid(X_train, y_train, self.config['n_clients'])
        
        nodes = []
        for i in range(self.config['n_clients']):
            train_idx = idx_batch[i]
            test_size = max(1, min(len(X_test), len(train_idx) // 4))
            test_idx = np.random.choice(len(X_test), test_size, replace=False)
            
            is_malicious = i in malicious_config['malicious_nodes']
            attack_type = malicious_config.get('attack_type') if is_malicious else None
            attack_intensity = malicious_config.get('attack_intensity', 0.3) if is_malicious else 0.0
            
            node = FedLearningNode(
                node_id=i,
                X_train=X_train[train_idx],
                y_train=y_train[train_idx],
                X_test=X_test[test_idx],
                y_test=y_test[test_idx],
                is_malicious=is_malicious,
                attack_type=attack_type,
                attack_intensity=attack_intensity
            )
            nodes.append(node)
        
        return nodes
    
    def aggregate_fedavg(self, nodes):
        """Standard FedAvg aggregation (baseline)"""
        n_samples = [len(node.X_train) for node in nodes]
        total_samples = sum(n_samples)
        
        # Weighted average
        global_coef = np.zeros_like(nodes[0].model.coef_)
        global_intercept = np.zeros_like(nodes[0].model.intercept_)
        
        for node, n in zip(nodes, n_samples):
            weight = n / total_samples
            global_coef += weight * node.model.coef_
            global_intercept += weight * node.model.intercept_
        
        return {
            'coef': global_coef,
            'intercept': global_intercept
        }
    
    def compute_nsds(self, shap_values_list):
        """Compute Node-Specific Divergence Scores"""
        shap_mean = np.mean(shap_values_list, axis=0)
        nsds_list = []
        
        for shap_values in shap_values_list:
            # Cosine distance
            cos_sim = np.dot(shap_values, shap_mean) / (
                np.linalg.norm(shap_values) * np.linalg.norm(shap_mean) + 1e-10
            )
            nsds = 1 - cos_sim
            nsds_list.append(nsds)
        
        return np.array(nsds_list)
    
    def update_trust_scores(self, nodes, accuracies, nsds_list):
        """Update trust scores using ETASR formula"""
        alpha = self.config.get('trust_alpha', 0.4)
        beta = self.config.get('trust_beta', 0.3)
        gamma = self.config.get('trust_gamma', 0.3)
        
        # Normalize metrics to [0, 1]
        acc_norm = np.array(accuracies)
        nsds_norm = 1 - np.array(nsds_list)  # Lower NSDS = higher trust
        
        # Previous trust scores
        prev_trust = np.array([node.trust_score for node in nodes])
        
        # Combined trust score
        trust_scores = alpha * acc_norm + beta * nsds_norm + gamma * prev_trust
        
        # Normalize to sum to 1
        trust_scores = trust_scores / trust_scores.sum()
        
        # Update node trust scores
        for node, trust in zip(nodes, trust_scores):
            node.trust_score = trust
        
        return trust_scores
    
    def aggregate_fedxchain(self, nodes, X_global, y_global):
        """FedXChain aggregation with PoEx (proposed method)"""
        # Compute SHAP values
        shap_values_list = [node.compute_shap(self.config.get('shap_samples', 10)) 
                           for node in nodes]
        
        # Compute NSDS
        nsds_list = self.compute_nsds(shap_values_list)
        
        # Evaluate local models
        accuracies = [node.evaluate(X_global, y_global)['accuracy'] for node in nodes]
        
        # Update trust scores
        trust_scores = self.update_trust_scores(nodes, accuracies, nsds_list)
        
        # Trust-weighted aggregation
        global_coef = np.zeros_like(nodes[0].model.coef_)
        global_intercept = np.zeros_like(nodes[0].model.intercept_)
        
        for node, trust in zip(nodes, trust_scores):
            global_coef += trust * node.model.coef_
            global_intercept += trust * node.model.intercept_
        
        return {
            'coef': global_coef,
            'intercept': global_intercept,
            'trust_scores': trust_scores,
            'nsds': nsds_list,
            'accuracies': accuracies
        }
    
    def run_single_experiment(self, run_id, method, malicious_config):
        """Run single experiment instance"""
        print(f"\n{'='*70}")
        print(f"Run {run_id + 1} - Method: {method} - Attack: {malicious_config.get('attack_type', 'none')}")
        print(f"{'='*70}")
        
        # Set random seed for reproducibility
        if self.config.get('random_seeds') and run_id < len(self.config['random_seeds']):
            seed = int(self.config['random_seeds'][run_id])
        else:
            seed = int(self.config.get('seed', 42) + run_id)
        np.random.seed(seed)
        
        # Create dataset
        X_train, X_test, y_train, y_test = self.create_dataset()
        
        # Create nodes
        nodes = self.create_nodes(X_train, y_train, X_test, y_test, malicious_config)
        
        # Initialize global model
        global_model = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
        global_model.fit(X_train[:100], y_train[:100])
        
        global_weights = {
            'coef': global_model.coef_.copy(),
            'intercept': global_model.intercept_.copy()
        }
        
        # Training rounds
        round_results = []
        
        for round_num in range(self.config['rounds']):
            print(f"\n[Round {round_num + 1}/{self.config['rounds']}]")
            start_time = time.time()
            
            # Local training
            for node in nodes:
                node.train_local(global_weights, self.config['local_epochs'])
            
            # Aggregation
            if method == 'fedavg':
                global_weights = self.aggregate_fedavg(nodes)
                trust_scores = None
                nsds = None
            else:  # fedxchain
                agg_result = self.aggregate_fedxchain(nodes, X_test, y_test)
                global_weights = {
                    'coef': agg_result['coef'],
                    'intercept': agg_result['intercept']
                }
                trust_scores = agg_result['trust_scores']
                nsds = agg_result['nsds']
            
            # Update global model
            global_model.coef_ = global_weights['coef']
            global_model.intercept_ = global_weights['intercept']
            
            # Evaluate
            y_pred = global_model.predict(X_test)
            try:
                y_score = global_model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_score)
            except Exception:
                auc = None
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': auc,
            }
            
            round_time = time.time() - start_time
            
            # Store results
            # Communication cost estimate (bytes): coef+intercept per client per round
            coef_bytes = int(nodes[0].model.coef_.nbytes) if hasattr(nodes[0].model, 'coef_') else 0
            intercept_bytes = int(nodes[0].model.intercept_.nbytes) if hasattr(nodes[0].model, 'intercept_') else 0
            # Approximate SHAP payload size (float64 per feature) for FedXChain only
            shap_bytes = int(self.config.get('n_features', 0) * 8) if nsds is not None else 0
            comm_bytes_per_client = coef_bytes + intercept_bytes + shap_bytes

            result = {
                'run': run_id,
                'seed': seed,
                'method': method,
                'attack_type': malicious_config.get('attack_type', 'none'),
                'attack_intensity': malicious_config.get('attack_intensity', 0.0),
                'n_malicious': len(malicious_config['malicious_nodes']),
                'round': round_num + 1,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1'],
                'auc_roc': metrics['auc_roc'],
                'time': round_time,
                'comm_bytes_per_client': comm_bytes_per_client,
                'comm_bytes_total': comm_bytes_per_client * self.config['n_clients'],
                'avg_nsds': np.mean(nsds) if nsds is not None else None,
                'min_trust': np.min(trust_scores) if trust_scores is not None else None,
                'max_trust': np.max(trust_scores) if trust_scores is not None else None
            }
            
            round_results.append(result)
            
            print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Time: {round_time:.2f}s")
            if trust_scores is not None:
                print(f"  NSDS: {np.mean(nsds):.4f} | Trust: [{np.min(trust_scores):.3f}, {np.max(trust_scores):.3f}]")
        
        return round_results
    
    def run_full_experiment(self):
        """Run complete experimental suite"""
        print("\n" + "="*80)
        print("FedXChain IEEE Access Experimental Framework")
        print("="*80)
        print(f"\nDataset: {self.config['dataset']}")
        print(f"Clients: {self.config['n_clients']}")
        print(f"Rounds: {self.config['rounds']}")
        print(f"Runs per configuration: {self.config['n_runs']}")
        
        all_results: List[dict] = []

        methods = list(self.config.get('methods', {}).keys()) or ['fedavg', 'fedxchain']
        scenarios = self.config.get('attack_scenarios', {})
        if not scenarios:
            scenarios = {
                'no_attack': {'malicious_nodes': [], 'attack_type': 'none', 'attack_intensity': 0.0}
            }

        # Full factorial: methods × attack_scenarios
        experiments = []
        for scenario_name, scenario_cfg in scenarios.items():
            malicious_nodes = [int(x) for x in scenario_cfg.get('malicious_nodes', [])]
            malicious_nodes = [m for m in malicious_nodes if 0 <= m < self.config['n_clients']]
            experiments.append({
                'scenario': scenario_name,
                'malicious_nodes': malicious_nodes,
                'attack_type': scenario_cfg.get('attack_type', 'none'),
                'attack_intensity': float(scenario_cfg.get('attack_intensity', 0.0)),
            })

        for exp in experiments:
            for method in methods:
                for run in range(self.config['n_runs']):
                    exp_cfg = dict(exp)
                    exp_cfg['method'] = method
                    results = self.run_single_experiment(run, method, exp_cfg)
                    for r in results:
                        r['dataset'] = self.config['dataset']
                        r['scenario'] = exp['scenario']
                    all_results.extend(results)

        return pd.DataFrame(all_results)


def run_ablation_suite(base_config: dict) -> pd.DataFrame:
    """Run ablation studies defined in config (trust weights, malicious ratios, shap samples)."""
    ab = base_config.get('ablation_study', {}) or {}
    results: List[dict] = []

    # Only meaningful for fedxchain
    methods = ['fedxchain']
    base_scenarios = {
        'no_attack': {'malicious_nodes': [], 'attack_type': 'none', 'attack_intensity': 0.0}
    }

    trust_weights = ab.get('trust_weights', [])
    shap_samples = ab.get('shap_samples', [])

    # Malicious ratios -> pick first k nodes as malicious
    malicious_ratios = ab.get('malicious_ratios', [])
    ratio_to_nodes = {}
    for r in malicious_ratios:
        k = int(round(float(r) * base_config['n_clients']))
        k = max(0, min(base_config['n_clients'], k))
        ratio_to_nodes[r] = list(range(k))

    ablations = []
    if trust_weights:
        for tw in trust_weights:
            ablations.append({'ablation': 'trust_weights', 'trust_alpha': float(tw['alpha']), 'trust_beta': float(tw['beta']), 'trust_gamma': float(tw['gamma'])})
    if shap_samples:
        for s in shap_samples:
            ablations.append({'ablation': 'shap_samples', 'shap_samples': int(s)})
    if ratio_to_nodes:
        for r, nodes in ratio_to_nodes.items():
            ablations.append({'ablation': 'malicious_ratio', 'malicious_ratio': float(r), 'malicious_nodes': nodes, 'attack_type': 'label_flip', 'attack_intensity': 0.3})

    for ab_cfg in ablations:
        cfg = json.loads(json.dumps(base_config))
        _deep_update(cfg, ab_cfg)
        exp = FedXChainExperiment(cfg)
        for method in methods:
            for run in range(cfg['n_runs']):
                scenario = base_scenarios['no_attack']
                if ab_cfg.get('ablation') == 'malicious_ratio':
                    scenario = {
                        'malicious_nodes': ab_cfg['malicious_nodes'],
                        'attack_type': ab_cfg.get('attack_type', 'label_flip'),
                        'attack_intensity': float(ab_cfg.get('attack_intensity', 0.3)),
                        'scenario': 'ablation_malicious_ratio',
                    }
                else:
                    scenario = {'malicious_nodes': [], 'attack_type': 'none', 'attack_intensity': 0.0, 'scenario': 'ablation_no_attack'}

                rr = exp.run_single_experiment(run, method, scenario)
                for r in rr:
                    r['dataset'] = cfg['dataset']
                    r['scenario'] = scenario.get('scenario', 'ablation')
                    r['ablation'] = ab_cfg.get('ablation')
                    for k in ['trust_alpha', 'trust_beta', 'trust_gamma', 'shap_samples', 'malicious_ratio']:
                        if k in ab_cfg:
                            r[k] = ab_cfg[k]
                results.extend(rr)

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='IEEE Access Compliant FedXChain Experiments')
    parser.add_argument('--config', type=str, default=str(Path('configs') / 'ieee_experiment_config.yaml'), help='Path to IEEE experiment YAML config')
    parser.add_argument('--dataset', type=str, default=None, choices=['synthetic', 'breast_cancer'], help='Override dataset (optional)')
    parser.add_argument('--n_clients', type=int, default=None, help='Override number of clients (optional)')
    parser.add_argument('--rounds', type=int, default=None, help='Override number of rounds (optional)')
    parser.add_argument('--n_runs', type=int, default=None, help='Override runs per configuration (optional)')
    parser.add_argument('--output', type=str, default=None, help='Override output directory (optional)')
    parser.add_argument('--run_ablation', action='store_true', help='Also run ablation suite from config')
    args = parser.parse_args()

    # Load YAML config
    cfg = load_ieee_config(Path(args.config))
    overrides = {}
    if args.dataset is not None:
        overrides['dataset'] = args.dataset
    if args.n_clients is not None:
        overrides['n_clients'] = args.n_clients
    if args.rounds is not None:
        overrides['rounds'] = args.rounds
    if args.n_runs is not None:
        overrides['n_runs'] = args.n_runs
    if args.output is not None:
        overrides.setdefault('output', {})
        overrides['output']['directory'] = args.output
    _deep_update(cfg, overrides)
    
    # Run experiments
    experiment = FedXChainExperiment(cfg)
    df_results = experiment.run_full_experiment()

    # Save results
    output_dir = Path(cfg.get('output', {}).get('directory', 'results_ieee'))
    output_dir.mkdir(exist_ok=True)

    if cfg.get('reproducibility', {}).get('log_environment', True):
        log_environment(output_dir)

    if cfg.get('reproducibility', {}).get('save_config', True):
        with open(output_dir / 'config_resolved.json', 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'ieee_results_{cfg["dataset"]}_{timestamp}.csv'
    df_results.to_csv(output_file, index=False)

    if args.run_ablation:
        df_ab = run_ablation_suite(cfg)
        ab_file = output_dir / f'ieee_ablation_{cfg["dataset"]}_{timestamp}.csv'
        df_ab.to_csv(ab_file, index=False)
    
    print(f"\n{'='*80}")
    print("Experiment Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_file}")
    print(f"Total experiments run: {len(df_results)}")
    if args.run_ablation:
        print(f"Ablation results saved to: {ab_file}")
    
    # Quick summary
    # Summary on final round only (per run)
    final_round = df_results.groupby(['run', 'method', 'attack_type'])['round'].max().min()
    df_final = df_results[df_results['round'] == final_round]
    summary = df_final.groupby(['method', 'attack_type'])['accuracy'].agg(['mean', 'std']).round(4)
    print("\n" + "="*80)
    print("Summary Statistics (Final Round Accuracy)")
    print("="*80)
    print(summary)

if __name__ == '__main__':
    main()
