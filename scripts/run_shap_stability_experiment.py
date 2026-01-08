#!/usr/bin/env python3
"""
SHAP Background Sample Stability Experiment
============================================
This script runs real experiments to measure:
1. Coefficient of Variation (CV) of SHAP values across different background sample sizes
2. NSDS detection accuracy at each sample size
3. Computation time

This provides REAL experimental data, not estimates.
"""

import numpy as np
import time
import json
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
from scipy.spatial.distance import jensenshannon
from datetime import datetime

# Configuration
SAMPLE_SIZES = [50, 100, 200, 500]
N_RUNS = 5  # Number of runs per sample size for stability measurement
N_CLIENTS = 10
BYZANTINE_FRACTION = 0.3
RANDOM_SEEDS = [42, 43, 44, 45, 46]

RESULTS_DIR = "../results/shap_stability_experiment"
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_nsds(shap1, shap2):
    """Compute Normalized Symmetric Divergence Score using Jensen-Shannon divergence"""
    # Normalize to probability distributions
    p1 = np.abs(shap1) / (np.sum(np.abs(shap1)) + 1e-10)
    p2 = np.abs(shap2) / (np.sum(np.abs(shap2)) + 1e-10)
    # Jensen-Shannon divergence normalized to [0,1]
    js_div = jensenshannon(p1, p2) ** 2  # squared to get divergence
    nsds = js_div / np.log(2)  # normalize
    return min(nsds, 1.0)

def apply_attack(weights, attack_type='sign_flip'):
    """Apply Byzantine attack to model weights"""
    if attack_type == 'sign_flip':
        return {k: -v for k, v in weights.items()}
    elif attack_type == 'noise':
        return {k: v + np.random.normal(0, 0.5, v.shape) for k, v in weights.items()}
    return weights

def run_shap_experiment(n_background_samples, seed):
    """Run single SHAP experiment with given background sample size"""
    np.random.seed(seed)
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train reference model
    ref_model = LogisticRegression(max_iter=1000, random_state=seed)
    ref_model.fit(X_train_scaled, y_train)
    
    # Compute reference SHAP values
    start_time = time.time()
    
    # Select background samples
    bg_indices = np.random.choice(len(X_train_scaled), min(n_background_samples, len(X_train_scaled)), replace=False)
    background = X_train_scaled[bg_indices]
    
    explainer = shap.LinearExplainer(ref_model, background)
    ref_shap = explainer.shap_values(X_test_scaled[:20])  # Use subset for speed
    ref_shap_mean = np.mean(np.abs(ref_shap), axis=0)
    
    shap_time = time.time() - start_time
    
    # Simulate clients
    n_byzantine = int(N_CLIENTS * BYZANTINE_FRACTION)
    n_honest = N_CLIENTS - n_byzantine
    
    honest_nsds = []
    byzantine_nsds = []
    
    # Honest clients - train on same distribution
    for i in range(n_honest):
        client_seed = seed + i + 100
        np.random.seed(client_seed)
        
        # Sample client data
        client_indices = np.random.choice(len(X_train_scaled), len(X_train_scaled)//N_CLIENTS, replace=False)
        X_client = X_train_scaled[client_indices]
        y_client = y_train[client_indices]
        
        # Train client model
        client_model = LogisticRegression(max_iter=1000, random_state=client_seed)
        client_model.fit(X_client, y_client)
        
        # Compute client SHAP
        client_explainer = shap.LinearExplainer(client_model, background)
        client_shap = client_explainer.shap_values(X_test_scaled[:20])
        client_shap_mean = np.mean(np.abs(client_shap), axis=0)
        
        nsds = compute_nsds(client_shap_mean, ref_shap_mean)
        honest_nsds.append(nsds)
    
    # Byzantine clients - train on flipped labels
    for i in range(n_byzantine):
        client_seed = seed + i + 200
        np.random.seed(client_seed)
        
        # Sample client data with flipped labels
        client_indices = np.random.choice(len(X_train_scaled), len(X_train_scaled)//N_CLIENTS, replace=False)
        X_client = X_train_scaled[client_indices]
        y_client = 1 - y_train[client_indices]  # Flip labels
        
        # Train client model
        client_model = LogisticRegression(max_iter=1000, random_state=client_seed)
        client_model.fit(X_client, y_client)
        
        # Compute client SHAP
        client_explainer = shap.LinearExplainer(client_model, background)
        client_shap = client_explainer.shap_values(X_test_scaled[:20])
        client_shap_mean = np.mean(np.abs(client_shap), axis=0)
        
        nsds = compute_nsds(client_shap_mean, ref_shap_mean)
        byzantine_nsds.append(nsds)
    
    return {
        'honest_nsds': honest_nsds,
        'byzantine_nsds': byzantine_nsds,
        'shap_time': shap_time,
        'ref_shap_mean': ref_shap_mean.tolist(),
    }

def compute_cv(values):
    """Compute coefficient of variation"""
    mean = np.mean(values)
    std = np.std(values)
    if mean == 0:
        return 0
    return (std / mean) * 100

def main():
    print("="*70)
    print("SHAP BACKGROUND SAMPLE STABILITY EXPERIMENT")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sample sizes to test: {SAMPLE_SIZES}")
    print(f"Runs per sample size: {N_RUNS}")
    print(f"Clients: {N_CLIENTS}, Byzantine: {BYZANTINE_FRACTION*100}%")
    print()
    
    all_results = []
    
    for n_samples in SAMPLE_SIZES:
        print(f"\n{'='*50}")
        print(f"Testing n_background_samples = {n_samples}")
        print(f"{'='*50}")
        
        run_times = []
        all_honest_nsds = []
        all_byzantine_nsds = []
        all_shap_vectors = []
        
        for run_idx, seed in enumerate(RANDOM_SEEDS):
            print(f"  Run {run_idx+1}/{N_RUNS} (seed={seed})...", end=" ")
            
            result = run_shap_experiment(n_samples, seed)
            
            run_times.append(result['shap_time'])
            all_honest_nsds.extend(result['honest_nsds'])
            all_byzantine_nsds.extend(result['byzantine_nsds'])
            all_shap_vectors.append(result['ref_shap_mean'])
            
            print(f"time={result['shap_time']:.2f}s")
        
        # Compute statistics
        avg_time = np.mean(run_times)
        std_time = np.std(run_times)
        
        # CV of SHAP values across runs
        shap_array = np.array(all_shap_vectors)
        shap_cv_per_feature = []
        for feat_idx in range(shap_array.shape[1]):
            cv = compute_cv(shap_array[:, feat_idx])
            shap_cv_per_feature.append(cv)
        avg_shap_cv = np.mean(shap_cv_per_feature)
        
        # Detection accuracy at threshold 0.5
        threshold = 0.5
        tp = sum(1 for nsds in all_byzantine_nsds if nsds >= threshold)
        fn = sum(1 for nsds in all_byzantine_nsds if nsds < threshold)
        fp = sum(1 for nsds in all_honest_nsds if nsds >= threshold)
        tn = sum(1 for nsds in all_honest_nsds if nsds < threshold)
        
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        
        result_entry = {
            'n_samples': n_samples,
            'cv_percent': round(avg_shap_cv, 2),
            'nsds_accuracy': round(accuracy * 100, 1),
            'avg_time_sec': round(avg_time, 2),
            'std_time_sec': round(std_time, 2),
            'tpr': round(tpr, 3),
            'fpr': round(fpr, 3),
            'honest_nsds_mean': round(np.mean(all_honest_nsds), 4),
            'honest_nsds_std': round(np.std(all_honest_nsds), 4),
            'byzantine_nsds_mean': round(np.mean(all_byzantine_nsds), 4),
            'byzantine_nsds_std': round(np.std(all_byzantine_nsds), 4),
        }
        all_results.append(result_entry)
        
        print(f"\n  Results for n_samples={n_samples}:")
        print(f"    SHAP CV: {avg_shap_cv:.2f}%")
        print(f"    NSDS Accuracy: {accuracy*100:.1f}%")
        print(f"    Avg Time: {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"    TPR: {tpr:.3f}, FPR: {fpr:.3f}")
        print(f"    Honest NSDS: {np.mean(all_honest_nsds):.4f} ± {np.std(all_honest_nsds):.4f}")
        print(f"    Byzantine NSDS: {np.mean(all_byzantine_nsds):.4f} ± {np.std(all_byzantine_nsds):.4f}")
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, 'shap_stability_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'SHAP Background Sample Stability',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'sample_sizes': SAMPLE_SIZES,
                'n_runs': N_RUNS,
                'n_clients': N_CLIENTS,
                'byzantine_fraction': BYZANTINE_FRACTION,
                'seeds': RANDOM_SEEDS,
            },
            'results': all_results,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE (for paper)")
    print("="*70)
    print(f"{'Samples':<10} {'CV(%)':<10} {'Accuracy':<12} {'Time(s)':<10}")
    print("-"*42)
    for r in all_results:
        print(f"{r['n_samples']:<10} {r['cv_percent']:<10} {r['nsds_accuracy']:<12} {r['avg_time_sec']:<10}")
    
    return all_results

if __name__ == "__main__":
    main()
