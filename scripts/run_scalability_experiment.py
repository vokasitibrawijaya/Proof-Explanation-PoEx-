#!/usr/bin/env python3
"""
Scalability Experiment: n=100 Clients
=====================================
This script runs REAL experiments with 100 clients to measure actual performance,
not theoretical projections.

Measures:
1. SHAP computation time per client
2. NSDS computation time (total for all clients)
3. Aggregation time for different methods
4. Total round time
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
CLIENT_COUNTS = [20, 50, 100]
BYZANTINE_FRACTION = 0.3
N_ROUNDS = 3
RANDOM_SEED = 42

RESULTS_DIR = "../results/scalability_experiment"
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_nsds(shap1, shap2):
    """Compute NSDS using Jensen-Shannon divergence"""
    p1 = np.abs(shap1) / (np.sum(np.abs(shap1)) + 1e-10)
    p2 = np.abs(shap2) / (np.sum(np.abs(shap2)) + 1e-10)
    js_div = jensenshannon(p1, p2) ** 2
    nsds = js_div / np.log(2)
    return min(nsds, 1.0)

def krum_aggregate(updates, f):
    """Krum aggregation - O(n^2) complexity"""
    n = len(updates)
    if n == 0:
        return None
    
    # Convert to arrays
    update_arrays = []
    for u in updates:
        flat = np.concatenate([v.flatten() for v in u.values()])
        update_arrays.append(flat)
    update_arrays = np.array(update_arrays)
    
    # Compute pairwise distances - O(n^2)
    scores = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(update_arrays[i] - update_arrays[j])
                distances.append(dist)
        distances.sort()
        # Sum of n-f-2 closest distances
        score = sum(distances[:n - f - 2]) if len(distances) >= n - f - 2 else sum(distances)
        scores.append(score)
    
    # Select update with minimum score
    selected_idx = np.argmin(scores)
    return updates[selected_idx]

def trimmed_mean_aggregate(updates, trim_ratio=0.1):
    """TrimmedMean aggregation - O(n log n) per coordinate"""
    if len(updates) == 0:
        return None
    
    # Get keys from first update
    keys = list(updates[0].keys())
    result = {}
    
    for key in keys:
        stacked = np.stack([u[key] for u in updates])
        n = len(stacked)
        trim_count = int(n * trim_ratio)
        
        # Sort and trim along axis 0
        sorted_vals = np.sort(stacked, axis=0)
        if trim_count > 0:
            trimmed = sorted_vals[trim_count:-trim_count]
        else:
            trimmed = sorted_vals
        
        result[key] = np.mean(trimmed, axis=0)
    
    return result

def poex_aggregate(updates, nsds_scores, threshold=0.5):
    """PoEx aggregation - O(n) complexity"""
    if len(updates) == 0:
        return None
    
    # Filter by threshold
    accepted_indices = [i for i, nsds in enumerate(nsds_scores) if nsds < threshold]
    
    if len(accepted_indices) == 0:
        # Fallback: accept the one with lowest NSDS
        accepted_indices = [np.argmin(nsds_scores)]
    
    # Simple average of accepted updates
    keys = list(updates[0].keys())
    result = {}
    
    for key in keys:
        result[key] = np.mean([updates[i][key] for i in accepted_indices], axis=0)
    
    return result

def run_scalability_experiment(n_clients, seed):
    """Run experiment with specified number of clients"""
    np.random.seed(seed)
    
    print(f"\n  Running with {n_clients} clients...")
    
    # Load and prepare data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Background samples for SHAP
    n_background = 100
    bg_indices = np.random.choice(len(X_train_scaled), n_background, replace=False)
    background = X_train_scaled[bg_indices]
    
    # Train reference model
    ref_model = LogisticRegression(max_iter=1000, random_state=seed)
    ref_model.fit(X_train_scaled, y_train)
    
    ref_explainer = shap.LinearExplainer(ref_model, background)
    ref_shap = ref_explainer.shap_values(X_test_scaled[:20])
    ref_shap_mean = np.mean(np.abs(ref_shap), axis=0)
    
    n_byzantine = int(n_clients * BYZANTINE_FRACTION)
    f = n_byzantine
    
    # Simulate all clients
    client_updates = []
    client_shap_vectors = []
    shap_times = []
    
    for i in range(n_clients):
        client_seed = seed + i
        np.random.seed(client_seed)
        
        is_byzantine = i < n_byzantine
        
        # Sample client data
        client_size = max(len(X_train_scaled) // n_clients, 10)
        client_indices = np.random.choice(len(X_train_scaled), client_size, replace=True)
        X_client = X_train_scaled[client_indices]
        
        if is_byzantine:
            y_client = 1 - y_train[client_indices]  # Flip labels
        else:
            y_client = y_train[client_indices]
        
        # Train client model
        client_model = LogisticRegression(max_iter=1000, random_state=client_seed)
        try:
            client_model.fit(X_client, y_client)
        except:
            # If fitting fails, use reference model
            client_model = ref_model
        
        # Compute SHAP - measure time
        start_shap = time.time()
        client_explainer = shap.LinearExplainer(client_model, background)
        client_shap = client_explainer.shap_values(X_test_scaled[:20])
        client_shap_mean = np.mean(np.abs(client_shap), axis=0)
        shap_time = time.time() - start_shap
        shap_times.append(shap_time)
        
        # Store update (simulate weight updates)
        update = {
            'coef': client_model.coef_.copy(),
            'intercept': client_model.intercept_.copy(),
        }
        client_updates.append(update)
        client_shap_vectors.append(client_shap_mean)
    
    # Compute NSDS for all clients - measure time
    start_nsds = time.time()
    nsds_scores = []
    for shap_vec in client_shap_vectors:
        nsds = compute_nsds(shap_vec, ref_shap_mean)
        nsds_scores.append(nsds)
    nsds_time = time.time() - start_nsds
    
    # Test different aggregation methods
    
    # Krum - O(n^2)
    start_krum = time.time()
    _ = krum_aggregate(client_updates, f)
    krum_time = time.time() - start_krum
    
    # TrimmedMean - O(n log n)
    start_trim = time.time()
    _ = trimmed_mean_aggregate(client_updates)
    trim_time = time.time() - start_trim
    
    # PoEx - O(n)
    start_poex = time.time()
    _ = poex_aggregate(client_updates, nsds_scores)
    poex_time = time.time() - start_poex
    
    # Total times
    avg_shap_time = np.mean(shap_times)
    total_shap_time = sum(shap_times)
    
    return {
        'n_clients': n_clients,
        'avg_shap_time_per_client': avg_shap_time,
        'total_shap_time': total_shap_time,
        'nsds_computation_time': nsds_time,
        'krum_aggregation_time': krum_time,
        'trimmedmean_aggregation_time': trim_time,
        'poex_aggregation_time': poex_time,
        'total_round_time_poex': total_shap_time + nsds_time + poex_time,
        'total_round_time_krum': total_shap_time + krum_time,
        'total_round_time_trimmed': total_shap_time + trim_time,
    }

def main():
    print("="*70)
    print("SCALABILITY EXPERIMENT: n=20, 50, 100 CLIENTS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Client counts: {CLIENT_COUNTS}")
    print(f"Byzantine fraction: {BYZANTINE_FRACTION*100}%")
    print(f"Rounds per config: {N_ROUNDS}")
    print()
    
    all_results = []
    
    for n_clients in CLIENT_COUNTS:
        print(f"\n{'='*50}")
        print(f"Testing n_clients = {n_clients}")
        print(f"{'='*50}")
        
        round_results = []
        
        for round_idx in range(N_ROUNDS):
            seed = RANDOM_SEED + round_idx
            result = run_scalability_experiment(n_clients, seed)
            round_results.append(result)
            
            print(f"  Round {round_idx+1}: SHAP={result['avg_shap_time_per_client']:.3f}s/client, "
                  f"NSDS={result['nsds_computation_time']*1000:.1f}ms, "
                  f"Krum={result['krum_aggregation_time']*1000:.1f}ms, "
                  f"PoEx={result['poex_aggregation_time']*1000:.1f}ms")
        
        # Average across rounds
        avg_result = {
            'n_clients': n_clients,
            'avg_shap_time_per_client': np.mean([r['avg_shap_time_per_client'] for r in round_results]),
            'std_shap_time_per_client': np.std([r['avg_shap_time_per_client'] for r in round_results]),
            'nsds_computation_ms': np.mean([r['nsds_computation_time'] for r in round_results]) * 1000,
            'krum_aggregation_ms': np.mean([r['krum_aggregation_time'] for r in round_results]) * 1000,
            'trimmedmean_aggregation_ms': np.mean([r['trimmedmean_aggregation_time'] for r in round_results]) * 1000,
            'poex_aggregation_ms': np.mean([r['poex_aggregation_time'] for r in round_results]) * 1000,
            'total_round_time_poex': np.mean([r['total_round_time_poex'] for r in round_results]),
            'total_round_time_krum': np.mean([r['total_round_time_krum'] for r in round_results]),
        }
        all_results.append(avg_result)
        
        print(f"\n  AVERAGE for n={n_clients}:")
        print(f"    SHAP/client: {avg_result['avg_shap_time_per_client']:.3f}s ± {avg_result['std_shap_time_per_client']:.3f}s")
        print(f"    NSDS total: {avg_result['nsds_computation_ms']:.2f}ms")
        print(f"    Krum: {avg_result['krum_aggregation_ms']:.2f}ms")
        print(f"    TrimmedMean: {avg_result['trimmedmean_aggregation_ms']:.2f}ms")
        print(f"    PoEx: {avg_result['poex_aggregation_ms']:.2f}ms")
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, 'scalability_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Scalability Experiment',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'client_counts': CLIENT_COUNTS,
                'byzantine_fraction': BYZANTINE_FRACTION,
                'n_rounds': N_ROUNDS,
            },
            'results': all_results,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    
    # Print summary table for paper
    print("\n" + "="*70)
    print("SUMMARY TABLE (for paper)")
    print("="*70)
    print(f"{'n_clients':<12} {'SHAP/client':<14} {'NSDS(ms)':<12} {'Krum(ms)':<12} {'PoEx(ms)':<12}")
    print("-"*62)
    for r in all_results:
        print(f"{r['n_clients']:<12} {r['avg_shap_time_per_client']:.3f}s{'':<7} "
              f"{r['nsds_computation_ms']:.2f}{'':<6} "
              f"{r['krum_aggregation_ms']:.2f}{'':<6} "
              f"{r['poex_aggregation_ms']:.2f}")
    
    return all_results

if __name__ == "__main__":
    main()
