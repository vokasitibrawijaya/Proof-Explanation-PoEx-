#!/usr/bin/env python3
"""
Quick CIFAR-10 CNN Experiment for PoEx Validation - Fast Version

Reduced configuration for quick demonstration while still showing
meaningful NSDS separation between honest and Byzantine clients.
"""

import numpy as np
import pandas as pd
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import mannwhitneyu
from scipy.special import rel_entr

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'cifar10_cnn_experiment')
os.makedirs(RESULTS_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================================
# FAST CONFIGURATION
# ==============================================================================

CONFIG = {
    'n_clients': 10,           # Reduced for speed
    'n_rounds': 5,             # Fewer rounds
    'n_seeds': 2,              # Just 2 seeds
    'byzantine_fraction': 0.3,
    'local_epochs': 2,         # Fewer epochs
    'batch_size': 64,          # Larger batch
    'learning_rate': 0.01,
    'threshold': 0.5,
    'thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'subset_size': 5000,       # Use subset of CIFAR-10
}


# ==============================================================================
# SIMPLE CNN MODEL
# ==============================================================================

class SimpleCNN(nn.Module):
    """Simplified CNN for faster training"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def get_weights(self):
        return {name: param.data.cpu().numpy().copy() 
                for name, param in self.named_parameters()}
    
    def set_weights(self, weights):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    param.data = torch.tensor(weights[name], dtype=torch.float32, device=param.device)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_cifar10_subset():
    """Load a subset of CIFAR-10"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    # Use subset for speed
    subset_indices = np.random.choice(len(trainset), CONFIG['subset_size'], replace=False)
    trainset = Subset(trainset, subset_indices.tolist())
    
    return trainset, testset


def dirichlet_partition(n_samples, n_clients, n_classes=10, alpha=0.5, seed=42):
    """Non-IID partition using Dirichlet distribution"""
    np.random.seed(seed)
    
    # Generate random labels for simulation
    labels = np.random.randint(0, n_classes, n_samples)
    
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_indices = [np.where(labels == k)[0] for k in range(n_classes)]
    
    client_indices = [[] for _ in range(n_clients)]
    
    for c, indices in enumerate(class_indices):
        if len(indices) == 0:
            continue
        np.random.shuffle(indices)
        proportions = label_distribution[c]
        proportions = proportions / proportions.sum()
        splits = (proportions * len(indices)).astype(int)
        splits[-1] = len(indices) - splits[:-1].sum()
        
        idx_start = 0
        for client_id, n_samples_client in enumerate(splits):
            if n_samples_client > 0 and idx_start + n_samples_client <= len(indices):
                client_indices[client_id].extend(indices[idx_start:idx_start + n_samples_client].tolist())
            idx_start += n_samples_client
    
    return [np.array(idx) for idx in client_indices]


# ==============================================================================
# WEIGHT IMPORTANCE & NSDS
# ==============================================================================

def compute_weight_importance(old_weights, new_weights):
    """Compute importance based on weight changes"""
    importance = []
    for key in sorted(old_weights.keys()):
        if 'weight' in key:
            diff = np.abs(new_weights[key] - old_weights[key])
            importance.extend([
                diff.mean(), 
                diff.std(), 
                diff.max(),
                np.percentile(diff.flatten(), 75),
                np.percentile(diff.flatten(), 95),
            ])
    return np.array(importance)


def compute_nsds(importance_local, importance_ref):
    """Compute NSDS using Jensen-Shannon divergence"""
    eps = 1e-10
    p = np.abs(importance_local) + eps
    q = np.abs(importance_ref) + eps
    p = p / p.sum()
    q = q / q.sum()
    
    m = 0.5 * (p + q)
    js_div = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
    nsds = js_div / np.log(2)
    return min(nsds, 1.0)


# ==============================================================================
# ATTACKS
# ==============================================================================

def apply_attack(weights, attack_type='sign_flip'):
    """Apply Byzantine attack to weights"""
    attacked = {}
    for key, value in weights.items():
        if attack_type == 'sign_flip':
            attacked[key] = -1.5 * value
        elif attack_type == 'noise':
            noise = np.random.normal(0, 0.5, value.shape)
            attacked[key] = value + noise
        elif attack_type == 'scaling':
            attacked[key] = 5.0 * value
        else:
            attacked[key] = value
    return attacked


# ==============================================================================
# AGGREGATION
# ==============================================================================

def fedavg_aggregate(updates):
    """FedAvg aggregation"""
    aggregated = {}
    for key in updates[0].keys():
        aggregated[key] = np.mean([u[key] for u in updates], axis=0)
    return aggregated


def poex_aggregate(updates, nsds_scores, threshold):
    """PoEx aggregation"""
    accepted = [i for i, nsds in enumerate(nsds_scores) if nsds < threshold]
    rejected = [i for i, nsds in enumerate(nsds_scores) if nsds >= threshold]
    
    if len(accepted) == 0:
        best_idx = np.argmin(nsds_scores)
        accepted = [best_idx]
        rejected = [i for i in range(len(updates)) if i != best_idx]
    
    aggregated = {}
    for key in updates[0].keys():
        aggregated[key] = np.mean([updates[i][key] for i in accepted], axis=0)
    
    return aggregated, accepted, rejected


# ==============================================================================
# EXPERIMENT
# ==============================================================================

def run_experiment(seed=42):
    """Run one experiment seed"""
    print(f"\n{'='*60}")
    print(f"Running experiment with seed={seed}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    print("Loading CIFAR-10 subset...")
    trainset, testset = load_cifar10_subset()
    testloader = DataLoader(testset, batch_size=100, shuffle=False)
    
    # Partition data
    n_samples = len(trainset)
    client_indices = dirichlet_partition(n_samples, CONFIG['n_clients'], alpha=0.5, seed=seed)
    
    # Determine Byzantine clients
    n_byzantine = int(CONFIG['n_clients'] * CONFIG['byzantine_fraction'])
    byzantine_clients = set(np.random.choice(CONFIG['n_clients'], n_byzantine, replace=False))
    print(f"Byzantine clients: {sorted(byzantine_clients)}")
    
    # Initialize model
    global_model = SimpleCNN().to(DEVICE)
    global_weights = global_model.get_weights()
    
    # Storage
    all_nsds_honest = []
    all_nsds_byzantine = []
    round_results = []
    
    for round_idx in range(CONFIG['n_rounds']):
        print(f"\nRound {round_idx + 1}/{CONFIG['n_rounds']}")
        
        client_updates = []
        client_importance = []
        is_byzantine = []
        
        for client_id in range(CONFIG['n_clients']):
            # Create client model
            client_model = SimpleCNN().to(DEVICE)
            client_model.set_weights(global_weights)
            old_weights = client_model.get_weights()
            
            # Get client data
            client_idx = client_indices[client_id]
            if len(client_idx) < 10:
                client_updates.append({k: np.zeros_like(v) for k, v in global_weights.items()})
                client_importance.append(np.zeros(25))  # Placeholder
                is_byzantine.append(client_id in byzantine_clients)
                continue
            
            client_subset = Subset(trainset.dataset if hasattr(trainset, 'dataset') else trainset, 
                                   client_idx.tolist())
            client_loader = DataLoader(client_subset, batch_size=CONFIG['batch_size'], shuffle=True)
            
            # Local training
            optimizer = optim.SGD(client_model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            
            client_model.train()
            for epoch in range(CONFIG['local_epochs']):
                for batch_idx, (data, target) in enumerate(client_loader):
                    if batch_idx >= 10:  # Limit batches for speed
                        break
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = client_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Get update
            new_weights = client_model.get_weights()
            update = {key: new_weights[key] - old_weights[key] for key in new_weights.keys()}
            
            # Apply attack if Byzantine
            if client_id in byzantine_clients:
                attack_type = np.random.choice(['sign_flip', 'noise', 'scaling'])
                update = apply_attack(update, attack_type)
            
            # Compute importance
            importance = compute_weight_importance(old_weights, 
                                                   {k: old_weights[k] + update[k] for k in update.keys()})
            
            client_updates.append(update)
            client_importance.append(importance)
            is_byzantine.append(client_id in byzantine_clients)
        
        # Reference importance
        reference_importance = np.mean([imp for imp in client_importance if len(imp) > 0], axis=0)
        
        # Compute NSDS
        nsds_scores = []
        for i, imp in enumerate(client_importance):
            if len(imp) == 0:
                nsds = 0.5
            else:
                nsds = compute_nsds(imp, reference_importance)
            nsds_scores.append(nsds)
            
            if is_byzantine[i]:
                all_nsds_byzantine.append(nsds)
            else:
                all_nsds_honest.append(nsds)
        
        # PoEx aggregation
        aggregated, accepted, rejected = poex_aggregate(client_updates, nsds_scores, CONFIG['threshold'])
        
        # Update global model
        new_global_weights = {}
        for key in global_weights.keys():
            new_global_weights[key] = global_weights[key] + aggregated[key]
        global_weights = new_global_weights
        global_model.set_weights(global_weights)
        
        # Evaluate
        global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        
        # Detection metrics
        tp = sum(1 for i in rejected if is_byzantine[i])
        fp = sum(1 for i in rejected if not is_byzantine[i])
        fn = sum(1 for i in accepted if is_byzantine[i])
        tn = sum(1 for i in accepted if not is_byzantine[i])
        
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        
        honest_nsds = [nsds_scores[i] for i in range(len(nsds_scores)) if not is_byzantine[i]]
        byz_nsds = [nsds_scores[i] for i in range(len(nsds_scores)) if is_byzantine[i]]
        
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  NSDS - Honest: {np.mean(honest_nsds):.4f}, Byzantine: {np.mean(byz_nsds):.4f}")
        print(f"  TPR: {tpr:.3f}, FPR: {fpr:.3f}, Accepted: {len(accepted)}, Rejected: {len(rejected)}")
        
        round_results.append({
            'round': round_idx + 1,
            'accuracy': accuracy,
            'tpr': tpr,
            'fpr': fpr,
        })
    
    return {
        'seed': seed,
        'final_accuracy': round_results[-1]['accuracy'],
        'round_results': round_results,
        'nsds_honest': all_nsds_honest,
        'nsds_byzantine': all_nsds_byzantine,
    }


def analyze_thresholds(all_nsds_honest, all_nsds_byzantine):
    """Analyze TPR/FPR across thresholds"""
    results = []
    for tau in CONFIG['thresholds']:
        tp = sum(1 for nsds in all_nsds_byzantine if nsds >= tau)
        fn = sum(1 for nsds in all_nsds_byzantine if nsds < tau)
        fp = sum(1 for nsds in all_nsds_honest if nsds >= tau)
        tn = sum(1 for nsds in all_nsds_honest if nsds < tau)
        
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * tpr / max(precision + tpr, 1e-10)
        
        results.append({
            'threshold': tau,
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,
            'f1': f1,
        })
    return results


def create_visualizations(all_honest, all_byzantine, threshold_results):
    """Create figures"""
    print("\nCreating visualizations...")
    
    # Figure 1: NSDS Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    bins = np.linspace(0, 1, 25)
    ax1.hist(all_honest, bins=bins, alpha=0.7, label=f'Honest (n={len(all_honest)})', color='green', density=True)
    ax1.hist(all_byzantine, bins=bins, alpha=0.7, label=f'Byzantine (n={len(all_byzantine)})', color='red', density=True)
    ax1.axvline(x=CONFIG['threshold'], color='black', linestyle='--', linewidth=2, label=f'τ={CONFIG["threshold"]}')
    ax1.set_xlabel('NSDS Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('NSDS Distribution: CIFAR-10 CNN (Non-IID)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Boxplot
    ax2 = axes[1]
    bp = ax2.boxplot([all_honest, all_byzantine], labels=['Honest', 'Byzantine'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.axhline(y=CONFIG['threshold'], color='black', linestyle='--', linewidth=2, label=f'τ={CONFIG["threshold"]}')
    ax2.set_ylabel('NSDS Score', fontsize=12)
    ax2.set_title('NSDS Boxplot: CIFAR-10 CNN', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cifar10_nsds_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: TPR/FPR Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    
    thresholds = [r['threshold'] for r in threshold_results]
    tprs = [r['tpr'] for r in threshold_results]
    fprs = [r['fpr'] for r in threshold_results]
    
    ax.plot(thresholds, tprs, 'b-o', label='TPR (True Positive Rate)', linewidth=2, markersize=8)
    ax.plot(thresholds, fprs, 'r-s', label='FPR (False Positive Rate)', linewidth=2, markersize=8)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default τ=0.5')
    ax.set_xlabel('Threshold (τ)', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('TPR/FPR vs Threshold: CIFAR-10 CNN', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cifar10_tpr_fpr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved to {FIGURES_DIR}")


def main():
    """Main function"""
    print("="*70)
    print("CIFAR-10 CNN Experiment - Fast Version")
    print("="*70)
    
    start_time = time.time()
    
    # Run experiments
    all_results = []
    for seed in range(42, 42 + CONFIG['n_seeds']):
        result = run_experiment(seed=seed)
        all_results.append(result)
    
    # Aggregate
    all_honest = []
    all_byzantine = []
    final_accuracies = []
    
    for result in all_results:
        all_honest.extend(result['nsds_honest'])
        all_byzantine.extend(result['nsds_byzantine'])
        final_accuracies.append(result['final_accuracy'])
    
    # Threshold analysis
    threshold_results = analyze_thresholds(all_honest, all_byzantine)
    
    # Print results
    print("\n" + "="*70)
    print("CIFAR-10 CNN EXPERIMENT RESULTS")
    print("="*70)
    
    print(f"\nNSDS Distribution:")
    print(f"  Honest clients:    mean={np.mean(all_honest):.4f} ± std={np.std(all_honest):.4f}")
    print(f"  Byzantine clients: mean={np.mean(all_byzantine):.4f} ± std={np.std(all_byzantine):.4f}")
    print(f"  Separation: {abs(np.mean(all_byzantine) - np.mean(all_honest)):.4f}")
    
    # Mann-Whitney U test
    if len(all_honest) > 0 and len(all_byzantine) > 0:
        stat, pvalue = mannwhitneyu(all_honest, all_byzantine, alternative='two-sided')
        print(f"  Mann-Whitney U test: p-value={pvalue:.6f}")
        if pvalue < 0.05:
            print(f"  *** STATISTICALLY SIGNIFICANT (p < 0.05) ***")
    
    print(f"\nFinal Accuracy: {np.mean(final_accuracies):.2f}% ± {np.std(final_accuracies):.2f}%")
    
    print(f"\nThreshold Analysis (TPR/FPR):")
    for r in threshold_results:
        print(f"  τ={r['threshold']:.1f}: TPR={r['tpr']:.3f}, FPR={r['fpr']:.3f}, F1={r['f1']:.3f}")
    
    # Create visualizations
    create_visualizations(all_honest, all_byzantine, threshold_results)
    
    # Save results
    pd.DataFrame(threshold_results).to_csv(os.path.join(RESULTS_DIR, 'cifar10_threshold_analysis.csv'), index=False)
    
    nsds_stats = {
        'honest_mean': float(np.mean(all_honest)),
        'honest_std': float(np.std(all_honest)),
        'byzantine_mean': float(np.mean(all_byzantine)),
        'byzantine_std': float(np.std(all_byzantine)),
        'separation': float(abs(np.mean(all_byzantine) - np.mean(all_honest))),
        'pvalue': float(pvalue) if len(all_honest) > 0 and len(all_byzantine) > 0 else None,
        'final_accuracy_mean': float(np.mean(final_accuracies)),
        'final_accuracy_std': float(np.std(final_accuracies)),
    }
    
    with open(os.path.join(RESULTS_DIR, 'cifar10_nsds_stats.json'), 'w') as f:
        json.dump(nsds_stats, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.2f} minutes")
    print(f"Results saved to: {RESULTS_DIR}")
    
    return nsds_stats, threshold_results


if __name__ == "__main__":
    nsds_stats, threshold_results = main()
