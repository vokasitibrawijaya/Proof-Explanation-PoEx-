#!/usr/bin/env python3
"""
CIFAR-10 Full CNN Experiment for PoEx Validation

This experiment addresses the reviewer concern about NSDS discriminative power
by running PoEx on a more complex model (full CNN) on real CIFAR-10 data.

Goal: Demonstrate that on complex models, NSDS provides better separation
between honest and Byzantine clients, with non-zero TPR/FPR.

Author: FedXChain Research Team
Date: January 2026
"""

import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
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
# CONFIGURATION
# ==============================================================================

CONFIG = {
    'n_clients': 20,
    'n_rounds': 10,
    'n_seeds': 3,              # 3 seeds for faster but still valid results
    'byzantine_fraction': 0.3,  # 30% Byzantine
    'local_epochs': 3,
    'batch_size': 32,
    'learning_rate': 0.01,
    'threshold': 0.5,
    'thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}


# ==============================================================================
# CNN MODEL - More Complex for Better NSDS Separation
# ==============================================================================

class CIFAR10CNN(nn.Module):
    """Full CNN for CIFAR-10 with more capacity"""
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Conv Block 3
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # FC layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 2: 16x16 -> 8x8
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Block 3: 8x8 -> 4x4
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        
        # FC
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def get_weights(self):
        return {name: param.data.cpu().numpy().copy() 
                for name, param in self.named_parameters()}
    
    def set_weights(self, weights):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    param.data = torch.tensor(weights[name], dtype=torch.float32, device=param.device)
    
    def get_feature_vector(self, x):
        """Extract feature vector for SHAP-like analysis"""
        with torch.no_grad():
            # Block 1
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            # Block 2
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            # Block 3
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.pool(F.relu(self.bn6(self.conv6(x))))
            # Flatten
            x = x.view(-1, 256 * 4 * 4)
            # FC1 features
            x = F.relu(self.fc1(x))
        return x


# ==============================================================================
# DATA LOADING AND PARTITIONING
# ==============================================================================

def load_cifar10():
    """Load CIFAR-10 with augmentation"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    
    return trainset, testset


def dirichlet_partition(dataset, n_clients, alpha=0.5, seed=42):
    """Non-IID partition using Dirichlet distribution"""
    np.random.seed(seed)
    labels = np.array(dataset.targets)
    n_classes = 10
    
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_indices = [np.where(labels == k)[0] for k in range(n_classes)]
    
    client_indices = [[] for _ in range(n_clients)]
    
    for c, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        proportions = label_distribution[c]
        proportions = proportions / proportions.sum()
        splits = (proportions * len(indices)).astype(int)
        splits[-1] = len(indices) - splits[:-1].sum()
        
        idx_start = 0
        for client_id, n_samples in enumerate(splits):
            if n_samples > 0:
                client_indices[client_id].extend(indices[idx_start:idx_start + n_samples].tolist())
            idx_start += n_samples
    
    return [np.array(idx) for idx in client_indices]


# ==============================================================================
# GRADIENT-BASED FEATURE IMPORTANCE (SHAP-like for CNNs)
# ==============================================================================

def compute_gradient_importance(model, dataloader, device, num_samples=50):
    """
    Compute gradient-based feature importance for CNN
    This serves as SHAP-like explanation for model updates
    """
    model.eval()
    model.to(device)
    
    # Collect gradients of loss w.r.t. FC layer weights
    fc_grad_sum = None
    count = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        if count >= num_samples:
            break
            
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Get gradients from fc1 layer (most informative for our purpose)
        fc1_grad = model.fc1.weight.grad.abs().mean(dim=0).cpu().numpy()
        
        if fc_grad_sum is None:
            fc_grad_sum = fc1_grad
        else:
            fc_grad_sum += fc1_grad
        
        count += data.size(0)
    
    # Normalize
    importance = fc_grad_sum / max(count, 1)
    return importance


def compute_weight_difference_importance(old_weights, new_weights):
    """
    Compute importance based on weight differences (update magnitude per layer)
    This creates a feature vector representing how the model changed
    """
    importance = []
    for key in sorted(old_weights.keys()):
        if 'weight' in key:  # Only consider weight layers
            diff = np.abs(new_weights[key] - old_weights[key])
            # Get mean and std of changes per layer
            importance.extend([diff.mean(), diff.std(), diff.max()])
    
    return np.array(importance)


# ==============================================================================
# ATTACK IMPLEMENTATIONS
# ==============================================================================

def apply_sign_flip_attack(weights, scale=1.0):
    """Sign flip attack on model weights"""
    attacked = {}
    for key, value in weights.items():
        attacked[key] = -scale * value
    return attacked


def apply_noise_attack(weights, noise_scale=0.5):
    """Gaussian noise attack"""
    attacked = {}
    for key, value in weights.items():
        noise = np.random.normal(0, noise_scale, value.shape)
        attacked[key] = value + noise
    return attacked


def apply_scaling_attack(weights, scale=10.0):
    """Scaling attack - amplify updates"""
    attacked = {}
    for key, value in weights.items():
        attacked[key] = scale * value
    return attacked


# ==============================================================================
# NSDS COMPUTATION
# ==============================================================================

def compute_nsds(importance_local, importance_ref):
    """Compute Normalized Symmetric Divergence Score using Jensen-Shannon"""
    eps = 1e-10
    p = np.abs(importance_local) + eps
    q = np.abs(importance_ref) + eps
    p = p / p.sum()
    q = q / q.sum()
    
    m = 0.5 * (p + q)
    js_div = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
    nsds = js_div / np.log(2)  # Normalize to [0, 1]
    return min(nsds, 1.0)


# ==============================================================================
# AGGREGATION
# ==============================================================================

def fedavg_aggregate(updates):
    """Simple FedAvg aggregation"""
    aggregated = {}
    for key in updates[0].keys():
        aggregated[key] = np.mean([u[key] for u in updates], axis=0)
    return aggregated


def poex_aggregate(updates, nsds_scores, threshold):
    """PoEx aggregation based on NSDS threshold"""
    accepted = [i for i, nsds in enumerate(nsds_scores) if nsds < threshold]
    rejected = [i for i, nsds in enumerate(nsds_scores) if nsds >= threshold]
    
    if len(accepted) == 0:
        # Fallback: accept the one with lowest NSDS
        best_idx = np.argmin(nsds_scores)
        accepted = [best_idx]
        rejected = [i for i in range(len(updates)) if i != best_idx]
    
    aggregated = {}
    for key in updates[0].keys():
        aggregated[key] = np.mean([updates[i][key] for i in accepted], axis=0)
    
    return aggregated, accepted, rejected


# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

def run_experiment(seed=42):
    """Run one seed of the CIFAR-10 CNN experiment"""
    print(f"\n{'='*60}")
    print(f"Running CIFAR-10 CNN Experiment with seed={seed}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    print("Loading CIFAR-10...")
    trainset, testset = load_cifar10()
    testloader = DataLoader(testset, batch_size=100, shuffle=False)
    
    # Partition data (Non-IID)
    print("Partitioning data (Non-IID, alpha=0.5)...")
    client_indices = dirichlet_partition(trainset, CONFIG['n_clients'], alpha=0.5, seed=seed)
    
    # Determine Byzantine clients
    n_byzantine = int(CONFIG['n_clients'] * CONFIG['byzantine_fraction'])
    byzantine_clients = set(np.random.choice(CONFIG['n_clients'], n_byzantine, replace=False))
    print(f"Byzantine clients: {sorted(byzantine_clients)}")
    
    # Initialize global model
    global_model = CIFAR10CNN().to(DEVICE)
    global_weights = global_model.get_weights()
    
    # Reference importance (from initial model)
    small_loader = DataLoader(Subset(trainset, list(range(500))), batch_size=32, shuffle=True)
    reference_importance = compute_gradient_importance(global_model, small_loader, DEVICE)
    
    # Storage for NSDS values
    all_nsds_honest = []
    all_nsds_byzantine = []
    
    # Storage for per-round results
    round_results = []
    
    for round_idx in range(CONFIG['n_rounds']):
        print(f"\nRound {round_idx + 1}/{CONFIG['n_rounds']}")
        
        client_updates = []
        client_importance = []
        is_byzantine = []
        
        for client_id in range(CONFIG['n_clients']):
            # Create client model
            client_model = CIFAR10CNN().to(DEVICE)
            client_model.set_weights(global_weights)
            old_weights = client_model.get_weights()
            
            # Get client data
            client_data = Subset(trainset, client_indices[client_id].tolist())
            if len(client_data) < 10:
                # Skip clients with too little data
                client_updates.append(global_weights.copy())
                client_importance.append(reference_importance.copy())
                is_byzantine.append(client_id in byzantine_clients)
                continue
            
            client_loader = DataLoader(client_data, batch_size=CONFIG['batch_size'], shuffle=True)
            
            # Local training
            optimizer = optim.SGD(client_model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            
            client_model.train()
            for epoch in range(CONFIG['local_epochs']):
                for data, target in client_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = client_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Get new weights
            new_weights = client_model.get_weights()
            
            # Compute update
            update = {}
            for key in new_weights.keys():
                update[key] = new_weights[key] - old_weights[key]
            
            # Apply attack if Byzantine
            if client_id in byzantine_clients:
                attack_type = np.random.choice(['sign_flip', 'noise', 'scaling'])
                if attack_type == 'sign_flip':
                    update = apply_sign_flip_attack(update, scale=1.5)
                elif attack_type == 'noise':
                    update = apply_noise_attack(update, noise_scale=0.5)
                else:
                    update = apply_scaling_attack(update, scale=5.0)
            
            # Compute importance (SHAP-like)
            importance = compute_weight_difference_importance(old_weights, 
                                                              {k: old_weights[k] + update[k] for k in update.keys()})
            
            client_updates.append(update)
            client_importance.append(importance)
            is_byzantine.append(client_id in byzantine_clients)
        
        # Ensure reference importance matches
        if len(reference_importance) != len(client_importance[0]):
            reference_importance = np.zeros_like(client_importance[0])
            for imp in client_importance:
                reference_importance += imp
            reference_importance /= len(client_importance)
        
        # Compute NSDS scores
        nsds_scores = []
        for i, imp in enumerate(client_importance):
            nsds = compute_nsds(imp, reference_importance)
            nsds_scores.append(nsds)
            
            if is_byzantine[i]:
                all_nsds_byzantine.append(nsds)
            else:
                all_nsds_honest.append(nsds)
        
        # PoEx aggregation
        aggregated_update, accepted, rejected = poex_aggregate(client_updates, nsds_scores, CONFIG['threshold'])
        
        # Apply update to global model
        new_global_weights = {}
        for key in global_weights.keys():
            new_global_weights[key] = global_weights[key] + aggregated_update[key]
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
        
        # Compute detection metrics for this round
        tp = sum(1 for i in rejected if is_byzantine[i])
        fp = sum(1 for i in rejected if not is_byzantine[i])
        fn = sum(1 for i in accepted if is_byzantine[i])
        tn = sum(1 for i in accepted if not is_byzantine[i])
        
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  NSDS - Honest: {np.mean([nsds_scores[i] for i in range(len(nsds_scores)) if not is_byzantine[i]]):.4f}")
        print(f"  NSDS - Byzantine: {np.mean([nsds_scores[i] for i in range(len(nsds_scores)) if is_byzantine[i]]):.4f}")
        print(f"  TPR: {tpr:.3f}, FPR: {fpr:.3f}")
        print(f"  Accepted: {len(accepted)}, Rejected: {len(rejected)}")
        
        round_results.append({
            'round': round_idx + 1,
            'accuracy': accuracy,
            'tpr': tpr,
            'fpr': fpr,
            'accepted': len(accepted),
            'rejected': len(rejected),
        })
        
        # Update reference importance (moving average)
        if len(accepted) > 0:
            accepted_importance = np.mean([client_importance[i] for i in accepted], axis=0)
            reference_importance = 0.7 * reference_importance + 0.3 * accepted_importance
    
    return {
        'seed': seed,
        'final_accuracy': round_results[-1]['accuracy'],
        'round_results': round_results,
        'nsds_honest': all_nsds_honest,
        'nsds_byzantine': all_nsds_byzantine,
    }


def run_threshold_analysis(seed=42):
    """Analyze TPR/FPR across different thresholds"""
    print(f"\n{'='*60}")
    print(f"Running Threshold Analysis with seed={seed}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    trainset, testset = load_cifar10()
    testloader = DataLoader(testset, batch_size=100, shuffle=False)
    client_indices = dirichlet_partition(trainset, CONFIG['n_clients'], alpha=0.5, seed=seed)
    
    n_byzantine = int(CONFIG['n_clients'] * CONFIG['byzantine_fraction'])
    byzantine_clients = set(np.random.choice(CONFIG['n_clients'], n_byzantine, replace=False))
    
    # Initialize and train for a few rounds to get meaningful NSDS values
    global_model = CIFAR10CNN().to(DEVICE)
    global_weights = global_model.get_weights()
    
    # Collect NSDS from multiple rounds
    all_nsds_honest = []
    all_nsds_byzantine = []
    
    for round_idx in range(5):  # Quick analysis
        client_updates = []
        client_importance = []
        is_byzantine = []
        reference_importance = None
        
        for client_id in range(CONFIG['n_clients']):
            client_model = CIFAR10CNN().to(DEVICE)
            client_model.set_weights(global_weights)
            old_weights = client_model.get_weights()
            
            client_data = Subset(trainset, client_indices[client_id].tolist())
            if len(client_data) < 10:
                continue
            client_loader = DataLoader(client_data, batch_size=CONFIG['batch_size'], shuffle=True)
            
            optimizer = optim.SGD(client_model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            
            client_model.train()
            for epoch in range(CONFIG['local_epochs']):
                for data, target in client_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = client_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            new_weights = client_model.get_weights()
            update = {key: new_weights[key] - old_weights[key] for key in new_weights.keys()}
            
            if client_id in byzantine_clients:
                attack_type = np.random.choice(['sign_flip', 'noise', 'scaling'])
                if attack_type == 'sign_flip':
                    update = apply_sign_flip_attack(update, scale=1.5)
                elif attack_type == 'noise':
                    update = apply_noise_attack(update, noise_scale=0.5)
                else:
                    update = apply_scaling_attack(update, scale=5.0)
            
            importance = compute_weight_difference_importance(old_weights, 
                                                              {k: old_weights[k] + update[k] for k in update.keys()})
            
            client_updates.append(update)
            client_importance.append(importance)
            is_byzantine.append(client_id in byzantine_clients)
        
        if len(client_importance) == 0:
            continue
            
        reference_importance = np.mean(client_importance, axis=0)
        
        for i, imp in enumerate(client_importance):
            nsds = compute_nsds(imp, reference_importance)
            if is_byzantine[i]:
                all_nsds_byzantine.append(nsds)
            else:
                all_nsds_honest.append(nsds)
        
        # Update global model
        if len(client_updates) > 0:
            aggregated = fedavg_aggregate(client_updates)
            for key in global_weights.keys():
                global_weights[key] = global_weights[key] + aggregated[key]
            global_model.set_weights(global_weights)
    
    # Analyze thresholds
    threshold_results = []
    for tau in CONFIG['thresholds']:
        tp = sum(1 for nsds in all_nsds_byzantine if nsds >= tau)
        fn = sum(1 for nsds in all_nsds_byzantine if nsds < tau)
        fp = sum(1 for nsds in all_nsds_honest if nsds >= tau)
        tn = sum(1 for nsds in all_nsds_honest if nsds < tau)
        
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * tpr / max(precision + tpr, 1e-10)
        
        threshold_results.append({
            'threshold': tau,
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,
            'f1': f1,
        })
        
    return {
        'threshold_results': threshold_results,
        'nsds_honest': all_nsds_honest,
        'nsds_byzantine': all_nsds_byzantine,
    }


def create_visualizations(all_results, threshold_analysis):
    """Create visualization figures"""
    print("\nCreating visualizations...")
    
    # Aggregate NSDS values
    all_honest = []
    all_byzantine = []
    for result in all_results:
        all_honest.extend(result['nsds_honest'])
        all_byzantine.extend(result['nsds_byzantine'])
    
    # Figure 1: NSDS Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    bins = np.linspace(0, 1, 30)
    ax1.hist(all_honest, bins=bins, alpha=0.7, label=f'Honest (n={len(all_honest)})', color='green', density=True)
    ax1.hist(all_byzantine, bins=bins, alpha=0.7, label=f'Byzantine (n={len(all_byzantine)})', color='red', density=True)
    ax1.axvline(x=CONFIG['threshold'], color='black', linestyle='--', label=f'τ={CONFIG["threshold"]}')
    ax1.set_xlabel('NSDS Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('NSDS Distribution: CIFAR-10 CNN', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Boxplot
    ax2 = axes[1]
    data_to_plot = [all_honest, all_byzantine]
    bp = ax2.boxplot(data_to_plot, labels=['Honest', 'Byzantine'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.axhline(y=CONFIG['threshold'], color='black', linestyle='--', label=f'τ={CONFIG["threshold"]}')
    ax2.set_ylabel('NSDS Score', fontsize=12)
    ax2.set_title('NSDS Boxplot: CIFAR-10 CNN', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cifar10_nsds_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: TPR/FPR Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    
    thresholds = [r['threshold'] for r in threshold_analysis['threshold_results']]
    tprs = [r['tpr'] for r in threshold_analysis['threshold_results']]
    fprs = [r['fpr'] for r in threshold_analysis['threshold_results']]
    
    ax.plot(thresholds, tprs, 'b-o', label='TPR (True Positive Rate)', linewidth=2, markersize=8)
    ax.plot(thresholds, fprs, 'r-s', label='FPR (False Positive Rate)', linewidth=2, markersize=8)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default τ=0.5')
    ax.set_xlabel('Threshold (τ)', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('TPR/FPR vs Threshold: CIFAR-10 CNN', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cifar10_tpr_fpr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Accuracy over rounds
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for result in all_results:
        rounds = [r['round'] for r in result['round_results']]
        accs = [r['accuracy'] for r in result['round_results']]
        ax.plot(rounds, accs, 'o-', alpha=0.7, label=f'Seed {result["seed"]}')
    
    ax.set_xlabel('FL Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('PoEx Accuracy on CIFAR-10 CNN (30% Byzantine)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cifar10_accuracy_rounds.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved to {FIGURES_DIR}")


def main():
    """Main experiment runner"""
    print("="*70)
    print("CIFAR-10 Full CNN Experiment for PoEx Validation")
    print("="*70)
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    start_time = time.time()
    
    # Run experiments with multiple seeds
    all_results = []
    for seed in range(42, 42 + CONFIG['n_seeds']):
        result = run_experiment(seed=seed)
        all_results.append(result)
    
    # Run threshold analysis
    threshold_analysis = run_threshold_analysis(seed=42)
    
    # Aggregate statistics
    all_honest = []
    all_byzantine = []
    final_accuracies = []
    
    for result in all_results:
        all_honest.extend(result['nsds_honest'])
        all_byzantine.extend(result['nsds_byzantine'])
        final_accuracies.append(result['final_accuracy'])
    
    # Statistical analysis
    print("\n" + "="*70)
    print("CIFAR-10 CNN EXPERIMENT RESULTS")
    print("="*70)
    
    print(f"\nNSDS Distribution:")
    print(f"  Honest clients:    mean={np.mean(all_honest):.4f} ± std={np.std(all_honest):.4f}")
    print(f"  Byzantine clients: mean={np.mean(all_byzantine):.4f} ± std={np.std(all_byzantine):.4f}")
    
    # Mann-Whitney U test
    if len(all_honest) > 0 and len(all_byzantine) > 0:
        stat, pvalue = mannwhitneyu(all_honest, all_byzantine, alternative='two-sided')
        print(f"  Mann-Whitney U test: p-value={pvalue:.6f}")
        if pvalue < 0.05:
            print(f"  *** SIGNIFICANT DIFFERENCE (p < 0.05) ***")
    
    print(f"\nFinal Accuracy: {np.mean(final_accuracies):.2f}% ± {np.std(final_accuracies):.2f}%")
    
    print(f"\nThreshold Analysis (TPR/FPR):")
    for r in threshold_analysis['threshold_results']:
        print(f"  τ={r['threshold']:.1f}: TPR={r['tpr']:.3f}, FPR={r['fpr']:.3f}, F1={r['f1']:.3f}")
    
    # Create visualizations
    create_visualizations(all_results, threshold_analysis)
    
    # Save results to CSV
    results_df = pd.DataFrame(threshold_analysis['threshold_results'])
    results_df.to_csv(os.path.join(RESULTS_DIR, 'cifar10_threshold_analysis.csv'), index=False)
    
    # Save NSDS statistics
    nsds_stats = {
        'honest_mean': np.mean(all_honest),
        'honest_std': np.std(all_honest),
        'byzantine_mean': np.mean(all_byzantine),
        'byzantine_std': np.std(all_byzantine),
        'pvalue': pvalue if len(all_honest) > 0 and len(all_byzantine) > 0 else None,
        'final_accuracy_mean': np.mean(final_accuracies),
        'final_accuracy_std': np.std(final_accuracies),
    }
    
    with open(os.path.join(RESULTS_DIR, 'cifar10_nsds_stats.json'), 'w') as f:
        import json
        json.dump(nsds_stats, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.2f} minutes")
    print(f"Results saved to: {RESULTS_DIR}")
    
    return all_results, threshold_analysis


if __name__ == "__main__":
    all_results, threshold_analysis = main()
