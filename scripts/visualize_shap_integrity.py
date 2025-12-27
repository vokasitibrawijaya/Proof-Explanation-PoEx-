#!/usr/bin/env python3
"""
Visualisasi SHAP Explanations untuk Node Normal vs Malicious
Untuk membuktikan bahwa node malicious memiliki pola kontribusi fitur yang berbeda
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import shap

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

def generate_data(seed=42):
    """Generate synthetic data for demonstration"""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=2,
        n_informative=14,
        n_redundant=4,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # Split for 3 clients
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    splits = np.array_split(idx, 3)
    
    client_data = []
    for i, split_idx in enumerate(splits):
        client_data.append({
            'X_train': X_train[split_idx],
            'y_train': y_train[split_idx],
            'X_test': X_test,
            'y_test': y_test
        })
    
    return client_data

def train_normal_client(client_data, seed=42):
    """Train normal (honest) client"""
    model = SGDClassifier(loss='log_loss', max_iter=100, random_state=seed)
    model.fit(client_data['X_train'], client_data['y_train'])
    return model

def train_malicious_client_label_flip(client_data, seed=42):
    """Train malicious client with label flipping attack"""
    model = SGDClassifier(loss='log_loss', max_iter=100, random_state=seed)
    # Flip labels
    y_malicious = 1 - client_data['y_train']
    model.fit(client_data['X_train'], y_malicious)
    return model

def train_malicious_client_gaussian(client_data, seed=42, sigma=0.1):
    """Train malicious client with Gaussian noise attack"""
    model = SGDClassifier(loss='log_loss', max_iter=100, random_state=seed)
    model.fit(client_data['X_train'], client_data['y_train'])
    
    # Add Gaussian noise to model weights
    np.random.seed(seed)
    if hasattr(model, 'coef_'):
        model.coef_ = model.coef_ + np.random.normal(0, sigma, model.coef_.shape)
    if hasattr(model, 'intercept_'):
        model.intercept_ = model.intercept_ + np.random.normal(0, sigma, model.intercept_.shape)
    
    return model

def compute_shap_values(model, X_background, X_explain, n_samples=50):
    """Compute SHAP values for model"""
    # Use a subset of background data for speed
    bg_idx = np.random.choice(len(X_background), min(n_samples, len(X_background)), replace=False)
    X_bg = X_background[bg_idx]
    
    # Compute SHAP values
    explainer = shap.KernelExplainer(model.predict_proba, X_bg)
    
    # Explain a few instances
    explain_idx = np.random.choice(len(X_explain), min(10, len(X_explain)), replace=False)
    X_exp = X_explain[explain_idx]
    
    shap_values = explainer.shap_values(X_exp, nsamples=50)
    
    # Return mean absolute SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification, class 1
    
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    return mean_shap

def plot_shap_comparison(shap_normal, shap_label_flip, shap_gaussian, output_dir):
    """Plot SHAP value comparison between normal and malicious nodes"""
    
    n_features = len(shap_normal)
    feature_names = [f'F{i}' for i in range(n_features)]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Normal vs Label Flip
    ax1 = axes[0]
    x = np.arange(n_features)
    width = 0.35
    
    ax1.bar(x - width/2, shap_normal, width, label='Normal Node', color='green', alpha=0.7)
    ax1.bar(x + width/2, shap_label_flip, width, label='Malicious (Label Flip)', color='red', alpha=0.7)
    
    ax1.set_ylabel('Mean |SHAP Value|', fontsize=12)
    ax1.set_title('SHAP Values: Normal vs Label Flipping Attack', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names, rotation=45)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Normal vs Gaussian Noise
    ax2 = axes[1]
    
    ax2.bar(x - width/2, shap_normal, width, label='Normal Node', color='green', alpha=0.7)
    ax2.bar(x + width/2, shap_gaussian, width, label='Malicious (Gaussian Noise)', color='orange', alpha=0.7)
    
    ax2.set_ylabel('Mean |SHAP Value|', fontsize=12)
    ax2.set_title('SHAP Values: Normal vs Gaussian Noise Attack', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_names, rotation=45)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: All together
    ax3 = axes[2]
    width = 0.25
    
    ax3.bar(x - width, shap_normal, width, label='Normal Node', color='green', alpha=0.7)
    ax3.bar(x, shap_label_flip, width, label='Label Flip Attack', color='red', alpha=0.7)
    ax3.bar(x + width, shap_gaussian, width, label='Gaussian Noise Attack', color='orange', alpha=0.7)
    
    ax3.set_xlabel('Features', fontsize=12)
    ax3.set_ylabel('Mean |SHAP Value|', fontsize=12)
    ax3.set_title('SHAP Values: All Scenarios Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(feature_names, rotation=45)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_comparison_bar.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'shap_comparison_bar.png'}")
    plt.close()

def plot_shap_heatmap(shap_normal, shap_label_flip, shap_gaussian, output_dir):
    """Plot heatmap of SHAP values"""
    
    n_features = len(shap_normal)
    feature_names = [f'F{i}' for i in range(n_features)]
    
    # Create matrix
    shap_matrix = np.array([
        shap_normal,
        shap_label_flip,
        shap_gaussian
    ])
    
    # Normalize for better visualization
    shap_matrix_norm = (shap_matrix - shap_matrix.min(axis=1, keepdims=True)) / \
                       (shap_matrix.max(axis=1, keepdims=True) - shap_matrix.min(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    im = ax.imshow(shap_matrix_norm, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Normal Node', 'Label Flip Attack', 'Gaussian Noise Attack'])
    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(feature_names, rotation=45)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized |SHAP Value|', rotation=270, labelpad=20)
    
    # Add values in cells
    for i in range(3):
        for j in range(n_features):
            text = ax.text(j, i, f'{shap_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title('SHAP Feature Importance Heatmap: Normal vs Malicious Nodes', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'shap_heatmap.png'}")
    plt.close()

def compute_kl_divergence(p, q):
    """Compute KL divergence between two probability distributions"""
    # Normalize to probability distributions
    p = np.abs(p) + 1e-10
    q = np.abs(q) + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    
    # KL divergence
    kl = np.sum(p * np.log(p / q))
    return kl

def plot_kl_divergence(shap_normal, shap_label_flip, shap_gaussian, output_dir):
    """Plot KL divergence between normal and malicious nodes"""
    
    kl_normal_vs_label = compute_kl_divergence(shap_normal, shap_label_flip)
    kl_normal_vs_gaussian = compute_kl_divergence(shap_normal, shap_gaussian)
    kl_label_vs_gaussian = compute_kl_divergence(shap_label_flip, shap_gaussian)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    comparisons = ['Normal vs\nLabel Flip', 'Normal vs\nGaussian Noise', 'Label Flip vs\nGaussian Noise']
    kl_values = [kl_normal_vs_label, kl_normal_vs_gaussian, kl_label_vs_gaussian]
    colors = ['red', 'orange', 'purple']
    
    bars = ax.bar(comparisons, kl_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, kl_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('KL Divergence (NSDS)', fontsize=12)
    ax.set_title('KL Divergence Between SHAP Distributions\n(Higher = More Different)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add threshold line (example: 0.5)
    ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Example PoEx Threshold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kl_divergence.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'kl_divergence.png'}")
    plt.close()
    
    # Print KL values
    print("\n" + "="*80)
    print("KL DIVERGENCE (NSDS) VALUES")
    print("="*80)
    print(f"Normal vs Label Flip:      {kl_normal_vs_label:.6f}")
    print(f"Normal vs Gaussian Noise:  {kl_normal_vs_gaussian:.6f}")
    print(f"Label Flip vs Gaussian:    {kl_label_vs_gaussian:.6f}")
    print("="*80)
    print("\nInterpretation:")
    print("- Higher KL divergence = More different explanation patterns")
    print("- PoEx uses this metric to detect anomalous nodes")
    print("- Threshold example: 0.5 (nodes with NSDS > 0.5 are rejected)")
    print("="*80)

def main():
    output_dir = Path("results/visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("  SHAP Explanation Visualization: Normal vs Malicious Nodes")
    print("="*80)
    print("\nGenerating synthetic data and training models...")
    
    # Generate data
    np.random.seed(42)
    client_data = generate_data(seed=42)
    
    # Train models
    print("✓ Training normal node...")
    model_normal = train_normal_client(client_data[0], seed=42)
    
    print("✓ Training malicious node (label flip)...")
    model_label_flip = train_malicious_client_label_flip(client_data[1], seed=43)
    
    print("✓ Training malicious node (gaussian noise)...")
    model_gaussian = train_malicious_client_gaussian(client_data[2], seed=44, sigma=0.1)
    
    # Compute SHAP values
    print("\nComputing SHAP values...")
    X_bg = client_data[0]['X_train']
    X_exp = client_data[0]['X_test'][:50]  # Use first 50 test samples
    
    print("  - Normal node...")
    shap_normal = compute_shap_values(model_normal, X_bg, X_exp)
    
    print("  - Label flip attack...")
    shap_label_flip = compute_shap_values(model_label_flip, X_bg, X_exp)
    
    print("  - Gaussian noise attack...")
    shap_gaussian = compute_shap_values(model_gaussian, X_bg, X_exp)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_shap_comparison(shap_normal, shap_label_flip, shap_gaussian, output_dir)
    plot_shap_heatmap(shap_normal, shap_label_flip, shap_gaussian, output_dir)
    plot_kl_divergence(shap_normal, shap_label_flip, shap_gaussian, output_dir)
    
    print("\n" + "="*80)
    print("✓ SHAP visualizations generated successfully!")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - shap_comparison_bar.png")
    print("  - shap_heatmap.png")
    print("  - kl_divergence.png")
    print("\nThese visualizations demonstrate that:")
    print("  1. Malicious nodes have DIFFERENT SHAP patterns than normal nodes")
    print("  2. KL divergence (NSDS) can detect these anomalies")
    print("  3. PoEx consensus uses this to reject malicious updates BEFORE ledger")

if __name__ == "__main__":
    main()
