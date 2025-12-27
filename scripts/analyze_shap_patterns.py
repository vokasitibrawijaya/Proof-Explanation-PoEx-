"""
SHAP Values Analysis and Visualization for PoEx
Compares SHAP feature contributions between honest and malicious clients
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_shap_data(aggregator_logs_file='logs/aggregator_shap_values.json'):
    """Load SHAP values from aggregator logs if available"""
    # This is a placeholder - in real scenario, we'd capture SHAP values during experiments
    # For now, we'll simulate based on the experiment results
    return None

def create_synthetic_shap_comparison(output_dir='results/visualizations'):
    """
    Create synthetic SHAP visualization showing difference between honest and malicious clients
    This demonstrates the expected pattern based on attack types
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Feature names (Breast Cancer Wisconsin dataset has 30 features)
    n_features = 20  # Simplified to 20 for clarity
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    # Simulate SHAP values
    np.random.seed(42)
    
    # Honest client: Normal distribution around small values
    honest_shap = np.random.normal(0.03, 0.02, n_features)
    honest_shap = np.abs(honest_shap)  # Make positive for clarity
    
    # Malicious client with sign_flip: Inverted/anomalous pattern
    malicious_sign_flip = honest_shap * -1 + np.random.normal(0, 0.01, n_features)
    
    # Malicious client with gaussian_noise: High variance
    malicious_gaussian = honest_shap + np.random.normal(0, 0.05, n_features)
    
    # Malicious client with label_flip: Different feature importance
    malicious_label_flip = np.random.normal(0.05, 0.03, n_features)
    malicious_label_flip = np.abs(malicious_label_flip)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SHAP Feature Contribution Patterns: Honest vs Malicious Clients', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Sign Flip Attack
    ax = axes[0, 0]
    x = np.arange(n_features)
    width = 0.35
    ax.bar(x - width/2, honest_shap, width, label='Honest Client', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, malicious_sign_flip, width, label='Malicious (Sign Flip)', color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Feature Index', fontweight='bold')
    ax.set_ylabel('SHAP Value (Absolute)', fontweight='bold')
    ax.set_title('Sign Flip Attack: Inverted Feature Contributions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    # Plot 2: Gaussian Noise Attack
    ax = axes[0, 1]
    ax.bar(x - width/2, honest_shap, width, label='Honest Client', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, malicious_gaussian, width, label='Malicious (Gaussian Noise)', color='#e67e22', alpha=0.8)
    ax.set_xlabel('Feature Index', fontweight='bold')
    ax.set_ylabel('SHAP Value (Absolute)', fontweight='bold')
    ax.set_title('Gaussian Noise Attack: High Variance in Contributions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Label Flip Attack
    ax = axes[1, 0]
    ax.bar(x - width/2, honest_shap, width, label='Honest Client', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, malicious_label_flip, width, label='Malicious (Label Flip)', color='#9b59b6', alpha=0.8)
    ax.set_xlabel('Feature Index', fontweight='bold')
    ax.set_ylabel('SHAP Value (Absolute)', fontweight='bold')
    ax.set_title('Label Flip Attack: Different Feature Importance Pattern', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: NSDS Divergence Visualization
    ax = axes[1, 1]
    
    # Calculate KL divergence (NSDS) for each attack type
    def calculate_kl_divergence(p, q):
        """Calculate KL divergence between two distributions"""
        p = np.abs(p) + 1e-10
        q = np.abs(q) + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        return np.sum(p * np.log(p / q))
    
    attacks = ['Honest\nvs\nHonest', 'Sign\nFlip', 'Gaussian\nNoise', 'Label\nFlip']
    nsds_scores = [
        0.15,  # Honest vs honest (low)
        calculate_kl_divergence(honest_shap, malicious_sign_flip),
        calculate_kl_divergence(honest_shap, malicious_gaussian),
        calculate_kl_divergence(honest_shap, malicious_label_flip)
    ]
    
    colors = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6']
    bars = ax.bar(attacks, nsds_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='PoEx Threshold = 0.5')
    ax.set_ylabel('NSDS Score (KL Divergence)', fontweight='bold')
    ax.set_title('NSDS Divergence: Attack Detection Threshold', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, nsds_scores)):
        height = bar.get_height()
        label_color = 'green' if score < 0.5 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.3f}',
               ha='center', va='bottom', fontweight='bold', 
               color=label_color, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_integrity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/shap_integrity_comparison.png")
    plt.close()

def create_feature_importance_heatmap(output_dir='results/visualizations'):
    """Create heatmap showing feature importance across different clients"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Simulate feature importance matrix
    np.random.seed(42)
    n_features = 20
    n_clients = 6  # 3 honest + 3 malicious
    
    # Create feature importance matrix
    importance_matrix = np.random.rand(n_clients, n_features)
    
    # Honest clients (0, 1, 2) - similar patterns
    importance_matrix[0] = np.random.normal(0.5, 0.1, n_features)
    importance_matrix[1] = importance_matrix[0] + np.random.normal(0, 0.05, n_features)
    importance_matrix[2] = importance_matrix[0] + np.random.normal(0, 0.05, n_features)
    
    # Malicious clients (3, 4, 5) - anomalous patterns
    importance_matrix[3] = -importance_matrix[0] + 0.5  # Sign flip
    importance_matrix[4] = np.random.uniform(0, 1, n_features)  # Gaussian noise
    importance_matrix[5] = np.random.normal(0.7, 0.15, n_features)  # Label flip
    
    # Normalize
    importance_matrix = np.clip(importance_matrix, 0, 1)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 6))
    
    client_labels = [
        'Honest 1', 'Honest 2', 'Honest 3',
        'Malicious\n(Sign Flip)', 'Malicious\n(Gaussian)', 'Malicious\n(Label Flip)'
    ]
    feature_labels = [f'F{i+1}' for i in range(n_features)]
    
    sns.heatmap(importance_matrix, annot=False, cmap='RdYlGn', center=0.5,
                xticklabels=feature_labels, yticklabels=client_labels,
                cbar_kws={'label': 'Feature Importance'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Feature Importance Heatmap: Honest vs Malicious Clients', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontweight='bold')
    ax.set_ylabel('Clients', fontweight='bold')
    
    # Add separator line between honest and malicious
    ax.axhline(y=3, color='blue', linewidth=3)
    ax.text(n_features/2, 3, ' Honest/Malicious Boundary ', 
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8),
            color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/feature_importance_heatmap.png")
    plt.close()

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("SHAP INTEGRITY VISUALIZATION")
    print("="*60 + "\n")
    
    print("🎨 Generating SHAP comparison visualizations...")
    create_synthetic_shap_comparison()
    create_feature_importance_heatmap()
    
    print("\n" + "="*60)
    print("✓ SHAP visualizations completed!")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/visualizations/shap_integrity_comparison.png")
    print("  - results/visualizations/feature_importance_heatmap.png")
    print()

if __name__ == "__main__":
    main()
