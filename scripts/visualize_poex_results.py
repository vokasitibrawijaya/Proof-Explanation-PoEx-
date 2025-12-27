#!/usr/bin/env python3
"""
Visualisasi Hasil Eksperimen PoEx
Membuat grafik perbandingan Baseline vs Proposed untuk semua metrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_results(csv_path):
    """Load experiment results from CSV"""
    df = pd.read_csv(csv_path)
    return df

def plot_accuracy_comparison(df, output_dir):
    """Plot accuracy comparison between baseline and proposed"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    attacks = [
        ('none', 'No Attack'),
        ('label_flip', 'Label Flipping'),
        ('gaussian_noise', 'Gaussian Noise')
    ]
    
    for idx, (attack_type, attack_label) in enumerate(attacks):
        ax = axes[idx]
        
        # Filter data
        baseline = df[(df['poex_enabled'] == 0) & (df['attack_type'] == attack_type)]
        proposed = df[(df['poex_enabled'] == 1) & (df['attack_type'] == attack_type)]
        
        if not baseline.empty:
            ax.plot(baseline['round'], baseline['global_accuracy'], 
                   'o-', label='Baseline (PoEx OFF)', color='red', linewidth=2, markersize=8)
        if not proposed.empty:
            ax.plot(proposed['round'], proposed['global_accuracy'], 
                   's-', label='Proposed (PoEx ON)', color='green', linewidth=2, markersize=8)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Global Accuracy', fontsize=12)
        ax.set_title(f'{attack_label}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'accuracy_comparison.png'}")
    plt.close()

def plot_precision_recall_f1(df, output_dir):
    """Plot precision, recall, F1 for all scenarios"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    attacks = [
        ('none', 'No Attack'),
        ('label_flip', 'Label Flipping'),
        ('gaussian_noise', 'Gaussian Noise')
    ]
    
    metrics = ['global_precision', 'global_recall', 'global_f1']
    metric_labels = ['Precision', 'Recall', 'F1-Score']
    
    for col_idx, (attack_type, attack_label) in enumerate(attacks):
        # Baseline
        ax_baseline = axes[0, col_idx]
        baseline = df[(df['poex_enabled'] == 0) & (df['attack_type'] == attack_type)]
        
        if not baseline.empty:
            for metric, label in zip(metrics, metric_labels):
                ax_baseline.plot(baseline['round'], baseline[metric], 
                               'o-', label=label, linewidth=2, markersize=6)
        
        ax_baseline.set_title(f'Baseline - {attack_label}', fontsize=12, fontweight='bold')
        ax_baseline.set_xlabel('Round')
        ax_baseline.set_ylabel('Score')
        ax_baseline.legend()
        ax_baseline.grid(True, alpha=0.3)
        ax_baseline.set_ylim([0, 1.05])
        
        # Proposed
        ax_proposed = axes[1, col_idx]
        proposed = df[(df['poex_enabled'] == 1) & (df['attack_type'] == attack_type)]
        
        if not proposed.empty:
            for metric, label in zip(metrics, metric_labels):
                ax_proposed.plot(proposed['round'], proposed[metric], 
                               's-', label=label, linewidth=2, markersize=6)
        
        ax_proposed.set_title(f'Proposed - {attack_label}', fontsize=12, fontweight='bold')
        ax_proposed.set_xlabel('Round')
        ax_proposed.set_ylabel('Score')
        ax_proposed.legend()
        ax_proposed.grid(True, alpha=0.3)
        ax_proposed.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_f1.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'precision_recall_f1.png'}")
    plt.close()

def plot_security_metrics(df, output_dir):
    """Plot accepted vs rejected updates"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    attacks = [
        ('none', 'No Attack'),
        ('label_flip', 'Label Flipping'),
        ('gaussian_noise', 'Gaussian Noise')
    ]
    
    for idx, (attack_type, attack_label) in enumerate(attacks):
        ax = axes[idx]
        
        # Only show proposed (PoEx enabled)
        proposed = df[(df['poex_enabled'] == 1) & (df['attack_type'] == attack_type)]
        
        if not proposed.empty:
            x = np.arange(len(proposed))
            width = 0.35
            
            ax.bar(x - width/2, proposed['accepted_updates'], width, 
                  label='Accepted', color='green', alpha=0.7)
            ax.bar(x + width/2, proposed['rejected_updates'], width, 
                  label='Rejected', color='red', alpha=0.7)
            
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'PoEx Decision - {attack_label}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(proposed['round'])
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'security_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'security_metrics.png'}")
    plt.close()

def plot_latency(df, output_dir):
    """Plot PoEx validation latency"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    attacks = [
        ('none', 'No Attack'),
        ('label_flip', 'Label Flipping'),
        ('gaussian_noise', 'Gaussian Noise')
    ]
    
    colors = ['blue', 'orange', 'purple']
    
    for (attack_type, attack_label), color in zip(attacks, colors):
        proposed = df[(df['poex_enabled'] == 1) & (df['attack_type'] == attack_type)]
        
        if not proposed.empty:
            ax.plot(proposed['round'], proposed['avg_poex_latency_ms'], 
                   'o-', label=attack_label, color=color, linewidth=2, markersize=8)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('PoEx Latency (ms)', fontsize=12)
    ax.set_title('PoEx Validation Overhead', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'poex_latency.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'poex_latency.png'}")
    plt.close()

def generate_summary_table(df, output_dir):
    """Generate summary statistics table"""
    summary_rows = []
    
    for poex in [0, 1]:
        poex_label = "Proposed (PoEx ON)" if poex else "Baseline (PoEx OFF)"
        
        for attack in ['none', 'label_flip', 'gaussian_noise']:
            subset = df[(df['poex_enabled'] == poex) & (df['attack_type'] == attack)]
            
            if not subset.empty:
                # Get last round metrics
                last_round = subset[subset['round'] == subset['round'].max()].iloc[0]
                
                summary_rows.append({
                    'Method': poex_label,
                    'Attack': attack.replace('_', ' ').title(),
                    'Final Accuracy': f"{last_round['global_accuracy']:.4f}",
                    'Final Precision': f"{last_round['global_precision']:.4f}",
                    'Final Recall': f"{last_round['global_recall']:.4f}",
                    'Final F1': f"{last_round['global_f1']:.4f}",
                    'Avg NSDS': f"{subset['avg_nsds'].mean():.4f}",
                    'Accepted': int(subset['accepted_updates'].sum()),
                    'Rejected': int(subset['rejected_updates'].sum()),
                    'Avg Latency (ms)': f"{subset['avg_poex_latency_ms'].mean():.2f}" if poex else "N/A"
                })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"✓ Saved: {output_dir / 'summary_statistics.csv'}")
    
    # Print to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    return summary_df

def main():
    # Paths
    results_file = Path("results/poex_results.csv")
    output_dir = Path("results/visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("Run experiments first with: ./run_all_poex_experiments.ps1")
        return
    
    print("="*80)
    print("  PoEx Results Visualization")
    print("="*80)
    print(f"\nLoading results from: {results_file}")
    
    df = load_results(results_file)
    print(f"✓ Loaded {len(df)} rows")
    print(f"\nScenarios found:")
    for _, row in df.groupby(['poex_enabled', 'attack_type']).size().reset_index().iterrows():
        method = "Proposed" if row['poex_enabled'] else "Baseline"
        print(f"  - {method} + {row['attack_type']}")
    
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(df, output_dir)
    plot_precision_recall_f1(df, output_dir)
    plot_security_metrics(df, output_dir)
    plot_latency(df, output_dir)
    generate_summary_table(df, output_dir)
    
    print("\n" + "="*80)
    print("✓ All visualizations generated successfully!")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - accuracy_comparison.png")
    print("  - precision_recall_f1.png")
    print("  - security_metrics.png")
    print("  - poex_latency.png")
    print("  - summary_statistics.csv")

if __name__ == "__main__":
    main()
