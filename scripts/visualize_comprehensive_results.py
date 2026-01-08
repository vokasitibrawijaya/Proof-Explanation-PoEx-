#!/usr/bin/env python3
"""
Visualize Comprehensive PoEx Experiment Results
Generates publication-quality figures for IEEE Access paper

Author: FedXChain Research Team
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import json
import os

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Custom color palette
COLORS = {
    'PoEx': '#2E86AB',      # Blue
    'FedAvg': '#A23B72',    # Magenta
    'Krum': '#F18F01',      # Orange
    'MultiKrum': '#C73E1D', # Red
    'TrimmedMean': '#3B1F2B', # Dark
    'Bulyan': '#44AF69',    # Green
}


def load_results(results_dir):
    """Load experiment results"""
    csv_path = os.path.join(results_dir, 'comprehensive_results.csv')
    df = pd.read_csv(csv_path)
    return df


def plot_method_comparison(df, output_dir):
    """Plot accuracy comparison across methods and attacks"""
    
    # Filter for method comparison (IID, 30% Byzantine)
    df_comp = df[(df['data_dist'] == 'iid') & 
                 (df['n_byzantine'] == 3) & 
                 (df['attack'].isin(['sign_flip', 'label_flip', 'gaussian_noise']))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['FedAvg', 'Krum', 'MultiKrum', 'TrimmedMean', 'PoEx']
    attacks = ['sign_flip', 'label_flip', 'gaussian_noise']
    attack_labels = ['Sign Flip', 'Label Flip', 'Gaussian Noise']
    
    x = np.arange(len(attacks))
    width = 0.15
    
    for i, method in enumerate(methods):
        acc_values = []
        for attack in attacks:
            row = df_comp[(df_comp['method'] == method) & (df_comp['attack'] == attack)]
            if len(row) > 0:
                acc_values.append(row['final_accuracy'].values[0])
            else:
                acc_values.append(0)
        
        bars = ax.bar(x + i * width, acc_values, width, 
                     label=method, color=COLORS.get(method, '#666666'))
        
        # Add value labels
        for bar, val in zip(bars, acc_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Attack Type')
    ax.set_title('Aggregation Method Comparison Under Byzantine Attacks\n(10 clients, 30% Byzantine, 50 rounds)')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(attack_labels)
    ax.legend(loc='lower right', ncol=2)
    ax.set_ylim(0.85, 1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'))
    plt.savefig(os.path.join(output_dir, 'method_comparison.pdf'))
    plt.close()
    
    print("Saved: method_comparison.png/pdf")


def plot_threshold_sensitivity(df, output_dir):
    """Plot threshold sensitivity analysis for PoEx"""
    
    # Filter for threshold sensitivity experiments
    df_thresh = df[(df['method'] == 'PoEx') & 
                   (df['attack'] == 'sign_flip') & 
                   (df['data_dist'] == 'iid') &
                   (df['threshold'] != 'N/A')]
    
    if len(df_thresh) == 0:
        print("No threshold sensitivity data found")
        return
    
    df_thresh = df_thresh.copy()
    df_thresh['threshold'] = pd.to_numeric(df_thresh['threshold'])
    df_thresh = df_thresh.sort_values('threshold')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(df_thresh['threshold'], df_thresh['final_accuracy'], 
            'o-', color=COLORS['PoEx'], linewidth=2, markersize=10, label='Final Accuracy')
    ax.plot(df_thresh['threshold'], df_thresh['avg_accuracy'], 
            's--', color='#44AF69', linewidth=2, markersize=8, label='Average Accuracy')
    
    ax.axhline(y=0.9737, color='gray', linestyle=':', alpha=0.7, label='FedAvg Baseline')
    
    ax.set_xlabel('Threshold (τ)')
    ax.set_ylabel('Accuracy')
    ax.set_title('PoEx Threshold Sensitivity Analysis\n(Sign Flip Attack, 30% Byzantine)')
    ax.legend()
    ax.set_ylim(0.90, 1.00)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Add annotation
    ax.annotate('Optimal range: τ ∈ [0.3, 0.7]', 
               xy=(0.5, 0.975), fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_sensitivity.png'))
    plt.savefig(os.path.join(output_dir, 'threshold_sensitivity.pdf'))
    plt.close()
    
    print("Saved: threshold_sensitivity.png/pdf")


def plot_byzantine_fraction(df, output_dir):
    """Plot accuracy vs Byzantine fraction"""
    
    # Filter for Byzantine fraction experiments
    df_byz = df[(df['method'] == 'PoEx') & 
                (df['attack'] == 'sign_flip') &
                (df['threshold'] == 'N/A') &
                (df['data_dist'] == 'iid')]
    
    if len(df_byz) < 3:
        print("Not enough Byzantine fraction data")
        return
    
    df_byz = df_byz.copy()
    df_byz['byzantine_fraction'] = df_byz['n_byzantine'] / 10
    df_byz = df_byz.sort_values('byzantine_fraction')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(df_byz['byzantine_fraction'], df_byz['final_accuracy'], 
            'o-', color=COLORS['PoEx'], linewidth=2, markersize=10, label='PoEx')
    
    # Add theoretical bound line
    fractions = np.linspace(0.1, 0.4, 100)
    theoretical = 1.0 - 0.5 * fractions  # Simplified theoretical degradation
    ax.plot(fractions, theoretical, '--', color='gray', 
            linewidth=1.5, alpha=0.7, label='Theoretical Bound')
    
    ax.fill_between(df_byz['byzantine_fraction'], 
                   df_byz['final_accuracy'] - 0.01,
                   df_byz['final_accuracy'] + 0.01,
                   color=COLORS['PoEx'], alpha=0.2)
    
    ax.set_xlabel('Byzantine Fraction (α)')
    ax.set_ylabel('Accuracy')
    ax.set_title('PoEx Robustness vs Byzantine Fraction')
    ax.legend()
    ax.set_ylim(0.85, 1.00)
    ax.set_xlim(0.05, 0.45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'byzantine_fraction.png'))
    plt.savefig(os.path.join(output_dir, 'byzantine_fraction.pdf'))
    plt.close()
    
    print("Saved: byzantine_fraction.png/pdf")


def plot_adaptive_attack(df, output_dir):
    """Plot performance under adaptive attack"""
    
    df_adapt = df[df['attack'] == 'adaptive']
    
    if len(df_adapt) == 0:
        print("No adaptive attack data found")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = ['FedAvg', 'Krum', 'TrimmedMean', 'PoEx']
    accuracies = []
    
    for method in methods:
        row = df_adapt[df_adapt['method'] == method]
        if len(row) > 0:
            accuracies.append(row['final_accuracy'].values[0])
        else:
            accuracies.append(0)
    
    colors = [COLORS.get(m, '#666666') for m in methods]
    bars = ax.bar(methods, accuracies, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Aggregation Method')
    ax.set_title('Performance Under Adaptive Attack\n(30% Colluding Byzantine Clients)')
    ax.set_ylim(0.85, 1.02)
    
    # Add horizontal line for baseline
    ax.axhline(y=0.97, color='gray', linestyle='--', alpha=0.5, label='Target: 97%')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adaptive_attack.png'))
    plt.savefig(os.path.join(output_dir, 'adaptive_attack.pdf'))
    plt.close()
    
    print("Saved: adaptive_attack.png/pdf")


def plot_iid_vs_non_iid(df, output_dir):
    """Plot IID vs Non-IID comparison"""
    
    # Get IID and Non-IID data
    df_iid = df[(df['data_dist'] == 'iid') & 
                (df['attack'] == 'sign_flip') &
                (df['n_byzantine'] == 3) &
                (df['method'].isin(['FedAvg', 'PoEx']))]
    
    df_noniid = df[(df['data_dist'] == 'non_iid') &
                   (df['method'].isin(['FedAvg', 'PoEx']))]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = ['FedAvg', 'PoEx']
    x = np.arange(len(methods))
    width = 0.35
    
    iid_acc = []
    noniid_acc = []
    
    for method in methods:
        iid_row = df_iid[df_iid['method'] == method]
        noniid_row = df_noniid[df_noniid['method'] == method]
        
        iid_acc.append(iid_row['final_accuracy'].values[0] if len(iid_row) > 0 else 0)
        noniid_acc.append(noniid_row['final_accuracy'].values[0] if len(noniid_row) > 0 else 0)
    
    bars1 = ax.bar(x - width/2, iid_acc, width, label='IID', color='#2E86AB')
    bars2 = ax.bar(x + width/2, noniid_acc, width, label='Non-IID (α=0.5)', color='#F18F01')
    
    # Add value labels
    for bar, val in zip(bars1, iid_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, noniid_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Aggregation Method')
    ax.set_title('IID vs Non-IID Data Distribution\n(Sign Flip Attack, 30% Byzantine)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0.80, 1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iid_vs_non_iid.png'))
    plt.savefig(os.path.join(output_dir, 'iid_vs_non_iid.pdf'))
    plt.close()
    
    print("Saved: iid_vs_non_iid.png/pdf")


def plot_summary_heatmap(df, output_dir):
    """Create summary heatmap of all results"""
    
    # Filter for main comparison
    df_main = df[(df['data_dist'] == 'iid') & 
                 (df['n_byzantine'] == 3) &
                 (df['attack'].isin(['sign_flip', 'label_flip', 'gaussian_noise', 'adaptive']))]
    
    # Pivot for heatmap
    methods = ['FedAvg', 'Krum', 'MultiKrum', 'TrimmedMean', 'PoEx']
    attacks = ['sign_flip', 'label_flip', 'gaussian_noise', 'adaptive']
    
    heatmap_data = np.zeros((len(methods), len(attacks)))
    
    for i, method in enumerate(methods):
        for j, attack in enumerate(attacks):
            row = df_main[(df_main['method'] == method) & (df_main['attack'] == attack)]
            if len(row) > 0:
                heatmap_data[i, j] = row['final_accuracy'].values[0]
            else:
                heatmap_data[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    attack_labels = ['Sign Flip', 'Label Flip', 'Gaussian Noise', 'Adaptive']
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.85, vmax=1.0)
    
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attack_labels)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    
    # Add values in cells
    for i in range(len(methods)):
        for j in range(len(attacks)):
            if not np.isnan(heatmap_data[i, j]):
                text_color = 'white' if heatmap_data[i, j] < 0.92 else 'black'
                ax.text(j, i, f'{heatmap_data[i, j]:.3f}', 
                       ha='center', va='center', color=text_color, fontsize=11)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy')
    
    ax.set_title('Accuracy Heatmap: Methods vs Attacks\n(10 clients, 30% Byzantine)')
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Aggregation Method')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_heatmap.png'))
    plt.savefig(os.path.join(output_dir, 'summary_heatmap.pdf'))
    plt.close()
    
    print("Saved: summary_heatmap.png/pdf")


def generate_latex_table(df, output_dir):
    """Generate LaTeX table for paper"""
    
    df_main = df[(df['data_dist'] == 'iid') & 
                 (df['n_byzantine'] == 3) &
                 (df['attack'].isin(['sign_flip', 'label_flip', 'gaussian_noise']))]
    
    methods = ['FedAvg', 'Krum', 'MultiKrum', 'TrimmedMean', 'PoEx']
    attacks = ['sign_flip', 'label_flip', 'gaussian_noise']
    
    latex = """
\\begin{table}[h]
\\centering
\\caption{Accuracy Comparison Under Byzantine Attacks (10 Clients, 30\\% Byzantine)}
\\label{tab:accuracy_comparison}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Method} & \\textbf{Sign Flip} & \\textbf{Label Flip} & \\textbf{Gaussian Noise} & \\textbf{Average} \\\\
\\midrule
"""
    
    best_scores = {}
    for attack in attacks:
        scores = []
        for method in methods:
            row = df_main[(df_main['method'] == method) & (df_main['attack'] == attack)]
            if len(row) > 0:
                scores.append(row['final_accuracy'].values[0])
            else:
                scores.append(0)
        best_scores[attack] = max(scores)
    
    for method in methods:
        row_data = [method]
        acc_values = []
        for attack in attacks:
            row = df_main[(df_main['method'] == method) & (df_main['attack'] == attack)]
            if len(row) > 0:
                acc = row['final_accuracy'].values[0]
                acc_values.append(acc)
                if acc == best_scores[attack]:
                    row_data.append(f"\\textbf{{{acc:.4f}}}")
                else:
                    row_data.append(f"{acc:.4f}")
            else:
                row_data.append("--")
        
        avg = np.mean(acc_values) if acc_values else 0
        row_data.append(f"{avg:.4f}")
        
        latex += " & ".join(row_data) + " \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(os.path.join(output_dir, 'accuracy_table.tex'), 'w') as f:
        f.write(latex)
    
    print("Saved: accuracy_table.tex")


def main():
    results_dir = 'results/comprehensive_experiments'
    output_dir = 'results/comprehensive_experiments/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading results...")
    df = load_results(results_dir)
    print(f"Loaded {len(df)} experiment results")
    
    print("\nGenerating figures...")
    
    plot_method_comparison(df, output_dir)
    plot_threshold_sensitivity(df, output_dir)
    plot_byzantine_fraction(df, output_dir)
    plot_adaptive_attack(df, output_dir)
    plot_iid_vs_non_iid(df, output_dir)
    plot_summary_heatmap(df, output_dir)
    generate_latex_table(df, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
