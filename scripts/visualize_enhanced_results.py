"""
Visualization script for FedXChain enhanced experimental results
Creates publication-quality figures with error bars showing statistical validation
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set publication quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'

# Load results
results = {
    'Logistic\n(Breast Cancer)': pd.read_csv('results_enhanced/stats_breast_cancer_logistic.csv'),
    'MLP\n(Breast Cancer)': pd.read_csv('results_enhanced/stats_breast_cancer_mlp.csv'),
    'Random Forest\n(Breast Cancer)': pd.read_csv('results_enhanced/stats_breast_cancer_rf.csv'),
    'Logistic\n(Synthetic)': pd.read_csv('results_enhanced/stats_synthetic_logistic.csv')
}

# ===== Figure 1: Accuracy Comparison with Error Bars =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Extract final round results
final_results = {}
for name, df in results.items():
    final_row = df.iloc[-1]
    final_results[name] = {
        'accuracy_mean': final_row['global_accuracy_mean'],
        'accuracy_std': final_row['global_accuracy_std'],
        'nsds_mean': final_row['avg_nsds_mean'],
        'nsds_std': final_row['avg_nsds_std']
    }

# Plot 1: Final Accuracy Comparison
models = list(final_results.keys())
accuracies = [final_results[m]['accuracy_mean'] * 100 for m in models]
acc_stds = [final_results[m]['accuracy_std'] * 100 for m in models]

colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
bars = ax1.bar(range(len(models)), accuracies, yerr=acc_stds, 
               color=colors, alpha=0.7, capsize=8, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Model Configuration', fontweight='bold')
ax1.set_ylabel('Global Accuracy (%)', fontweight='bold')
ax1.set_title('Final Accuracy Comparison (5 runs)\nError bars show ± 1 standard deviation', 
              fontweight='bold', pad=15)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=0, ha='center')
ax1.set_ylim([0, 105])
ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
ax1.legend()

# Add value labels on bars
for i, (acc, std) in enumerate(zip(accuracies, acc_stds)):
    ax1.text(i, acc + std + 2, f'{acc:.1f}%\n±{std:.1f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 2: NSDS Comparison
nsds_means = [final_results[m]['nsds_mean'] for m in models]
nsds_stds = [final_results[m]['nsds_std'] for m in models]

bars2 = ax2.bar(range(len(models)), nsds_means, yerr=nsds_stds,
                color=colors, alpha=0.7, capsize=8, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Model Configuration', fontweight='bold')
ax2.set_ylabel('NSDS (Node-Specific Distribution Shift)', fontweight='bold')
ax2.set_title('Model Consistency Comparison (5 runs)\nLower NSDS = Better Consistency', 
              fontweight='bold', pad=15)
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=0, ha='center')

# Add value labels
for i, (nsds, std) in enumerate(zip(nsds_means, nsds_stds)):
    ax2.text(i, nsds + std + 0.02, f'{nsds:.3f}\n±{std:.3f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('results_enhanced/comparison_accuracy_nsds.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results_enhanced/comparison_accuracy_nsds.png")
plt.close()

# ===== Figure 2: Training Convergence with Confidence Intervals =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, df) in enumerate(results.items()):
    ax = axes[idx]
    
    rounds = df['rounds']
    acc_mean = df['global_accuracy_mean'] * 100
    acc_ci_low = df['global_accuracy_ci_low'] * 100
    acc_ci_high = df['global_accuracy_ci_high'] * 100
    
    # Plot mean line
    ax.plot(rounds, acc_mean, color=colors[idx], linewidth=2.5, marker='o', 
            markersize=6, label='Mean Accuracy')
    
    # Plot confidence interval
    ax.fill_between(rounds, acc_ci_low, acc_ci_high, 
                     color=colors[idx], alpha=0.2, label='95% CI')
    
    ax.set_xlabel('Federated Round', fontweight='bold')
    ax.set_ylabel('Global Accuracy (%)', fontweight='bold')
    ax.set_title(f'{name}\nConvergence with 95% CI', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

plt.tight_layout()
plt.savefig('results_enhanced/convergence_all_models.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results_enhanced/convergence_all_models.png")
plt.close()

# ===== Figure 3: NSDS Evolution Over Rounds =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, df) in enumerate(results.items()):
    ax = axes[idx]
    
    rounds = df['rounds']
    nsds_mean = df['avg_nsds_mean']
    nsds_ci_low = df['avg_nsds_ci_low']
    nsds_ci_high = df['avg_nsds_ci_high']
    
    # Plot mean line
    ax.plot(rounds, nsds_mean, color=colors[idx], linewidth=2.5, marker='s', 
            markersize=6, label='Mean NSDS')
    
    # Plot confidence interval (only if valid)
    valid_mask = pd.notna(nsds_ci_low) & pd.notna(nsds_ci_high)
    if valid_mask.any():
        ax.fill_between(rounds[valid_mask], 
                        nsds_ci_low[valid_mask], 
                        nsds_ci_high[valid_mask],
                        color=colors[idx], alpha=0.2, label='95% CI')
    
    ax.set_xlabel('Federated Round', fontweight='bold')
    ax.set_ylabel('NSDS', fontweight='bold')
    ax.set_title(f'{name}\nNSDS Evolution (Lower = Better)', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_enhanced/nsds_evolution_all_models.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results_enhanced/nsds_evolution_all_models.png")
plt.close()

# ===== Figure 4: Summary Table as Image =====
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create summary table
table_data = []
table_data.append(['Model', 'Dataset', 'Accuracy (%)', 'F1 Score (%)', 'NSDS', 'Runs'])

for name in models:
    clean_name = name.replace('\n', ' ')
    model_name = clean_name.split('(')[0].strip()
    dataset = clean_name.split('(')[1].replace(')', '').strip()
    
    df = results[name]
    final = df.iloc[-1]
    
    acc = f"{final['global_accuracy_mean']*100:.2f} ± {final['global_accuracy_std']*100:.2f}"
    f1 = f"{final['global_f1_mean']*100:.2f} ± {final['global_f1_std']*100:.2f}"
    nsds = f"{final['avg_nsds_mean']:.4f} ± {final['avg_nsds_std']:.4f}"
    
    table_data.append([model_name, dataset, acc, f1, nsds, '5'])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.18, 0.22, 0.22, 0.22, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    color = '#ecf0f1' if i % 2 == 0 else 'white'
    for j in range(6):
        table[(i, j)].set_facecolor(color)

ax.set_title('FedXChain Experimental Results Summary\n(5 Independent Runs per Configuration)', 
             fontweight='bold', fontsize=14, pad=20)

plt.savefig('results_enhanced/results_summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results_enhanced/results_summary_table.png")
plt.close()

# ===== Generate Statistics Summary =====
print("\n" + "="*70)
print("FEDXCHAIN ENHANCED EXPERIMENTAL RESULTS SUMMARY")
print("="*70)

for name, df in results.items():
    print(f"\n{name}:")
    final = df.iloc[-1]
    
    print(f"  Final Accuracy:  {final['global_accuracy_mean']*100:.2f}% ± {final['global_accuracy_std']*100:.2f}%")
    print(f"  Final F1 Score:  {final['global_f1_mean']*100:.2f}% ± {final['global_f1_std']*100:.2f}%")
    print(f"  Final NSDS:      {final['avg_nsds_mean']:.4f} ± {final['avg_nsds_std']:.4f}")
    
    # Calculate coefficient of variation
    cv_acc = (final['global_accuracy_std'] / final['global_accuracy_mean']) * 100
    print(f"  CV (Accuracy):   {cv_acc:.2f}% (lower is more reproducible)")

print("\n" + "="*70)
print("All visualizations saved in results_enhanced/")
print("="*70)
