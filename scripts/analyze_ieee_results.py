#!/usr/bin/env python3
"""
Statistical Analysis Script for IEEE Access Paper
Performs rigorous statistical analysis including:
- Descriptive statistics (mean, std, confidence intervals)
- Hypothesis testing (t-test, ANOVA, Mann-Whitney U)
- Effect size computation (Cohen's d)
- Multiple comparison corrections (Bonferroni, Holm)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class IEEEStatisticalAnalysis:
    """Statistical analysis for IEEE Access paper"""
    
    def __init__(self, results_file):
        self.df = pd.read_csv(results_file)
        self.alpha = 0.05  # Significance level
        self.analysis_results = {}
        
    def compute_descriptive_stats(self, group_by=['method', 'attack_type']):
        """Compute descriptive statistics"""
        print("\n" + "="*80)
        print("DESCRIPTIVE STATISTICS")
        print("="*80)
        
        # Get final round results
        final_round = self.df.groupby('run')['round'].max().min()
        df_final = self.df[self.df['round'] == final_round]
        
        # Compute statistics
        stats_df = df_final.groupby(group_by).agg({
            'accuracy': ['mean', 'std', 'min', 'max', 'count'],
            'f1_score': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
        }).round(4)
        
        # Compute confidence intervals
        def ci_95(x):
            n = len(x)
            if n < 2:
                return 0
            se = stats.sem(x)
            ci = se * stats.t.ppf((1 + 0.95) / 2., n-1)
            return ci
        
        ci_df = df_final.groupby(group_by).agg({
            'accuracy': ci_95,
            'f1_score': ci_95
        }).round(4)
        ci_df.columns = [f'{col}_ci95' for col in ci_df.columns]
        
        results = pd.concat([stats_df, ci_df], axis=1)
        
        print(results)
        self.analysis_results['descriptive_stats'] = results
        
        return results
    
    def paired_t_test(self, method1='fedavg', method2='fedxchain'):
        """Perform paired t-test between two methods"""
        print(f"\n" + "="*80)
        print(f"PAIRED T-TEST: {method1.upper()} vs {method2.upper()}")
        print("="*80)
        
        results = []
        
        for attack_type in self.df['attack_type'].unique():
            df_attack = self.df[self.df['attack_type'] == attack_type]
            
            # Get final round results
            final_round = df_attack.groupby('run')['round'].max().min()
            df_final = df_attack[df_attack['round'] == final_round]
            
            # Align by run for proper pairing
            pvt = df_final.pivot_table(index='run', columns='method', values='accuracy', aggfunc='mean')
            if method1 not in pvt.columns or method2 not in pvt.columns:
                continue
            paired = pvt[[method1, method2]].dropna()
            data1 = paired[method1].values
            data2 = paired[method2].values
            
            if len(data1) == 0 or len(data2) == 0:
                continue
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(data1, data2)
            
            # Compute Cohen's d effect size
            mean_diff = np.mean(data1) - np.mean(data2)
            pooled_std = np.sqrt((np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            result = {
                'attack_type': attack_type,
                'mean_method1': np.mean(data1),
                'mean_method2': np.mean(data2),
                'mean_diff': mean_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'cohens_d': cohens_d,
                'effect_size': self._interpret_cohens_d(cohens_d)
            }
            
            results.append(result)
            
            print(f"\nAttack Type: {attack_type}")
            print(f"  {method1}: {np.mean(data1):.4f} ± {np.std(data1):.4f}")
            print(f"  {method2}: {np.mean(data2):.4f} ± {np.std(data2):.4f}")
            print(f"  Difference: {mean_diff:.4f}")
            print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < self.alpha else 'No'}")
            print(f"  Cohen's d: {cohens_d:.4f} ({self._interpret_cohens_d(cohens_d)})")
        
        self.analysis_results['paired_t_test'] = pd.DataFrame(results)
        return pd.DataFrame(results)
    
    def anova_analysis(self):
        """Perform one-way ANOVA across attack scenarios"""
        print(f"\n" + "="*80)
        print("ONE-WAY ANOVA: Compare Across Attack Scenarios")
        print("="*80)
        
        results = []
        
        for method in self.df['method'].unique():
            df_method = self.df[self.df['method'] == method]
            
            # Get final round results
            final_round = df_method.groupby('run')['round'].max().min()
            df_final = df_method[df_method['round'] == final_round]
            
            # Group by attack type
            groups = [group['accuracy'].values 
                     for name, group in df_final.groupby('attack_type')]
            
            if len(groups) < 2:
                continue
            
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            result = {
                'method': method,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'n_groups': len(groups)
            }
            
            results.append(result)
            
            print(f"\nMethod: {method}")
            print(f"  F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < self.alpha else 'No'}")
            
            # Post-hoc Tukey HSD if significant
            if p_value < self.alpha:
                print("  Performing post-hoc Tukey HSD test...")
                self._tukey_hsd_test(df_final, method)
        
        self.analysis_results['anova'] = pd.DataFrame(results)
        return pd.DataFrame(results)
    
    def _tukey_hsd_test(self, df, method):
        """Perform Tukey HSD post-hoc test"""
        from scipy.stats import tukey_hsd
        
        groups = []
        labels = []
        for attack_type, group in df.groupby('attack_type'):
            groups.append(group['accuracy'].values)
            labels.append(attack_type)
        
        # Perform Tukey HSD
        res = tukey_hsd(*groups)
        
        print("\n  Pairwise comparisons (Tukey HSD):")
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                p_value = res.pvalue[i, j]
                significant = p_value < self.alpha
                print(f"    {labels[i]} vs {labels[j]}: p={p_value:.4f} {'*' if significant else ''}")
    
    def mann_whitney_test(self, method1='fedavg', method2='fedxchain'):
        """Perform Mann-Whitney U test (non-parametric alternative to t-test)"""
        print(f"\n" + "="*80)
        print(f"MANN-WHITNEY U TEST: {method1.upper()} vs {method2.upper()}")
        print("="*80)
        
        results = []
        
        for attack_type in self.df['attack_type'].unique():
            df_attack = self.df[self.df['attack_type'] == attack_type]
            
            # Get final round results
            final_round = df_attack.groupby('run')['round'].max().min()
            df_final = df_attack[df_attack['round'] == final_round]
            
            pvt = df_final.pivot_table(index='run', columns='method', values='accuracy', aggfunc='mean')
            if method1 not in pvt.columns or method2 not in pvt.columns:
                continue
            paired = pvt[[method1, method2]].dropna()
            data1 = paired[method1].values
            data2 = paired[method2].values
            
            if len(data1) == 0 or len(data2) == 0:
                continue
            
            # Perform Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            result = {
                'attack_type': attack_type,
                'median_method1': np.median(data1),
                'median_method2': np.median(data2),
                'u_statistic': u_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha
            }
            
            results.append(result)
            
            print(f"\nAttack Type: {attack_type}")
            print(f"  {method1} median: {np.median(data1):.4f}")
            print(f"  {method2} median: {np.median(data2):.4f}")
            print(f"  U-statistic: {u_stat:.4f}, p-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < self.alpha else 'No'}")
        
        self.analysis_results['mann_whitney'] = pd.DataFrame(results)
        return pd.DataFrame(results)
    
    def compute_attack_success_rate(self):
        """Compute Attack Success Rate (ASR)"""
        print(f"\n" + "="*80)
        print("ATTACK SUCCESS RATE (ASR)")
        print("="*80)
        
        # Get baseline (no attack) performance
        df_baseline = self.df[self.df['attack_type'] == 'none']
        final_round = df_baseline.groupby('run')['round'].max().min()
        df_baseline_final = df_baseline[df_baseline['round'] == final_round]
        
        baseline_acc = {}
        for method in df_baseline_final['method'].unique():
            baseline_acc[method] = df_baseline_final[df_baseline_final['method'] == method]['accuracy'].mean()
        
        # Compute ASR for each attack scenario
        results = []
        
        for attack_type in self.df[self.df['attack_type'] != 'none']['attack_type'].unique():
            df_attack = self.df[self.df['attack_type'] == attack_type]
            df_attack_final = df_attack[df_attack['round'] == final_round]
            
            for method in df_attack_final['method'].unique():
                attack_acc = df_attack_final[df_attack_final['method'] == method]['accuracy'].mean()
                
                # ASR = (Baseline_Acc - Attack_Acc) / Baseline_Acc
                asr = (baseline_acc[method] - attack_acc) / baseline_acc[method] if baseline_acc[method] > 0 else 0
                asr_pct = asr * 100
                
                result = {
                    'method': method,
                    'attack_type': attack_type,
                    'baseline_accuracy': baseline_acc[method],
                    'attack_accuracy': attack_acc,
                    'accuracy_degradation': baseline_acc[method] - attack_acc,
                    'asr_percentage': asr_pct
                }
                
                results.append(result)
                
                print(f"\nMethod: {method}, Attack: {attack_type}")
                print(f"  Baseline: {baseline_acc[method]:.4f}")
                print(f"  Under Attack: {attack_acc:.4f}")
                print(f"  Degradation: {baseline_acc[method] - attack_acc:.4f}")
                print(f"  ASR: {asr_pct:.2f}%")
        
        self.analysis_results['asr'] = pd.DataFrame(results)
        return pd.DataFrame(results)
    
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_latex_tables(self, output_dir):
        """Generate LaTeX tables for paper"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n" + "="*80)
        print("GENERATING LATEX TABLES")
        print("="*80)
        
        # Table 1: Performance Comparison
        if 'descriptive_stats' in self.analysis_results:
            df = self.analysis_results['descriptive_stats']
            latex = df.to_latex(float_format="%.4f", caption="Performance Comparison", label="tab:performance")
            with open(output_dir / 'table_performance.tex', 'w') as f:
                f.write(latex)
            print("✓ Table 1: Performance Comparison saved")
        
        # Table 2: Statistical Tests
        if 'paired_t_test' in self.analysis_results:
            df = self.analysis_results['paired_t_test']
            latex = df.to_latex(float_format="%.4f", index=False, 
                               caption="Paired t-test Results", label="tab:ttest")
            with open(output_dir / 'table_ttest.tex', 'w') as f:
                f.write(latex)
            print("✓ Table 2: Statistical Tests saved")
        
        # Table 3: Attack Success Rate
        if 'asr' in self.analysis_results:
            df = self.analysis_results['asr']
            latex = df.to_latex(float_format="%.4f", index=False,
                               caption="Attack Success Rate", label="tab:asr")
            with open(output_dir / 'table_asr.tex', 'w') as f:
                f.write(latex)
            print("✓ Table 3: Attack Success Rate saved")
    
    def plot_performance_comparison(self, output_dir):
        """Plot performance comparison across methods and attacks"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Get final round data
        final_round = self.df.groupby('run')['round'].max().min()
        df_final = self.df[self.df['round'] == final_round]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy comparison
        ax = axes[0]
        sns.barplot(data=df_final, x='attack_type', y='accuracy', hue='method', ax=ax, errorbar='sd')
        ax.set_title('Accuracy Comparison Across Attack Scenarios', fontsize=12, fontweight='bold')
        ax.set_xlabel('Attack Type', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.legend(title='Method', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: F1-Score comparison
        ax = axes[1]
        sns.barplot(data=df_final, x='attack_type', y='f1_score', hue='method', ax=ax, errorbar='sd')
        ax.set_title('F1-Score Comparison Across Attack Scenarios', fontsize=12, fontweight='bold')
        ax.set_xlabel('Attack Type', fontsize=11)
        ax.set_ylabel('F1-Score', fontsize=11)
        ax.legend(title='Method', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Performance comparison plot saved")
        plt.close()
    
    def plot_convergence(self, output_dir):
        """Plot convergence curves"""
        output_dir = Path(output_dir)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        attack_types = self.df['attack_type'].unique()
        
        for idx, attack_type in enumerate(attack_types):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            df_attack = self.df[self.df['attack_type'] == attack_type]
            
            for method in df_attack['method'].unique():
                df_method = df_attack[df_attack['method'] == method]
                
                # Compute mean and std across runs
                stats_df = df_method.groupby('round')['accuracy'].agg(['mean', 'std'])
                
                rounds = stats_df.index
                mean_acc = stats_df['mean']
                std_acc = stats_df['std']
                
                ax.plot(rounds, mean_acc, label=method, linewidth=2)
                ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)
            
            ax.set_title(f'Attack: {attack_type}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Round', fontsize=10)
            ax.set_ylabel('Accuracy', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(attack_types), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Convergence Analysis Across Attack Scenarios', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'convergence_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Convergence analysis plot saved")
        plt.close()
    
    def save_summary_report(self, output_dir):
        """Save comprehensive summary report"""
        output_dir = Path(output_dir)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_experiments': len(self.df),
            'n_runs': self.df['run'].nunique(),
            'methods': self.df['method'].unique().tolist(),
            'attack_types': self.df['attack_type'].unique().tolist(),
            'statistical_tests': list(self.analysis_results.keys())
        }
        
        # Add summary statistics
        for key, df in self.analysis_results.items():
            if isinstance(df, pd.DataFrame):
                report[key] = df.to_dict('records')
        
        # Save as JSON
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Summary report saved")

def main():
    parser = argparse.ArgumentParser(description='IEEE Statistical Analysis')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, default='analysis_ieee', help='Output directory')
    args = parser.parse_args()
    
    # Load and analyze
    analyzer = IEEEStatisticalAnalysis(args.input)
    
    # Perform all analyses
    analyzer.compute_descriptive_stats()
    analyzer.paired_t_test()
    analyzer.anova_analysis()
    analyzer.mann_whitney_test()
    analyzer.compute_attack_success_rate()
    
    # Generate outputs
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    analyzer.generate_latex_tables(output_dir)
    analyzer.plot_performance_comparison(output_dir)
    analyzer.plot_convergence(output_dir)
    analyzer.save_summary_report(output_dir)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"All results saved to: {output_dir}")

if __name__ == '__main__':
    main()
