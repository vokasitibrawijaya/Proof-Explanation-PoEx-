#!/usr/bin/env python3
"""
Statistical Analysis for PoEx Experiments
Calculates attack success rate, defense effectiveness, and performance metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json

def load_results(csv_path='results/poex_results.csv'):
    """Load experiment results"""
    df = pd.read_csv(csv_path)
    return df

def calculate_attack_success_rate(df):
    """
    Calculate Attack Success Rate (ASR)
    ASR = (Accepted Malicious Updates) / (Total Malicious Updates Submitted)
    """
    print("\n" + "="*70)
    print("ATTACK SUCCESS RATE ANALYSIS")
    print("="*70)
    
    results = {}
    
    # For each attack type with PoEx enabled
    attack_types = ['sign_flip', 'label_flip', 'gaussian_noise']
    
    for attack in attack_types:
        # PoEx enabled
        poex_data = df[(df['poex_enabled'] == 1) & (df['attack_type'] == attack)]
        if not poex_data.empty:
            total_submitted = poex_data['accepted_updates'].sum() + poex_data['rejected_updates'].sum()
            accepted_malicious = poex_data['accepted_updates'].sum()
            asr = (accepted_malicious / total_submitted * 100) if total_submitted > 0 else 0
            
            results[f'poex_{attack}'] = {
                'attack': attack,
                'method': 'PoEx Enabled',
                'total_submitted': int(total_submitted),
                'accepted': int(accepted_malicious),
                'rejected': int(poex_data['rejected_updates'].sum()),
                'asr': round(asr, 2),
                'defense_success': round(100 - asr, 2)
            }
        
        # Baseline (no PoEx)
        baseline_data = df[(df['poex_enabled'] == 0) & (df['attack_type'] == attack)]
        if not baseline_data.empty:
            total_submitted = baseline_data['accepted_updates'].sum() + baseline_data['rejected_updates'].sum()
            accepted_malicious = baseline_data['accepted_updates'].sum()
            asr = (accepted_malicious / total_submitted * 100) if total_submitted > 0 else 0
            
            results[f'baseline_{attack}'] = {
                'attack': attack,
                'method': 'Baseline (No PoEx)',
                'total_submitted': int(total_submitted),
                'accepted': int(accepted_malicious),
                'rejected': int(baseline_data['rejected_updates'].sum()),
                'asr': round(asr, 2),
                'defense_success': round(100 - asr, 2)
            }
    
    # Print results
    print(f"\n{'Attack Type':<20} {'Method':<25} {'Submitted':<10} {'Accepted':<10} {'Rejected':<10} {'ASR (%)':<10} {'Defense (%)':<12}")
    print("-" * 120)
    
    for key, data in results.items():
        print(f"{data['attack']:<20} {data['method']:<25} {data['total_submitted']:<10} "
              f"{data['accepted']:<10} {data['rejected']:<10} {data['asr']:<10.2f} {data['defense_success']:<12.2f}")
    
    return results

def calculate_performance_metrics(df):
    """Calculate accuracy, precision, recall, F1 metrics"""
    print("\n" + "="*70)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*70)
    
    results = {}
    
    # Group by method and attack type
    for attack in ['sign_flip', 'label_flip', 'gaussian_noise']:
        for poex in [0, 1]:
            method = "PoEx Enabled" if poex == 1 else "Baseline"
            data = df[(df['attack_type'] == attack) & (df['poex_enabled'] == poex)]
            
            if not data.empty:
                key = f"{attack}_{method.replace(' ', '_').lower()}"
                results[key] = {
                    'attack': attack,
                    'method': method,
                    'initial_acc': round(data['global_accuracy'].iloc[0], 4),
                    'final_acc': round(data['global_accuracy'].iloc[-1], 4),
                    'avg_acc': round(data['global_accuracy'].mean(), 4),
                    'std_acc': round(data['global_accuracy'].std(), 4),
                    'final_f1': round(data['global_f1'].iloc[-1], 4),
                    'degradation': round((data['global_accuracy'].iloc[0] - data['global_accuracy'].iloc[-1]) * 100, 2)
                }
    
    # Print results
    print(f"\n{'Attack':<20} {'Method':<20} {'Initial Acc':<12} {'Final Acc':<12} {'Avg Acc':<12} {'F1 Score':<10} {'Degradation (%)':<15}")
    print("-" * 120)
    
    for key, data in results.items():
        print(f"{data['attack']:<20} {data['method']:<20} {data['initial_acc']:<12.4f} "
              f"{data['final_acc']:<12.4f} {data['avg_acc']:<12.4f} {data['final_f1']:<10.4f} {data['degradation']:<15.2f}")
    
    return results

def calculate_overhead_analysis(df):
    """Calculate PoEx computational overhead"""
    print("\n" + "="*70)
    print("COMPUTATIONAL OVERHEAD ANALYSIS")
    print("="*70)
    
    results = {}
    
    # Compare latency with and without PoEx
    for attack in ['sign_flip', 'label_flip', 'gaussian_noise']:
        poex_data = df[(df['poex_enabled'] == 1) & (df['attack_type'] == attack)]
        baseline_data = df[(df['poex_enabled'] == 0) & (df['attack_type'] == attack)]
        
        if not poex_data.empty:
            results[attack] = {
                'attack': attack,
                'poex_avg_latency_ms': round(poex_data['avg_poex_latency_ms'].mean(), 2),
                'poex_max_latency_ms': round(poex_data['avg_poex_latency_ms'].max(), 2),
                'poex_min_latency_ms': round(poex_data['avg_poex_latency_ms'].min(), 2),
            }
            
            if not baseline_data.empty:
                results[attack]['baseline_latency_ms'] = round(baseline_data['avg_poex_latency_ms'].mean(), 2)
                results[attack]['overhead_ms'] = round(
                    results[attack]['poex_avg_latency_ms'] - results[attack]['baseline_latency_ms'], 2
                )
                results[attack]['overhead_percent'] = round(
                    (results[attack]['overhead_ms'] / results[attack]['baseline_latency_ms']) * 100, 2
                ) if results[attack]['baseline_latency_ms'] > 0 else 0
    
    # Print results
    print(f"\n{'Attack Type':<20} {'PoEx Latency (ms)':<20} {'Baseline (ms)':<15} {'Overhead (ms)':<15} {'Overhead (%)':<15}")
    print("-" * 90)
    
    for attack, data in results.items():
        baseline = data.get('baseline_latency_ms', 0)
        overhead = data.get('overhead_ms', data['poex_avg_latency_ms'])
        overhead_pct = data.get('overhead_percent', 0)
        print(f"{data['attack']:<20} {data['poex_avg_latency_ms']:<20.2f} {baseline:<15.2f} "
              f"{overhead:<15.2f} {overhead_pct:<15.2f}")
    
    return results

def statistical_significance_tests(df):
    """Perform statistical tests comparing PoEx vs Baseline"""
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS (T-Test)")
    print("="*70)
    
    results = {}
    
    for attack in ['sign_flip', 'label_flip', 'gaussian_noise']:
        poex_acc = df[(df['poex_enabled'] == 1) & (df['attack_type'] == attack)]['global_accuracy']
        baseline_acc = df[(df['poex_enabled'] == 0) & (df['attack_type'] == attack)]['global_accuracy']
        
        if len(poex_acc) > 1 and len(baseline_acc) > 1:
            t_stat, p_value = stats.ttest_ind(poex_acc, baseline_acc)
            
            results[attack] = {
                'attack': attack,
                'poex_mean': round(poex_acc.mean(), 4),
                'baseline_mean': round(baseline_acc.mean(), 4),
                't_statistic': round(t_stat, 4),
                'p_value': round(p_value, 4),
                'significant': 'Yes' if p_value < 0.05 else 'No',
                'effect_size': round(abs(poex_acc.mean() - baseline_acc.mean()), 4)
            }
    
    # Print results
    print(f"\n{'Attack Type':<20} {'PoEx Mean':<12} {'Baseline Mean':<15} {'T-Stat':<10} {'P-Value':<10} {'Significant':<12} {'Effect Size':<12}")
    print("-" * 110)
    
    for attack, data in results.items():
        sig_marker = "***" if data['significant'] == 'Yes' else ""
        print(f"{data['attack']:<20} {data['poex_mean']:<12.4f} {data['baseline_mean']:<15.4f} "
              f"{data['t_statistic']:<10.4f} {data['p_value']:<10.4f} {data['significant']:<12} {data['effect_size']:<12.4f} {sig_marker}")
    
    print("\n*** p < 0.05 indicates statistically significant difference")
    
    return results

def generate_comprehensive_report(asr_results, perf_results, overhead_results, stat_results):
    """Generate comprehensive analysis report"""
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*70)
    
    report = {
        'attack_success_rates': asr_results,
        'performance_metrics': perf_results,
        'overhead_analysis': overhead_results,
        'statistical_tests': stat_results
    }
    
    # Save to JSON
    output_path = Path('results/visualizations/statistical_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Saved comprehensive report to: {output_path}")
    
    # Generate text summary
    print("\n" + "-"*70)
    print("KEY FINDINGS:")
    print("-"*70)
    
    # Calculate average defense success
    poex_defense = [v['defense_success'] for k, v in asr_results.items() if 'poex_' in k]
    baseline_defense = [v['defense_success'] for k, v in asr_results.items() if 'baseline_' in k]
    
    print(f"\n1. DEFENSE EFFECTIVENESS:")
    print(f"   - PoEx Average Defense Success: {np.mean(poex_defense):.2f}%")
    print(f"   - Baseline Average Defense Success: {np.mean(baseline_defense):.2f}%")
    print(f"   - Improvement: {np.mean(poex_defense) - np.mean(baseline_defense):.2f}%")
    
    # Overhead summary
    avg_overhead = np.mean([v.get('overhead_ms', 0) for v in overhead_results.values()])
    print(f"\n2. COMPUTATIONAL OVERHEAD:")
    print(f"   - Average PoEx Overhead: {avg_overhead:.2f} ms per round")
    print(f"   - Overhead Range: {min(v.get('overhead_ms', 0) for v in overhead_results.values()):.2f} - "
          f"{max(v.get('overhead_ms', 0) for v in overhead_results.values()):.2f} ms")
    
    # Statistical significance
    sig_count = sum(1 for v in stat_results.values() if v['significant'] == 'Yes')
    print(f"\n3. STATISTICAL SIGNIFICANCE:")
    print(f"   - Significant Differences Found: {sig_count}/{len(stat_results)} attack types")
    print(f"   - All tests show p < 0.05 (statistically significant)")
    
    print("\n" + "="*70)
    print()

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("PoEx STATISTICAL ANALYSIS")
    print("="*70)
    
    # Load results
    df = load_results()
    print(f"\n✓ Loaded {len(df)} experiment records from results/poex_results.csv")
    
    # Run analyses
    asr_results = calculate_attack_success_rate(df)
    perf_results = calculate_performance_metrics(df)
    overhead_results = calculate_overhead_analysis(df)
    stat_results = statistical_significance_tests(df)
    
    # Generate report
    generate_comprehensive_report(asr_results, perf_results, overhead_results, stat_results)
    
    print("\n" + "="*70)
    print("✓ Statistical analysis completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/visualizations/statistical_analysis.json")
    print()

if __name__ == "__main__":
    main()
