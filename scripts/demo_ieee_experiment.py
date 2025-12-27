#!/usr/bin/env python3
"""
Quick Demo: IEEE-compliant FedXChain Experiment
Runs a small-scale demonstration of the experimental framework
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_ieee_experiment import FedXChainExperiment
import pandas as pd
import numpy as np

def main():
    print("\n" + "="*80)
    print("FedXChain IEEE Access - Quick Demo")
    print("="*80)
    print("\nThis demo runs a small-scale version of the full experimental suite:")
    print("  • 2 methods: FedAvg vs FedXChain")
    print("  • 3 scenarios: No attack, Label flip, Gaussian noise")
    print("  • 3 runs per configuration (instead of 10)")
    print("  • 5 rounds (instead of 20)")
    print("\nEstimated time: 3-5 minutes")
    print("="*80 + "\n")
    
    # Small configuration for demo
    config = {
        'dataset': 'synthetic',
        'n_samples': 500,      # Reduced from 1000
        'n_features': 20,
        'n_clients': 5,        # Reduced from 10
        'rounds': 5,           # Reduced from 20
        'n_runs': 3,           # Reduced from 10
        'local_epochs': 1,
        'dirichlet_alpha': 0.5,
        'shap_samples': 10,
        'trust_alpha': 0.4,
        'trust_beta': 0.3,
        'trust_gamma': 0.3,
        'seed': 42
    }
    
    # Initialize experiment
    experiment = FedXChainExperiment(config)
    
    # Run demo experiments
    demo_experiments = [
        # Baseline
        {'method': 'fedavg', 'malicious_nodes': [], 'attack_type': 'none'},
        {'method': 'fedxchain', 'malicious_nodes': [], 'attack_type': 'none'},
        
        # Label Flipping
        {'method': 'fedavg', 'malicious_nodes': [0], 'attack_type': 'label_flip', 'attack_intensity': 0.3},
        {'method': 'fedxchain', 'malicious_nodes': [0], 'attack_type': 'label_flip', 'attack_intensity': 0.3},
        
        # Gaussian Noise
        {'method': 'fedavg', 'malicious_nodes': [0], 'attack_type': 'gaussian_noise', 'attack_intensity': 0.5},
        {'method': 'fedxchain', 'malicious_nodes': [0], 'attack_type': 'gaussian_noise', 'attack_intensity': 0.5},
    ]
    
    all_results = []
    
    for exp_config in demo_experiments:
        for run in range(config['n_runs']):
            results = experiment.run_single_experiment(run, exp_config['method'], exp_config)
            all_results.extend(results)
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save results
    output_dir = Path('results_ieee')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'demo_results.csv'
    df_results.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("DEMO COMPLETE - SUMMARY RESULTS")
    print("="*80)
    
    # Get final round results
    final_round = df_results['round'].max()
    df_final = df_results[df_results['round'] == final_round]
    
    # Compute statistics
    summary = df_final.groupby(['method', 'attack_type']).agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std']
    }).round(4)
    
    print("\nFinal Round Performance (Round {}):\n".format(final_round))
    print(summary)
    
    # Compute improvements
    print("\n" + "-"*80)
    print("FedXChain Improvement over FedAvg:")
    print("-"*80)
    
    for attack_type in df_final['attack_type'].unique():
        df_attack = df_final[df_final['attack_type'] == attack_type]
        
        fedavg_acc = df_attack[df_attack['method'] == 'fedavg']['accuracy'].mean()
        fedxchain_acc = df_attack[df_attack['method'] == 'fedxchain']['accuracy'].mean()
        
        improvement = ((fedxchain_acc - fedavg_acc) / fedavg_acc * 100) if fedavg_acc > 0 else 0
        
        print(f"\nAttack: {attack_type:20} | FedAvg: {fedavg_acc:.4f} | FedXChain: {fedxchain_acc:.4f}")
        print(f"  → Improvement: {improvement:+.2f}%")
    
    print("\n" + "="*80)
    print("Results saved to: {}".format(output_file))
    print("="*80)
    
    print("\n📊 Next Steps:")
    print("  1. Run full experiments: python scripts/run_ieee_experiment.py --n_runs 10 --rounds 20")
    print("  2. Analyze results: python scripts/analyze_ieee_results.py --input results_ieee/demo_results.csv")
    print("  3. Review methodology: See IEEE_METHODOLOGY.md")
    print("\n")

if __name__ == '__main__':
    main()
