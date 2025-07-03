#!/usr/bin/env python
"""
Compare LVMOGP-SSVI results with original baseline results.
Loads both BO and CV results and provides comparison metrics.

Usage:
    python compare_with_baselines.py --bo_results gp_lvm_bo_run_results/latest_directory
    python compare_with_baselines.py --cv_results cross_validation_lvmogp_ssvi.csv
    python compare_with_baselines.py --auto_find   # Automatically find latest results
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_baseline_results():
    """Load original baseline results from results folder."""
    work_dir = Path.cwd()
    results_dir = work_dir / "results"
    
    # Load CV baselines
    cv_baseline = None
    cv_path = results_dir / "cross_validation.csv"
    if cv_path.exists():
        cv_baseline = pd.read_csv(cv_path, index_col=0)
        print(f"Loaded CV baseline: {len(cv_baseline)} rows")
    else:
        print(f"CV baseline not found at {cv_path}")
    
    # Load BO baselines
    bo_baselines = {}
    bo_files = [
        "bayes_opt_0_point_start_learning_1.csv",
        "bayes_opt_0_point_start_learning_many.csv", 
        "bayes_opt_centre_learning_1.csv",
        "bayes_opt_centre_learning_many.csv"
    ]
    
    for bo_file in bo_files:
        bo_path = results_dir / bo_file
        if bo_path.exists():
            try:
                bo_df = pd.read_csv(bo_path, index_col=0)
                bo_baselines[bo_file] = bo_df
                print(f"Loaded BO baseline {bo_file}: {len(bo_df)} rows")
            except Exception as e:
                print(f"Failed to load {bo_file}: {e}")
        else:
            print(f"BO baseline not found at {bo_path}")
    
    return cv_baseline, bo_baselines

def load_lvmogp_cv_results(cv_path=None):
    """Load LVMOGP-SSVI CV results."""
    work_dir = Path.cwd()
    
    if cv_path is None:
        # Try to find automatically
        cv_files = list(work_dir.glob("cross_validation_lvmogp_ssvi*.csv"))
        if cv_files:
            cv_path = cv_files[0]  # Use the first found
        else:
            print("No LVMOGP-SSVI CV results found")
            return None
    
    if Path(cv_path).exists():
        cv_results = pd.read_csv(cv_path, index_col=0)
        print(f"Loaded LVMOGP-SSVI CV results: {len(cv_results)} rows")
        return cv_results
    else:
        print(f"LVMOGP-SSVI CV results not found at {cv_path}")
        return None

def load_lvmogp_bo_results(bo_dir=None):
    """Load LVMOGP-SSVI BO results."""
    work_dir = Path.cwd()
    
    if bo_dir is None:
        # Try to find automatically - get latest directory
        bo_results_dirs = list((work_dir / "gp_lvm_bo_run_results").glob("*"))
        if bo_results_dirs:
            bo_dir = max(bo_results_dirs, key=lambda x: x.stat().st_mtime)  # Latest
        else:
            print("No LVMOGP-SSVI BO results found")
            return None
    
    bo_dir = Path(bo_dir)
    
    # Look for CSV files in the directory
    csv_files = list(bo_dir.glob("*_seed_*.csv"))
    if not csv_files:
        print(f"No BO CSV files found in {bo_dir}")
        return None
    
    bo_results = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, index_col=0)
            bo_results[csv_file.name] = df
            print(f"Loaded LVMOGP-SSVI BO result {csv_file.name}: {len(df)} rows")
        except Exception as e:
            print(f"Failed to load {csv_file}: {e}")
    
    return bo_results

def compare_cv_results(cv_baseline, cv_lvmogp):
    """Compare CV results between baseline models and LVMOGP-SSVI."""
    if cv_baseline is None or cv_lvmogp is None:
        print("Missing CV data for comparison")
        return
    
    print("\n" + "="*60)
    print("CROSS VALIDATION COMPARISON")
    print("="*60)
    
    # Model columns in baseline
    baseline_models = ['mo_indi', 'lmc', 'avg', 'lvm']
    lvmogp_model = 'lvm_ssvi'
    
    # Compare test RMSE across training percentages
    print("\nTest RMSE by Training Percentage:")
    print("-" * 40)
    
    for pct in sorted(cv_baseline['pct_train'].unique()):
        baseline_pct = cv_baseline[cv_baseline['pct_train'] == pct]
        lvmogp_pct = cv_lvmogp[cv_lvmogp['pct_train'] == pct]
        
        if len(lvmogp_pct) == 0:
            continue
            
        print(f"\nTraining {pct}%:")
        
        # Baseline model results
        for model in baseline_models:
            rmse_col = f'{model}_test_RMSE'
            if rmse_col in baseline_pct.columns:
                rmse_values = baseline_pct[rmse_col].dropna()
                if len(rmse_values) > 0:
                    print(f"  {model:8}: {rmse_values.mean():.4f} ± {rmse_values.std():.4f}")
        
        # LVMOGP-SSVI results
        lvmogp_rmse = lvmogp_pct[f'{lvmogp_model}_test_RMSE'].dropna()
        if len(lvmogp_rmse) > 0:
            print(f"  {lvmogp_model:8}: {lvmogp_rmse.mean():.4f} ± {lvmogp_rmse.std():.4f}")
    
    # Overall performance summary
    print(f"\n{'='*60}")
    print("OVERALL CV PERFORMANCE SUMMARY")
    print("="*60)
    
    overall_comparison = {}
    
    # Calculate overall metrics for baselines
    for model in baseline_models:
        rmse_col = f'{model}_test_RMSE'
        nlpd_col = f'{model}_test_NLPD'
        
        if rmse_col in cv_baseline.columns and nlpd_col in cv_baseline.columns:
            rmse_values = cv_baseline[rmse_col].dropna()
            nlpd_values = cv_baseline[nlpd_col].dropna()
            
            if len(rmse_values) > 0 and len(nlpd_values) > 0:
                overall_comparison[model] = {
                    'test_rmse_mean': rmse_values.mean(),
                    'test_rmse_std': rmse_values.std(),
                    'test_nlpd_mean': nlpd_values.mean(),
                    'test_nlpd_std': nlpd_values.std()
                }
    
    # Calculate overall metrics for LVMOGP-SSVI
    lvmogp_rmse = cv_lvmogp[f'{lvmogp_model}_test_RMSE'].dropna()
    lvmogp_nlpd = cv_lvmogp[f'{lvmogp_model}_test_NLPD'].dropna()
    
    if len(lvmogp_rmse) > 0 and len(lvmogp_nlpd) > 0:
        overall_comparison[lvmogp_model] = {
            'test_rmse_mean': lvmogp_rmse.mean(),
            'test_rmse_std': lvmogp_rmse.std(),
            'test_nlpd_mean': lvmogp_nlpd.mean(),
            'test_nlpd_std': lvmogp_nlpd.std()
        }
    
    # Display comparison table
    print(f"\n{'Model':<12} {'Test RMSE':<15} {'Test NLPD':<15}")
    print("-" * 45)
    
    for model, metrics in overall_comparison.items():
        rmse_str = f"{metrics['test_rmse_mean']:.4f}±{metrics['test_rmse_std']:.4f}"
        nlpd_str = f"{metrics['test_nlpd_mean']:.4f}±{metrics['test_nlpd_std']:.4f}"
        print(f"{model:<12} {rmse_str:<15} {nlpd_str:<15}")
    
    # Rank models
    print(f"\nModel Rankings (lower is better):")
    print("-" * 30)
    
    # RMSE ranking
    rmse_ranking = sorted(overall_comparison.items(), 
                         key=lambda x: x[1]['test_rmse_mean'])
    print("By Test RMSE:")
    for i, (model, metrics) in enumerate(rmse_ranking, 1):
        print(f"  {i}. {model}: {metrics['test_rmse_mean']:.4f}")
    
    # NLPD ranking  
    nlpd_ranking = sorted(overall_comparison.items(),
                         key=lambda x: x[1]['test_nlpd_mean'])
    print("\nBy Test NLPD:")
    for i, (model, metrics) in enumerate(nlpd_ranking, 1):
        print(f"  {i}. {model}: {metrics['test_nlpd_mean']:.4f}")

def compare_bo_results(bo_baselines, bo_lvmogp):
    """Compare BO results between baseline models and LVMOGP-SSVI."""
    if not bo_baselines or not bo_lvmogp:
        print("Missing BO data for comparison")
        return
    
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION COMPARISON")
    print("="*60)
    
    # Group LVMOGP results by experiment type
    lvmogp_grouped = {}
    for filename, df in bo_lvmogp.items():
        # Extract experiment info from filename
        if 'many_r' in filename:
            exp_type = 'many_surfaces'
        elif 'one_from_many' in filename:
            exp_type = 'one_surface'
        else:
            exp_type = 'unknown'
        
        if exp_type not in lvmogp_grouped:
            lvmogp_grouped[exp_type] = []
        lvmogp_grouped[exp_type].append(df)
    
    # Compare with baselines
    baseline_mapping = {
        'many_surfaces': ['bayes_opt_centre_learning_many.csv', 
                         'bayes_opt_0_point_start_learning_many.csv'],
        'one_surface': ['bayes_opt_centre_learning_1.csv',
                       'bayes_opt_0_point_start_learning_1.csv']
    }
    
    for exp_type, lvmogp_dfs in lvmogp_grouped.items():
        if exp_type == 'unknown':
            continue
            
        print(f"\n{exp_type.upper()} EXPERIMENTS:")
        print("-" * 40)
        
        # Combine LVMOGP results for this experiment type
        if lvmogp_dfs:
            lvmogp_combined = pd.concat(lvmogp_dfs, ignore_index=True)
            
            # Get final iteration results (highest iteration per seed/scenario)
            final_iter_lvmogp = lvmogp_combined.groupby(['seed']).tail(1)
            
            if len(final_iter_lvmogp) > 0:
                lvmogp_regret = final_iter_lvmogp['regret'].mean() if 'regret' in final_iter_lvmogp.columns else None
                lvmogp_rmse = final_iter_lvmogp['RMSE'].mean() if 'RMSE' in final_iter_lvmogp.columns else None
                lvmogp_nlpd = final_iter_lvmogp['NLPD'].mean() if 'NLPD' in final_iter_lvmogp.columns else None
                
                print(f"LVMOGP-SSVI:")
                if lvmogp_regret is not None:
                    print(f"  Regret: {lvmogp_regret:.6f}")
                if lvmogp_rmse is not None:
                    print(f"  RMSE:   {lvmogp_rmse:.6f}")
                if lvmogp_nlpd is not None:
                    print(f"  NLPD:   {lvmogp_nlpd:.6f}")
        
        # Compare with relevant baselines
        baseline_files = baseline_mapping.get(exp_type, [])
        for baseline_file in baseline_files:
            if baseline_file in bo_baselines:
                baseline_df = bo_baselines[baseline_file]
                
                # Get unique models in baseline
                if 'model' in baseline_df.columns:
                    models = baseline_df['model'].unique()
                    
                    print(f"\nBaseline models from {baseline_file}:")
                    for model in models:
                        model_df = baseline_df[baseline_df['model'] == model]
                        
                        # Get final iteration results per seed
                        if 'iteration' in model_df.columns and 'seed' in model_df.columns:
                            final_iter = model_df.groupby(['seed'])['iteration'].transform('max')
                            final_results = model_df[model_df['iteration'] == final_iter]
                        else:
                            final_results = model_df
                        
                        if len(final_results) > 0:
                            regret = final_results['regret'].mean() if 'regret' in final_results.columns else None
                            rmse = final_results['RMSE'].mean() if 'RMSE' in final_results.columns else None
                            nlpd = final_results['NLPD'].mean() if 'NLPD' in final_results.columns else None
                            
                            print(f"  {model}:")
                            if regret is not None:
                                print(f"    Regret: {regret:.6f}")
                            if rmse is not None:
                                print(f"    RMSE:   {rmse:.6f}")  
                            if nlpd is not None:
                                print(f"    NLPD:   {nlpd:.6f}")

def auto_find_latest_results():
    """Automatically find the latest LVMOGP-SSVI results."""
    work_dir = Path.cwd()
    
    # Find latest BO results
    bo_results_dir = work_dir / "gp_lvm_bo_run_results"
    latest_bo = None
    if bo_results_dir.exists():
        bo_dirs = [d for d in bo_results_dir.iterdir() if d.is_dir()]
        if bo_dirs:
            latest_bo = max(bo_dirs, key=lambda x: x.stat().st_mtime)
    
    # Find CV results
    cv_files = list(work_dir.glob("cross_validation_lvmogp_ssvi*.csv"))
    latest_cv = cv_files[0] if cv_files else None
    
    return latest_bo, latest_cv

def main():
    parser = argparse.ArgumentParser(description="Compare LVMOGP-SSVI with baseline results")
    parser.add_argument("--bo_results", type=Path, help="Path to BO results directory")
    parser.add_argument("--cv_results", type=Path, help="Path to CV results CSV")
    parser.add_argument("--auto_find", action="store_true", 
                       help="Automatically find latest results")
    
    args = parser.parse_args()
    
    # Load baseline results
    print("Loading baseline results...")
    cv_baseline, bo_baselines = load_baseline_results()
    
    # Load LVMOGP-SSVI results
    if args.auto_find:
        print("Auto-finding latest LVMOGP-SSVI results...")
        latest_bo, latest_cv = auto_find_latest_results()
        bo_lvmogp = load_lvmogp_bo_results(latest_bo)
        cv_lvmogp = load_lvmogp_cv_results(latest_cv)
    else:
        bo_lvmogp = load_lvmogp_bo_results(args.bo_results)
        cv_lvmogp = load_lvmogp_cv_results(args.cv_results)
    
    # Perform comparisons
    if cv_baseline is not None or cv_lvmogp is not None:
        compare_cv_results(cv_baseline, cv_lvmogp)
    
    if bo_baselines or bo_lvmogp:
        compare_bo_results(bo_baselines, bo_lvmogp)
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print("="*60)
    print("\nFor detailed analysis, use the plotting code from")
    print("notebooks/useful_notebook.ipynb with your results.")

if __name__ == "__main__":
    main() 