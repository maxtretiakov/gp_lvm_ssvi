#!/usr/bin/env python
"""
Cross Validation with LVMOGP-SSVI - exactly replicating the notebook's CV setup.

Evaluates predictive performance (RMSE, NLPD) across different training set sizes
to compare with baseline models (MOGP, LMC, AvgGP, original LVMOGP).

Example:
  $ python run_cv_gp_lvm_ssvi.py --config cv_ssvi_configs/original_cv_ssvi_config.yaml
"""
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import dataclasses
import datetime
import json

from src.bayesian_optimization.config_helper import load_full_config
from src.bayesian_optimization.targets_helper import load_targets
from src.bayesian_optimization.metrics_helper import get_nlpd, get_squared_error
from src.lvmogp.lvmogp_ssvi import LVMOGP_SSVI_Torch


def prepare_cv_data(train_df, test_df, Q=12, device=None):
    """
    Prepare data for LVMOGP cross validation (simpler than BO - no targets needed).
    
    Args:
        train_df: Training dataframe with BP, GC, PrimerPairReporter, Value
        test_df: Test dataframe with BP, GC, PrimerPairReporter, Value  
        Q: Latent dimension
        device: torch device
        
    Returns:
        Dict with prepared tensors for LVMOGP
    """
    device = device or torch.device('cpu')
    
    # Extract and standardize features
    X_train = train_df[['BP', 'GC']].values.astype(np.float32)
    X_test = test_df[['BP', 'GC']].values.astype(np.float32)
    
    # Standardize features globally
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    # Get surface mappings
    all_surfaces = sorted(set(train_df['PrimerPairReporter'].unique()) | 
                         set(test_df['PrimerPairReporter'].unique()))
    ppr_to_idx = {ppr: idx for idx, ppr in enumerate(all_surfaces)}
    n_surfaces = len(all_surfaces)
    
    print(f"Cross validation surfaces: {n_surfaces}")
    print(f"Sample surfaces: {all_surfaces[:3]}...")
    
    # Convert to surface indices
    fn_train = train_df['PrimerPairReporter'].map(ppr_to_idx).values
    fn_test = test_df['PrimerPairReporter'].map(ppr_to_idx).values
    
    # Get target values
    Y_train = train_df['Value'].values.astype(np.float32).reshape(-1, 1)
    Y_test = test_df['Value'].values.astype(np.float32).reshape(-1, 1)
    
    # Standardize targets globally
    Y_mean = Y_train.mean()
    Y_std = Y_train.std()
    Y_train_std = (Y_train - Y_mean) / Y_std
    Y_test_std = (Y_test - Y_mean) / Y_std
    
    # Initialize latent variables H (per surface)
    H_mean = torch.randn(n_surfaces, Q, device=device) * 0.1
    H_var = torch.ones(n_surfaces, Q, device=device) * 0.1
    
    # Convert to tensors
    data_dict = {
        'X_train': torch.tensor(X_train, device=device),
        'Y_train': torch.tensor(Y_train_std, device=device),
        'fn_train': torch.tensor(fn_train, device=device, dtype=torch.long),
        'H_mean': H_mean,
        'H_var': H_var,
        'X_test': torch.tensor(X_test, device=device),
        'Y_test': torch.tensor(Y_test_std, device=device),
        'fn_test': torch.tensor(fn_test, device=device, dtype=torch.long),
        'ppr_to_idx': ppr_to_idx,
        'n_surfaces': n_surfaces,
        'Y_mean': Y_mean,
        'Y_std': Y_std,
        'Y_test_original': Y_test  # Keep original scale for metrics
    }
    
    return data_dict


def run_single_cv_fold(train_df, test_df, config, device):
    """
    Run a single cross validation fold.
    
    Returns:
        Dict with train/test RMSE and NLPD metrics
    """
    
    # Prepare data
    data_dict = prepare_cv_data(train_df, test_df, Q=config.q_latent, device=device)
    
    print(f"Training LVMOGP on {len(train_df)} points...")
    
    # Create and train LVMOGP model
    model = LVMOGP_SSVI_Torch(
        data=data_dict['Y_train'],
        X_data=data_dict['X_train'], 
        X_data_fn=data_dict['fn_train'].unsqueeze(-1),
        H_data_mean=data_dict['H_mean'],
        H_data_var=data_dict['H_var'],
        num_inducing_variables=config.inducing.n_inducing,
        device=device
    )
    
    # Train the model
    model.ssvi_train(config)
    
    print(f"Making predictions on {len(test_df)} test points...")
    
    # Make predictions on train set (for training metrics)
    train_pred_mean, train_pred_var = model.predict_y((data_dict['X_train'], data_dict['fn_train']))
    train_pred_mean = train_pred_mean.squeeze(-1).detach().cpu().numpy()
    train_pred_var = train_pred_var.squeeze(-1).detach().cpu().numpy()
    
    # Convert back to original scale
    train_pred_mean_orig = train_pred_mean * data_dict['Y_std'] + data_dict['Y_mean'] 
    train_pred_var_orig = train_pred_var * (data_dict['Y_std'] ** 2)
    train_true_orig = train_df['Value'].values
    
    # Make predictions on test set
    test_pred_mean, test_pred_var = model.predict_y((data_dict['X_test'], data_dict['fn_test']))
    test_pred_mean = test_pred_mean.squeeze(-1).detach().cpu().numpy()
    test_pred_var = test_pred_var.squeeze(-1).detach().cpu().numpy()
    
    # Convert back to original scale  
    test_pred_mean_orig = test_pred_mean * data_dict['Y_std'] + data_dict['Y_mean']
    test_pred_var_orig = test_pred_var * (data_dict['Y_std'] ** 2)
    test_true_orig = test_df['Value'].values
    
    # Calculate metrics on original scale
    train_nlpd = get_nlpd(train_pred_mean_orig, train_pred_var_orig, train_true_orig)
    train_squared_error = get_squared_error(train_pred_mean_orig, train_true_orig)
    train_rmse = np.sqrt(np.mean(train_squared_error))
    train_nlpd_mean = np.mean(train_nlpd)
    
    test_nlpd = get_nlpd(test_pred_mean_orig, test_pred_var_orig, test_true_orig)
    test_squared_error = get_squared_error(test_pred_mean_orig, test_true_orig)
    test_rmse = np.sqrt(np.mean(test_squared_error))
    test_nlpd_mean = np.mean(test_nlpd)
    
    # Calculate standardized metrics (z-scores)
    all_true = np.concatenate([train_true_orig, test_true_orig])
    global_mean = np.mean(all_true)
    global_std = np.std(all_true)
    
    train_rmse_z = train_rmse / global_std
    train_nlpd_z = train_nlpd_mean  # NLPD already dimensionless
    test_rmse_z = test_rmse / global_std  
    test_nlpd_z = test_nlpd_mean
    
    metrics = {
        'train_RMSE': train_rmse,
        'train_NLPD': train_nlpd_mean,
        'train_RMSE_z': train_rmse_z,
        'train_NLPD_z': train_nlpd_z,
        'test_RMSE': test_rmse,
        'test_NLPD': test_nlpd_mean,
        'test_RMSE_z': test_rmse_z,
        'test_NLPD_z': test_nlpd_z,
        'n_train': len(train_df),
        'n_test': len(test_df)
    }
    
    print(f"CV Metrics - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"             Train NLPD: {train_nlpd_mean:.4f}, Test NLPD: {test_nlpd_mean:.4f}")
    
    return metrics


def run_cross_validation(config_path, output_file="cross_validation_results.csv"):
    """
    Run cross validation across multiple seeds and training percentages.
    """
    
    # Load config using proper GPSSVIConfig structure (same as BO script)
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create proper GPSSVIConfig from the gp_ssvi section
    from src.bayesian_optimization.config_helper import _to_dataclass
    from src.gp_dataclasses import GPSSVIConfig
    
    gp_cfg = _to_dataclass(GPSSVIConfig, cfg['gp_ssvi'])
    cv_cfg_dict = cfg['cv']
    
    work_dir = Path.cwd()
    device = torch.device(gp_cfg.device)
    
    # Load full data
    data = pd.read_csv(work_dir / "data" / "data.csv", index_col=[0])
    
    print(f"Running Cross Validation with LVMOGP-SSVI")
    print(f"Device: {device}")
    print(f"Seeds: {cv_cfg_dict['seeds']}")
    print(f"Training percentages: {cv_cfg_dict['train_percentages']}")
    
    # Results storage
    all_results = []
    
    total_folds = len(cv_cfg_dict['seeds']) * len(cv_cfg_dict['train_percentages'])
    current_fold = 0
    
    for seed in cv_cfg_dict['seeds']:
        for pct_train in cv_cfg_dict['train_percentages']:
            current_fold += 1
            
            print(f"\nCV Fold {current_fold}/{total_folds}: seed={seed}, pct_train={pct_train}")
            
            # Load train/test split
            train_file = work_dir / f"data/cross_validation/seed_{seed}_pct_train_{pct_train}/train.txt"
            test_file = work_dir / f"data/cross_validation/seed_{seed}_pct_train_{pct_train}/test.txt"
            
            if not train_file.exists() or not test_file.exists():
                print(f"WARNING: Missing CV split files for seed={seed}, pct_train={pct_train}")
                continue
            
            # Load train/test locations
            with train_file.open() as f:
                train_locs = [line.strip() for line in f]
            with test_file.open() as f:
                test_locs = [line.strip() for line in f]
            
            # Filter dataframes
            train_df = data[data['PrimerPairReporterBPGC'].isin(train_locs)]
            test_df = data[data['PrimerPairReporterBPGC'].isin(test_locs)]
            
            if len(train_df) == 0 or len(test_df) == 0:
                print(f"WARNING: Empty train ({len(train_df)}) or test ({len(test_df)}) set")
                continue
            
            print(f"Train: {len(train_df)} points, Test: {len(test_df)} points")
            
            try:
                # Run CV fold
                fold_metrics = run_single_cv_fold(train_df, test_df, gp_cfg, device)
                
                # Add experiment metadata
                fold_result = {
                    'seed': seed,
                    'pct_train': pct_train,
                    'param': 'r',  # We only optimize 'r' (rate), not 'm' (drift)
                    'no test points': fold_metrics['n_test'],
                    'no train points': fold_metrics['n_train'],
                    
                    # Our LVMOGP-SSVI results (following notebook naming convention)
                    'lvm_ssvi_train_RMSE': fold_metrics['train_RMSE'],
                    'lvm_ssvi_train_NLPD': fold_metrics['train_NLPD'], 
                    'lvm_ssvi_train_RMSE_z': fold_metrics['train_RMSE_z'],
                    'lvm_ssvi_train_NLPD_z': fold_metrics['train_NLPD_z'],
                    'lvm_ssvi_test_RMSE': fold_metrics['test_RMSE'],
                    'lvm_ssvi_test_NLPD': fold_metrics['test_NLPD'],
                    'lvm_ssvi_test_RMSE_z': fold_metrics['test_RMSE_z'],
                    'lvm_ssvi_test_NLPD_z': fold_metrics['test_NLPD_z']
                }
                
                all_results.append(fold_result)
                
            except Exception as e:
                print(f"ERROR in fold seed={seed}, pct_train={pct_train}: {e}")
                continue
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) > 0:
        # Save results
        output_path = work_dir / output_file
        results_df.to_csv(output_path, index=True)
        
        print(f"\nCross Validation Complete!")
        print(f"Total successful folds: {len(results_df)}")
        print(f"Results saved to: {output_path}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Test RMSE: {results_df['lvm_ssvi_test_RMSE'].mean():.4f} ± {results_df['lvm_ssvi_test_RMSE'].std():.4f}")
        print(f"Test NLPD: {results_df['lvm_ssvi_test_NLPD'].mean():.4f} ± {results_df['lvm_ssvi_test_NLPD'].std():.4f}")
        
        return results_df
    else:
        print("ERROR: No successful CV folds completed!")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, 
                       help="YAML config with gp_ssvi + cv sections")
    parser.add_argument("--output", type=str, default="cross_validation_lvmogp_ssvi.csv",
                       help="Output CSV filename")
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        exit(1)
    
    run_cross_validation(args.config, args.output) 