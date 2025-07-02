"""
Results converter to save Bayesian Optimization results in the exact format
expected by the notebook for direct comparison with baseline models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import torch


def convert_to_notebook_format(
    bo_results: Dict[str, Any],
    test_df: pd.DataFrame,
    targets: pd.DataFrame,
    pred_mean_history: List[np.ndarray],
    pred_var_history: List[np.ndarray],
    test_name: str,
    start_point: str,
    seed: int,
    model_name: str = "LVMOGP_SSVI"
) -> pd.DataFrame:
    """
    Convert BO results to the exact CSV format expected by the notebook.
    
    Args:
        bo_results: Results from bayesian_optimization_loop
        test_df: Test dataframe with BP, GC, PrimerPairReporter, Value
        targets: Targets dataframe
        pred_mean_history: List of prediction means for each BO iteration
        pred_var_history: List of prediction variances for each BO iteration  
        test_name: Experiment name (e.g., "many_r")
        start_point: Starting point strategy
        seed: Random seed
        model_name: Name of the model for comparison
        
    Returns:
        DataFrame in notebook format for direct comparison
    """
    
    chosen_indices = bo_results["chosen_indices"]
    surfaces_optimized = bo_results["surfaces_optimized"]
    
    # Create results list - each row is a data point added during BO
    results_rows = []
    
    for iteration, idx in enumerate(chosen_indices):
        # Get the test point that was selected
        selected_point = test_df.iloc[idx]
        
        # Get predictions for this iteration (before adding the point)
        if iteration < len(pred_mean_history):
            pred_mean = pred_mean_history[iteration][idx]
            pred_var = pred_var_history[iteration][idx]
        else:
            pred_mean = np.nan
            pred_var = np.nan
        
        # Get target for this surface
        ppr = selected_point['PrimerPairReporter']
        target_row = targets[targets['PrimerPairReporter'] == ppr]
        target_r = target_row['Target Rate'].iloc[0] if len(target_row) > 0 else np.nan
        
        # Calculate standardized values (using simple z-score for now)
        # Note: The notebook likely has specific standardization - may need adjustment
        r_mean = test_df['Value'].mean()
        r_std = test_df['Value'].std()
        stzd_r = (selected_point['Value'] - r_mean) / r_std
        
        # Calculate errors
        error_r = abs(selected_point['Value'] - pred_mean) if not np.isnan(pred_mean) else np.nan
        error_from_target_r = abs(selected_point['Value'] - target_r) if not np.isnan(target_r) else np.nan
        
        # Determine initial surface for "one_from_many" experiments
        if test_name.startswith("one_from_many_"):
            initial_surface = test_name.replace("one_from_many_", "").replace("_r", "")
        else:
            initial_surface = "many"  # For "many_r" experiments
        
        # Create row matching notebook format
        row = {
            # Input features
            'BP': selected_point['BP'],
            'GC': selected_point['GC'],
            'PrimerPairReporter': ppr,
            
            # Observed values
            'r': selected_point['Value'],  # Rate (main target)
            'stzd r': stzd_r,
            
            # Predictions  
            'r_mu': pred_mean,
            'r_sig2': pred_var,
            'r_mu_z': (pred_mean - r_mean) / r_std if not np.isnan(pred_mean) else np.nan,
            'r_sig2_z': pred_var / (r_std ** 2) if not np.isnan(pred_var) else np.nan,
            
            # Drift (not used in current setup, but notebook expects it)
            'm': np.nan,  # Drift parameter (not used)
            'stzd m': np.nan,
            'm_mu': np.nan,
            'm_sig2': np.nan,
            'm_mu_z': np.nan,
            'm_sig2_z': np.nan,
            
            # Expected Improvement
            'EI_z': bo_results["ei_values"][iteration] if iteration < len(bo_results["ei_values"]) else np.nan,
            
            # Sequence information
            'Sequence Name': f"{ppr}_{selected_point.get('Sequence Name', 'Unknown')}",
            
            # Targets
            'target r': target_r,
            'target r z': (target_r - r_mean) / r_std if not np.isnan(target_r) else np.nan,
            'target m': np.nan,  # Drift target (not used)
            'target m z': np.nan,
            
            # Model and experiment info
            'model': model_name,
            'iteration': iteration + 1,  # 1-indexed
            
            # Errors
            'error r': error_r,
            'error from target r z': (error_from_target_r - r_mean) / r_std if not np.isnan(error_from_target_r) else np.nan,
            'error from target r': error_from_target_r,
            'error r z': (error_r - r_mean) / r_std if not np.isnan(error_r) else np.nan,
            'error m': np.nan,  # Drift error (not used)
            'error from target m z': np.nan,
            'error from target m': np.nan,
            'error m z': np.nan,
            
            # Experiment metadata
            'initial_surface': initial_surface,
            'seed': seed
        }
        
        results_rows.append(row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_rows)
    
    # Ensure column order matches notebook exactly
    notebook_columns = [
        'BP', 'GC', 'PrimerPairReporter', 'r', 'stzd r', 'r_mu', 'r_sig2',
        'r_mu_z', 'r_sig2_z', 'm', 'stzd m', 'm_mu', 'm_sig2', 'm_mu_z',
        'm_sig2_z', 'EI_z', 'Sequence Name', 'target r', 'target r z',
        'target m', 'target m z', 'model', 'iteration', 'error r',
        'error from target r z', 'error from target r', 'error r z', 'error m',
        'error from target m z', 'error from target m', 'error m z',
        'initial_surface', 'seed'
    ]
    
    # Reorder columns to match notebook
    results_df = results_df.reindex(columns=notebook_columns)
    
    return results_df


def save_notebook_compatible_results(
    bo_results: Dict[str, Any],
    test_df: pd.DataFrame,
    targets: pd.DataFrame,
    pred_mean_history: List[np.ndarray],
    pred_var_history: List[np.ndarray],
    test_name: str,
    start_point: str,
    seed: int,
    save_path: Path,
    model_name: str = "LVMOGP_SSVI"
):
    """
    Save BO results in notebook-compatible CSV format.
    
    Args:
        bo_results: Results from bayesian_optimization_loop
        test_df: Test dataframe
        targets: Targets dataframe  
        pred_mean_history: Prediction means for each iteration
        pred_var_history: Prediction variances for each iteration
        test_name: Experiment name
        start_point: Starting point strategy
        seed: Random seed
        save_path: Directory to save results
        model_name: Model name for comparison
    """
    
    # Convert to notebook format
    notebook_df = convert_to_notebook_format(
        bo_results, test_df, targets, pred_mean_history, pred_var_history,
        test_name, start_point, seed, model_name
    )
    
    # Save CSV file with notebook-compatible name
    csv_filename = f"bayes_opt_{start_point}_{test_name}_seed_{seed}.csv"
    csv_path = save_path / csv_filename
    
    # Save with index for compatibility with notebook loading
    notebook_df.to_csv(csv_path, index=True)
    
    print(f"Saved notebook-compatible results to: {csv_path}")
    print(f"   Shape: {notebook_df.shape}")
    print(f"   Columns: {len(notebook_df.columns)} (matches notebook format)")
    
    return csv_path


def create_comparison_summary(results_dir: Path, baseline_csv: Path = None):
    """
    Create a summary comparing our results with baseline models.
    
    Args:
        results_dir: Directory containing our CSV results
        baseline_csv: Path to baseline results from notebook (optional)
    """
    
    # Find all our result CSV files
    our_csvs = list(results_dir.glob("bayes_opt_*.csv"))
    
    if not our_csvs:
        print("No CSV results found for comparison")
        return
    
    summary_data = []
    
    for csv_file in our_csvs:
        df = pd.read_csv(csv_file, index_col=0)
        
        # Extract experiment info
        filename = csv_file.stem
        parts = filename.split('_')
        start_point = parts[2] if len(parts) > 2 else "unknown"
        test_name = parts[3] if len(parts) > 3 else "unknown"
        seed = parts[-1] if parts[-1].startswith('seed') else "unknown"
        
        # Calculate summary metrics
        final_error = df['error from target r'].iloc[-1] if 'error from target r' in df.columns else np.nan
        final_regret = np.min(df['error from target r']) if 'error from target r' in df.columns else np.nan
        
        summary_data.append({
            'experiment': f"{test_name}_{start_point}",
            'seed': seed,
            'model': 'LVMOGP_SSVI',
            'n_iterations': len(df),
            'final_error_from_target': final_error,
            'best_error_from_target': final_regret,
            'file': csv_file.name
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = results_dir / "comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Created comparison summary: {summary_path}")
    print(f"   Ready for comparison with baseline models")
    
    return summary_path 