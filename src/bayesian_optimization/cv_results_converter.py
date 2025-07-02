"""
Cross Validation Results Converter for LVMOGP-SSVI.

Converts CV results to the exact format expected by the notebook for 
direct comparison with baseline models (MOGP, LMC, AvgGP, original LVMOGP).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def convert_cv_results_to_notebook_format(
    our_results_csv: Path,
    baseline_results_csv: Optional[Path] = None,
    output_csv: Path = None
) -> pd.DataFrame:
    """
    Convert our CV results to match the exact notebook baseline format.
    
    Args:
        our_results_csv: Path to our LVMOGP-SSVI CV results
        baseline_results_csv: Path to notebook baseline results (optional)
        output_csv: Path to save combined results (optional)
        
    Returns:
        DataFrame in notebook format with our results added
    """
    
    # Load our results
    our_df = pd.read_csv(our_results_csv, index_col=0)
    
    print(f"Loaded our CV results: {len(our_df)} folds")
    print(f"Our columns: {our_df.columns.tolist()}")
    
    # Load baseline results if provided
    if baseline_results_csv and baseline_results_csv.exists():
        baseline_df = pd.read_csv(baseline_results_csv, index_col=0)
        print(f"Loaded baseline results: {len(baseline_df)} folds")
        print(f"Baseline columns: {baseline_df.columns.tolist()}")
        
        # Start with baseline data
        combined_df = baseline_df.copy()
    else:
        print("No baseline results provided - creating new format")
        # Create empty DataFrame with expected structure
        combined_df = pd.DataFrame()
    
    # Map our results to notebook format
    notebook_rows = []
    
    for _, row in our_df.iterrows():
        # Create row matching notebook baseline format exactly
        notebook_row = {
            # Basic info
            'seed': row['seed'],
            'pct_train': row['pct_train'], 
            'param': row['param'],
            'no test points': row['no test points'],
            'no train points': row['no train points'],
            
            # Our LVMOGP-SSVI results (add to baseline models)
            'lvm_ssvi_test_RMSE': row['lvm_ssvi_test_RMSE'],
            'lvm_ssvi_test_NLPD': row['lvm_ssvi_test_NLPD'],
            'lvm_ssvi_test_RMSE_z': row['lvm_ssvi_test_RMSE_z'],
            'lvm_ssvi_test_NLPD_z': row['lvm_ssvi_test_NLPD_z'],
            'lvm_ssvi_train_RMSE': row['lvm_ssvi_train_RMSE'],
            'lvm_ssvi_train_NLPD': row['lvm_ssvi_train_NLPD'],
            'lvm_ssvi_train_RMSE_z': row['lvm_ssvi_train_RMSE_z'],
            'lvm_ssvi_train_NLPD_z': row['lvm_ssvi_train_NLPD_z']
        }
        
        # If we have baseline data, check for matching fold
        if len(combined_df) > 0:
            # Find matching baseline row (same seed, pct_train, param)
            mask = (
                (combined_df['seed'] == row['seed']) & 
                (combined_df['pct_train'] == row['pct_train']) &
                (combined_df['param'] == row['param'])
            )
            
            if mask.any():
                # Update existing row with our results
                idx = combined_df[mask].index[0]
                for col, val in notebook_row.items():
                    combined_df.loc[idx, col] = val
            else:
                # Add new row (seed/pct_train combination not in baseline)
                new_row = notebook_row.copy()
                # Fill missing baseline columns with NaN
                for col in combined_df.columns:
                    if col not in new_row:
                        new_row[col] = np.nan
                
                # Add to dataframe
                combined_df = pd.concat([combined_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # No baseline data - just add our results
            notebook_rows.append(notebook_row)
    
    # If no baseline data, create DataFrame from our results
    if len(combined_df) == 0:
        combined_df = pd.DataFrame(notebook_rows)
        
        # Add placeholder columns for baseline models (filled with NaN)
        baseline_cols = [
            'mo_indi_test_RMSE', 'mo_indi_test_NLPD', 'mo_indi_test_RMSE_z', 'mo_indi_test_NLPD_z',
            'lmc_test_RMSE', 'lmc_test_NLPD', 'lmc_test_RMSE_z', 'lmc_test_NLPD_z',
            'avg_test_RMSE', 'avg_test_NLPD', 'avg_test_RMSE_z', 'avg_test_NLPD_z',
            'lvm_test_RMSE', 'lvm_test_NLPD', 'lvm_test_RMSE_z', 'lvm_test_NLPD_z',
            'mo_indi_train_RMSE', 'mo_indi_train_NLPD', 'mo_indi_train_RMSE_z', 'mo_indi_train_NLPD_z',
            'lmc_train_RMSE', 'lmc_train_NLPD', 'lmc_train_RMSE_z', 'lmc_train_NLPD_z',
            'avg_train_RMSE', 'avg_train_NLPD', 'avg_train_RMSE_z', 'avg_train_NLPD_z',
            'lvm_train_RMSE', 'lvm_train_NLPD', 'lvm_train_RMSE_z', 'lvm_train_NLPD_z'
        ]
        
        for col in baseline_cols:
            if col not in combined_df.columns:
                combined_df[col] = np.nan
    
    # Ensure standard index name
    combined_df.index.name = None
    
    # Sort by seed, pct_train for consistent ordering
    combined_df = combined_df.sort_values(['seed', 'pct_train', 'param']).reset_index(drop=True)
    
    print(f"\nCombined results: {len(combined_df)} folds")
    print(f"Final columns: {len(combined_df.columns)}")
    
    # Save combined results if output path provided
    if output_csv:
        combined_df.to_csv(output_csv, index=True)
        print(f"Saved combined CV results to: {output_csv}")
    
    return combined_df


def create_cv_comparison_summary(cv_results_csv: Path) -> pd.DataFrame:
    """
    Create a summary of CV results for easy comparison.
    
    Args:
        cv_results_csv: Path to CV results in notebook format
        
    Returns:
        Summary DataFrame with mean ± std for each model
    """
    
    df = pd.read_csv(cv_results_csv, index_col=0)
    
    # Identify model columns
    model_names = []
    for col in df.columns:
        if '_test_RMSE' in col and not col.endswith('_z'):
            model_name = col.replace('_test_RMSE', '')
            model_names.append(model_name)
    
    print(f"Found models: {model_names}")
    
    # Calculate summary statistics
    summary_data = []
    
    for model in model_names:
        test_rmse_col = f"{model}_test_RMSE"
        test_nlpd_col = f"{model}_test_NLPD"
        
        if test_rmse_col in df.columns and test_nlpd_col in df.columns:
            # Calculate mean and std across all CV folds
            rmse_mean = df[test_rmse_col].mean()
            rmse_std = df[test_rmse_col].std()
            nlpd_mean = df[test_nlpd_col].mean()
            nlpd_std = df[test_nlpd_col].std()
            
            summary_data.append({
                'model': model,
                'test_RMSE_mean': rmse_mean,
                'test_RMSE_std': rmse_std,
                'test_RMSE_summary': f"{rmse_mean:.4f} ± {rmse_std:.4f}",
                'test_NLPD_mean': nlpd_mean,
                'test_NLPD_std': nlpd_std,
                'test_NLPD_summary': f"{nlpd_mean:.4f} ± {nlpd_std:.4f}",
                'n_folds': df[test_rmse_col].notna().sum()
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('test_RMSE_mean')  # Sort by RMSE performance
    
    return summary_df


def load_and_compare_cv_results(
    our_results_path: Path,
    baseline_results_path: Optional[Path] = None,
    output_dir: Path = None
):
    """
    Complete workflow: load our results, combine with baselines, create summary.
    
    Args:
        our_results_path: Path to our LVMOGP-SSVI CV results
        baseline_results_path: Path to notebook baseline results (optional)
        output_dir: Directory to save output files
    """
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert and combine results
    combined_csv = output_dir / "cross_validation_combined.csv" if output_dir else None
    combined_df = convert_cv_results_to_notebook_format(
        our_results_csv=our_results_path,
        baseline_results_csv=baseline_results_path,
        output_csv=combined_csv
    )
    
    # Create summary
    summary_df = create_cv_comparison_summary(combined_csv if combined_csv else our_results_path)
    
    if output_dir:
        summary_csv = output_dir / "cv_comparison_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved CV summary to: {summary_csv}")
    
    # Print summary
    print(f"\nCross Validation Performance Summary:")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:15} | RMSE: {row['test_RMSE_summary']:15} | NLPD: {row['test_NLPD_summary']:15} | Folds: {row['n_folds']:3}")
    
    return combined_df, summary_df


if __name__ == "__main__":
    # Example usage
    work_dir = Path.cwd()
    
    # Look for our CV results
    our_results = work_dir / "cross_validation_lvmogp_ssvi.csv"
    
    if our_results.exists():
        print(f"Processing CV results: {our_results}")
        
        # Look for baseline results from notebook
        baseline_results = work_dir / "results" / "cross_validation.csv"
        if not baseline_results.exists():
            baseline_results = None
            print("No baseline results found - will create standalone format")
        
        # Process results
        output_dir = work_dir / "cv_analysis"
        combined_df, summary_df = load_and_compare_cv_results(
            our_results_path=our_results,
            baseline_results_path=baseline_results,
            output_dir=output_dir
        )
        
        print(f"\nProcessing complete! Check {output_dir} for results.")
        
    else:
        print(f"No CV results found at: {our_results}")
        print("Run cross validation first:")
        print("python run_cv_gp_lvm_ssvi.py --config cv_ssvi_configs/quick_test_cv_config.yaml") 