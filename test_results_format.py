#!/usr/bin/env python
"""
Quick test to validate that our results format matches the notebook exactly.
"""

import pandas as pd
from pathlib import Path

def test_notebook_format_compatibility():
    """Test that our results format matches the notebook."""
    
    # Expected columns from notebook (Cell 28 output)
    expected_columns = [
        'BP', 'GC', 'PrimerPairReporter', 'r', 'stzd r', 'r_mu', 'r_sig2',
        'r_mu_z', 'r_sig2_z', 'm', 'stzd m', 'm_mu', 'm_sig2', 'm_mu_z',
        'm_sig2_z', 'EI_z', 'Sequence Name', 'target r', 'target r z',
        'target m', 'target m z', 'model', 'iteration', 'error r',
        'error from target r z', 'error from target r', 'error r z', 'error m',
        'error from target m z', 'error from target m', 'error m z',
        'initial_surface', 'seed'
    ]
    
    print("Notebook Format Compatibility Test")
    print(f"Expected columns ({len(expected_columns)}): {expected_columns[:5]}...")
    
    # Check if we have any result files to validate
    results_dir = Path("gp_lvm_bo_run_results")
    csv_files = []
    
    if results_dir.exists():
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                csv_files.extend(subdir.glob("bayes_opt_*.csv"))
    
    if csv_files:
        print(f"\nFound {len(csv_files)} result CSV files")
        
        # Test the first file
        test_file = csv_files[0]
        print(f"Testing: {test_file}")
        
        df = pd.read_csv(test_file, index_col=0)
        actual_columns = df.columns.tolist()
        
        print(f"Actual columns ({len(actual_columns)}): {actual_columns[:5]}...")
        
        # Check exact match
        if actual_columns == expected_columns:
            print("PERFECT MATCH: Column structure matches notebook exactly")
        else:
            print("MISMATCH: Column structure differs")
            missing = set(expected_columns) - set(actual_columns)
            extra = set(actual_columns) - set(expected_columns)
            if missing:
                print(f"Missing columns: {missing}")
            if extra:
                print(f"Extra columns: {extra}")
                
        # Check data types and values
        print(f"\nSample data preview:")
        print(df.head())
        
        # Check key columns exist and have reasonable values
        key_checks = {
            'BP': (0, 100),           # Base pair percentage
            'GC': (0, 100),           # GC content percentage  
            'iteration': (1, 100),    # BO iteration number
            'model': ['LVMOGP_SSVI']  # Our model name
        }
        
        for col, expected_range in key_checks.items():
            if col in df.columns:
                if isinstance(expected_range, tuple):
                    val_min, val_max = df[col].min(), df[col].max()
                    if expected_range[0] <= val_min and val_max <= expected_range[1]:
                        print(f"{col}: Range [{val_min:.2f}, {val_max:.2f}] within expected bounds")
                    else:
                        print(f"WARNING {col}: Range [{val_min:.2f}, {val_max:.2f}] outside expected {expected_range}")
                else:
                    unique_vals = df[col].unique()
                    if all(val in expected_range for val in unique_vals):
                        print(f"{col}: Values {unique_vals} as expected")
                    else:
                        print(f"WARNING {col}: Unexpected values {unique_vals}")
            else:
                print(f"ERROR {col}: Column missing")
                
    else:
        print("No CSV result files found. Run a BO experiment first:")
        print("python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/quick_test_bo_config.yaml")
    
    print("\nTest Complete")

if __name__ == "__main__":
    test_notebook_format_compatibility() 