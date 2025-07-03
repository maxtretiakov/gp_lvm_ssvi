#!/usr/bin/env python
"""
Parameter tuning script for LVMOGP-SSVI.
Systematically tests different parameter combinations on representative experiments.

Usage:
    python run_parameter_tuning.py --type bo    # Tune BO parameters
    python run_parameter_tuning.py --type cv    # Tune CV parameters
    python run_parameter_tuning.py --quick_test # Quick parameter test
"""

import argparse
import subprocess
import yaml
from pathlib import Path
import time
import datetime
import itertools

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    """Save YAML config file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def run_single_parameter_test(config_path, experiment_name, exp_type='bo'):
    """Run a single parameter test and return success status."""
    print(f"\n{'='*60}")
    print(f"Testing: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        if exp_type == 'bo':
            # Use batch experiments for BO (multiple seeds/scenarios automatically)
            result = subprocess.run([
                "python", "run_batch_experiments.py", 
                "--config_base", str(config_path)
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        else:  # cv
            # Use CV script directly
            result = subprocess.run([
                "python", "run_cv_gp_lvm_ssvi.py", 
                "--config", str(config_path)
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"SUCCESS: {experiment_name}")
            return True
        else:
            print(f"FAILED: {experiment_name}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {experiment_name} (exceeded 1 hour)")
        return False
    except Exception as e:
        print(f"EXCEPTION: {experiment_name} - {e}")
        return False

def run_bo_parameter_tuning(quick_test=False):
    """Run BO parameter tuning."""
    base_config_path = "bo_ssvi_configs/parameter_tuning_bo_config.yaml"
    base_config = load_config(base_config_path)
    
    if quick_test:
        # Minimal parameter grid for quick testing
        param_grid = {
            'q_latent': [8, 12],
            'n_inducing': [32, 64],
            'total_iters': [400, 800]
        }
        # Reduce experimental scope for quick test
        base_config['experimental']['seeds'] = [0]
        base_config['experimental']['test_scenarios'] = ['many_r']
        base_config['experimental']['start_points'] = ['centre']
        base_config['bo']['bo_steps'] = 5
        print("Running QUICK BO parameter test...")
    else:
        # Full parameter grid
        param_grid = {
            'q_latent': [8, 12, 16],
            'n_inducing': [32, 64, 128],
            'total_iters': [500, 800, 1200]
        }
        print("Running FULL BO parameter tuning...")
    
    results = []
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    print(f"Testing {len(param_combinations)} parameter combinations")
    print(f"Parameters: {param_names}")
    
    start_time = time.time()
    
    for i, param_values in enumerate(param_combinations):
        param_dict = dict(zip(param_names, param_values))
        experiment_name = f"bo_params_{'_'.join([f'{k}{v}' for k, v in param_dict.items()])}"
        
        print(f"\n[{i+1}/{len(param_combinations)}] Testing: {param_dict}")
        
        # Create parameter-specific config
        test_config = base_config.copy()
        test_config['gp_ssvi']['q_latent'] = param_dict['q_latent']
        test_config['gp_ssvi']['inducing']['n_inducing'] = param_dict['n_inducing']
        test_config['gp_ssvi']['training']['total_iters'] = param_dict['total_iters']
        
        # Save temporary config
        temp_config_path = Path(f"temp_param_config_{experiment_name}.yaml")
        save_config(test_config, temp_config_path)
        
        # Run experiment
        success = run_single_parameter_test(temp_config_path, experiment_name, 'bo')
        results.append({
            'experiment': experiment_name,
            'parameters': param_dict,
            'success': success
        })
        
        # Clean up temporary config
        temp_config_path.unlink()
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = (len(param_combinations) - i - 1) * avg_time
        print(f"Progress: {i+1}/{len(param_combinations)} "
              f"(Elapsed: {elapsed/60:.1f}min, ETA: {remaining/60:.1f}min)")
    
    return results

def run_cv_parameter_tuning(quick_test=False):
    """Run CV parameter tuning."""
    base_config_path = "cv_ssvi_configs/parameter_tuning_cv_config.yaml"
    base_config = load_config(base_config_path)
    
    if quick_test:
        # Minimal parameter grid for quick testing
        param_grid = {
            'q_latent': [8, 12],
            'n_inducing': [32, 64],
            'total_iters': [400, 800]
        }
        # Reduce experimental scope for quick test
        base_config['cv']['seeds'] = [0, 1]
        base_config['cv']['train_percentages'] = [30, 70]
        print("Running QUICK CV parameter test...")
    else:
        # Full parameter grid
        param_grid = {
            'q_latent': [8, 12, 16],
            'n_inducing': [32, 64, 128],
            'total_iters': [500, 800, 1200]
        }
        print("Running FULL CV parameter tuning...")
    
    results = []
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    print(f"Testing {len(param_combinations)} parameter combinations")
    print(f"Parameters: {param_names}")
    
    start_time = time.time()
    
    for i, param_values in enumerate(param_combinations):
        param_dict = dict(zip(param_names, param_values))
        experiment_name = f"cv_params_{'_'.join([f'{k}{v}' for k, v in param_dict.items()])}"
        
        print(f"\n[{i+1}/{len(param_combinations)}] Testing: {param_dict}")
        
        # Create parameter-specific config
        test_config = base_config.copy()
        test_config['gp_ssvi']['q_latent'] = param_dict['q_latent']
        test_config['gp_ssvi']['inducing']['n_inducing'] = param_dict['n_inducing']
        test_config['gp_ssvi']['training']['total_iters'] = param_dict['total_iters']
        
        # Save temporary config
        temp_config_path = Path(f"temp_param_config_{experiment_name}.yaml")
        save_config(test_config, temp_config_path)
        
        # Run experiment
        success = run_single_parameter_test(temp_config_path, experiment_name, 'cv')
        results.append({
            'experiment': experiment_name,
            'parameters': param_dict,
            'success': success
        })
        
        # Clean up temporary config
        temp_config_path.unlink()
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = (len(param_combinations) - i - 1) * avg_time
        print(f"Progress: {i+1}/{len(param_combinations)} "
              f"(Elapsed: {elapsed/60:.1f}min, ETA: {remaining/60:.1f}min)")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter tuning for LVMOGP-SSVI")
    parser.add_argument("--type", choices=['bo', 'cv'], required=True,
                       help="Type of experiments to tune")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run quick test with minimal parameters")
    
    args = parser.parse_args()
    
    print(f"Parameter tuning type: {args.type.upper()}")
    print(f"Quick test mode: {args.quick_test}")
    
    start_time = time.time()
    
    if args.type == 'bo':
        results = run_bo_parameter_tuning(args.quick_test)
    else:
        results = run_cv_parameter_tuning(args.quick_test)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"PARAMETER TUNING COMPLETED")
    print(f"{'='*60}")
    print(f"Total parameter combinations tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    if successful > 0:
        print(f"\nSuccessful parameter combinations:")
        for r in results:
            if r['success']:
                print(f"  SUCCESS: {r['parameters']}")
    
    if failed > 0:
        print(f"\nFailed parameter combinations:")
        for r in results:
            if not r['success']:
                print(f"  FAILED: {r['parameters']}")
    
    # Save results summary
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    work_dir = Path.cwd()
    RESULTS_ROOT = work_dir / "parameter_tuning_results"
    save_results_path = RESULTS_ROOT / f"tuning_{args.type}_{timestamp}"
    save_results_path.mkdir(parents=True, exist_ok=True)
    
    # Save parameter tuning summary
    tuning_summary = {
        'timestamp': timestamp,
        'experiment_type': args.type,
        'quick_test': args.quick_test,
        'total_combinations': len(results),
        'successful_combinations': successful,
        'failed_combinations': failed,
        'success_rate': successful / len(results) * 100 if results else 0,
        'results': results
    }
    
    results_file = save_results_path / f"parameter_tuning_summary_{timestamp}.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(tuning_summary, f, default_flow_style=False, indent=2)
    
    print(f"\nParameter tuning summary saved to: {results_file}")
    print(f"\nTo compare results with baselines, check the experiment output directories")
    print(f"and use the analysis code from notebooks/useful_notebook.ipynb") 