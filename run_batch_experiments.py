#!/usr/bin/env python
"""
Batch script to run multiple Bayesian Optimization experiments systematically.
This replicates the experimental setup from the notebook with multiple seeds,
test scenarios, and start points.

Usage:
    python run_batch_experiments.py --config_base bo_ssvi_configs/original_bo_ssvi_config.yaml
    python run_batch_experiments.py --quick_test  # For quick testing
"""

import argparse
import subprocess
import yaml
from pathlib import Path
import time
import datetime

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    """Save YAML config file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def run_single_experiment(config_path, experiment_name):
    """Run a single experiment and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            "python", "run_bo_gp_lvm_ssvi.py", 
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

def run_batch_experiments(base_config_path, quick_test=False):
    """Run a batch of experiments based on the base config."""
    
    base_config = load_config(base_config_path)
    results = []
    
    if quick_test:
        # Quick test with minimal experiments
        seeds = [0]
        test_scenarios = ["many_r", "one_from_many_FP004-RP004-Probe_r"]
        start_points = ["centre"]
        print("Running QUICK TEST experiments...")
    else:
        # Full experimental setup (reduced from notebook's full scope for manageability)
        seeds = base_config.get('experimental', {}).get('seeds', [0, 1, 2])
        test_scenarios = base_config.get('experimental', {}).get('test_scenarios', [
            "many_r", 
            "one_from_many_FP004-RP004-Probe_r",
            "one_from_many_FP001-RP001x-EvaGreen_r"
        ])[:3]  # Limit to first 3 for manageable runtime
        start_points = base_config.get('experimental', {}).get('start_points', ["centre"])
        print("Running FULL experiments...")
    
    print(f"Seeds: {seeds}")
    print(f"Test scenarios: {test_scenarios}")
    print(f"Start points: {start_points}")
    
    total_experiments = len(seeds) * len(test_scenarios) * len(start_points)
    print(f"Total experiments to run: {total_experiments}")
    
    experiment_count = 0
    start_time = time.time()
    
    for seed in seeds:
        for test_scenario in test_scenarios:
            for start_point in start_points:
                experiment_count += 1
                experiment_name = f"seed_{seed}_{test_scenario}_{start_point}"
                
                print(f"\n[{experiment_count}/{total_experiments}] Preparing: {experiment_name}")
                
                # Create experiment-specific config
                exp_config = base_config.copy()
                exp_config['bo']['seed'] = seed
                exp_config['bo']['test_name'] = test_scenario
                exp_config['bo']['start_point'] = start_point
                
                if quick_test:
                    # Reduce parameters for quick testing
                    exp_config['gp_ssvi']['training']['total_iters'] = 20
                    exp_config['bo']['bo_steps'] = 2
                    exp_config['gp_ssvi']['q_latent'] = 4
                    exp_config['gp_ssvi']['inducing']['n_inducing'] = 16
                
                # Save temporary config
                temp_config_path = Path(f"temp_config_{experiment_name}.yaml")
                save_config(exp_config, temp_config_path)
                
                # Run experiment
                success = run_single_experiment(temp_config_path, experiment_name)
                results.append({
                    'experiment': experiment_name,
                    'seed': seed,
                    'test_scenario': test_scenario,
                    'start_point': start_point,
                    'success': success
                })
                
                # Clean up temporary config
                temp_config_path.unlink()
                
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / experiment_count
                remaining = (total_experiments - experiment_count) * avg_time
                print(f"Progress: {experiment_count}/{total_experiments} "
                      f"(Elapsed: {elapsed/60:.1f}min, ETA: {remaining/60:.1f}min)")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"BATCH EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    if failed > 0:
        print(f"\nFailed experiments:")
        for r in results:
            if not r['success']:
                print(f"  FAILED: {r['experiment']}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch BO experiments")
    parser.add_argument("--config_base", type=Path, 
                       default="bo_ssvi_configs/original_bo_ssvi_config.yaml",
                       help="Base config file to use")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run quick test with minimal experiments")
    
    args = parser.parse_args()
    
    if not args.config_base.exists():
        print(f"ERROR: Config file not found: {args.config_base}")
        exit(1)
    
    print(f"Base config: {args.config_base}")
    print(f"Quick test mode: {args.quick_test}")
    
    results = run_batch_experiments(args.config_base, args.quick_test)
    
    # Save results summary in proper results folder structure
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    
    # Create results folder structure like other scripts
    work_dir = Path.cwd()
    RESULTS_ROOT = work_dir / "gp_lvm_batch_run_results"
    config_name = args.config_base.stem
    save_results_path = RESULTS_ROOT / f"batch_results_{config_name}_{timestamp}"
    save_results_path.mkdir(parents=True, exist_ok=True)
    
    # Save batch summary
    batch_summary = {
        'timestamp': timestamp,
        'base_config': str(args.config_base),
        'quick_test': args.quick_test,
        'total_experiments': len(results),
        'successful_experiments': sum(1 for r in results if r['success']),
        'failed_experiments': sum(1 for r in results if not r['success']),
        'success_rate': sum(1 for r in results if r['success']) / len(results) * 100 if results else 0,
        'results': results
    }
    
    results_file = save_results_path / f"batch_summary_{timestamp}.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(batch_summary, f, default_flow_style=False, indent=2)
    
    print(f"\nBatch results summary saved to: {results_file}") 