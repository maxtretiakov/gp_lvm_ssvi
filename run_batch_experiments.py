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

def run_single_experiment(config_path, experiment_name, log_dir, exp_config):
    """Run a single experiment with real-time output streaming and log saving."""
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Create log file for this experiment
    log_file = log_dir / f"{experiment_name}_log.txt"
    
    try:
        # Use Popen for real-time streaming with logging
        with open(log_file, 'w', encoding='utf-8') as log_f:
            # Write comprehensive header to log file
            log_f.write(f"Experiment: {experiment_name}\n")
            log_f.write(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write("="*60 + "\n")
            log_f.write("EXPERIMENT CONFIGURATION:\n")
            log_f.write("="*60 + "\n")
            
            # Save the complete configuration used for this experiment
            yaml.dump(exp_config, log_f, default_flow_style=False, indent=2)
            
            log_f.write("\n" + "="*60 + "\n")
            log_f.write("EXPERIMENT OUTPUT:\n")
            log_f.write("="*60 + "\n\n")
            log_f.flush()
            
            process = subprocess.Popen([
                "python", "run_bo_gp_lvm_ssvi.py", 
                "--config", str(config_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               text=True, bufsize=1, universal_newlines=True)
            
            # Stream output in real-time and save to log
            if process.stdout is not None:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output, end='')  # Print to console (real-time)
                        log_f.write(output)    # Write to log file
                        log_f.flush()          # Ensure immediate writing
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Write completion info to log
            log_f.write(f"\n" + "="*60 + "\n")
            log_f.write(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"Return code: {return_code}\n")
        
        if return_code == 0:
            print(f"\nSUCCESS: {experiment_name}")
            print(f"Log saved: {log_file}")
            return True
        else:
            print(f"\nFAILED: {experiment_name} (return code: {return_code})")
            print(f"Log with errors saved: {log_file}")
            return False
            
    except Exception as e:
        print(f"\nEXCEPTION: {experiment_name} - {e}")
        # Still save exception info to log
        try:
            with open(log_file, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\nEXCEPTION: {e}\n")
        except:
            pass
        return False

def run_batch_experiments(base_config_path, quick_test=False):
    """Run a batch of experiments based on the base config."""
    
    base_config = load_config(base_config_path)
    
    # Create logs directory for this batch
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    config_name = Path(base_config_path).stem
    log_dir = Path(f"batch_logs_{config_name}_{timestamp}")
    log_dir.mkdir(exist_ok=True)
    print(f"Experiment logs will be saved to: {log_dir}")
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
                success = run_single_experiment(temp_config_path, experiment_name, log_dir, exp_config)
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