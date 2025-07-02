#!/usr/bin/env python
"""
Bayesian Optimization with LVMOGP - exactly replicating the notebook pipeline.

Example:
  $ python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/original_bo_ssvi_config.yaml
"""
import argparse
from pathlib import Path
import torch
import pandas as pd
import dataclasses
import datetime
import json
import numpy as np

from src.bayesian_optimization.config_helper import load_full_config
from src.bayesian_optimization.bo_gp_lvm_ssvi import bayesian_optimization_loop
from src.bayesian_optimization.targets_helper import load_targets
from src.bayesian_optimization.results_converter import save_notebook_compatible_results


def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="YAML config with gp_ssvi + bo sections")
    args = parser.parse_args()

    # Load config
    cfg = load_full_config(args.config)
    gp_cfg = cfg.gp_ssvi
    bo_cfg = cfg.bo

    work_dir = Path.cwd()

    # Load full data exactly as in notebook
    data = pd.read_csv(work_dir / "data" / "data.csv", index_col=[0])

    # Use seed, test_name, start_point, pct_train from config
    seed = bo_cfg.seed
    pct_train = bo_cfg.pct_train
    test_name = bo_cfg.test_name
    start_point = bo_cfg.start_point

    print(f"Config: seed={seed}, test_name={test_name}, start_point={start_point}")

    # Load train/test split exactly as in notebook
    train_file = work_dir / f"data/bayes_opt/seed_{seed}_{test_name}_{start_point}/train.txt"
    test_file = work_dir / f"data/bayes_opt/seed_{seed}_{test_name}_{start_point}/test.txt"

    with train_file.open() as f:
        train_locs = [line.strip() for line in f]
    with test_file.open() as f:
        test_locs = [line.strip() for line in f]

    # Filter dataframes exactly as in notebook
    train_df = data[data['PrimerPairReporterBPGC'].isin(train_locs)]
    test_df = data[data['PrimerPairReporterBPGC'].isin(test_locs)]

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Load targets exactly as in notebook - including the EvaGreen/Probe logic
    targets = load_targets(work_dir)
    
    print(f"Available targets for {len(targets)} surfaces")
    print(f"Sample targets: {targets['PrimerPairReporter'].head().tolist()}")

    # Verify we have the required columns
    required_cols = ['BP', 'GC', 'PrimerPairReporter', 'Value']
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing required column: {col}")
        if col != 'Value' and col not in test_df.columns:
            raise ValueError(f"Missing required column in test_df: {col}")

    # Check that we have targets for the surfaces we're optimizing
    train_surfaces = set(train_df['PrimerPairReporter'].unique())
    test_surfaces = set(test_df['PrimerPairReporter'].unique())
    target_surfaces = set(targets['PrimerPairReporter'].unique())
    
    print(f"Train surfaces: {len(train_surfaces)}")
    print(f"Test surfaces: {len(test_surfaces)}")  
    print(f"Target surfaces: {len(target_surfaces)}")
    
    overlap = target_surfaces.intersection(test_surfaces)
    print(f"Surfaces with both test data and targets: {len(overlap)}")
    
    if len(overlap) == 0:
        raise ValueError("No surfaces have both test data and targets!")

    # Set device
    device = torch.device(gp_cfg.device)
    print(f"Using device: {device}")

    # Run Bayesian optimization loop using LVMOGP
    print(f"Starting BO with {bo_cfg.bo_steps} steps...")
    
    results = bayesian_optimization_loop(
        train_df=train_df,
        test_df=test_df, 
        targets=targets,
        config=gp_cfg,
        K_steps=bo_cfg.bo_steps,
        test_name=test_name,
        start_point=start_point,
        device=device
    )

    # Save results exactly as before but with new structure
    RESULTS_ROOT = work_dir / "gp_lvm_bo_run_results"
    config_name = args.config.stem
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    save_results_path = RESULTS_ROOT / f"bo_results_{config_name}_{timestamp}"
    save_results_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_results_path / f"config_used_{timestamp}.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    # Save final training data state
    final_data = results["final_train_data"]
    torch.save({
        "X_train": final_data["X_train"].cpu(),
        "Y_train": final_data["Y_train"].cpu(),
        "fn_train": final_data["fn_train"].cpu(),
        "H_mean": final_data["H_mean"].cpu(),
        "H_var": final_data["H_var"].cpu(),
        "ppr_to_idx": final_data["ppr_to_idx"],
        "targets": final_data["targets"]
    }, save_results_path / f"final_model_state_{timestamp}.pt")
    
    # Save BO metrics (convert numpy types to JSON-serializable types)
    bo_metrics = {
        "final_train_size": len(final_data["Y_train"]),
        "chosen_indices": convert_to_json_serializable(results["chosen_indices"]),
        "nlpd_values": convert_to_json_serializable(results["nlpd_values"]),
        "rmse_values": convert_to_json_serializable(results["rmse_values"]),
        "regret_values": convert_to_json_serializable(results["regret_values"]),
        "surfaces_optimized": convert_to_json_serializable(results["surfaces_optimized"]),
        "test_name": test_name,
        "start_point": start_point,
        "seed": seed
    }

    with open(save_results_path / f"bo_metrics_{timestamp}.json", "w") as f:
        json.dump(bo_metrics, f, indent=2)

    # Save EI values (convert tensors and numpy types to JSON-serializable lists)
    ei_values_list = []
    for ei_tensor in results["ei_values"]:
        if torch.is_tensor(ei_tensor):
            ei_values_list.append(ei_tensor.cpu().numpy().tolist())
        else:
            ei_values_list.append(convert_to_json_serializable(ei_tensor))
            
    with open(save_results_path / f"ei_values_{timestamp}.json", "w") as f:
        json.dump(ei_values_list, f, indent=2)

    # Save results in notebook-compatible CSV format for direct comparison
    print("\nSaving Notebook-Compatible Results")
    csv_path = save_notebook_compatible_results(
        bo_results=results,
        test_df=test_df,
        targets=targets,
        pred_mean_history=results["pred_mean_history"],
        pred_var_history=results["pred_var_history"],
        test_name=test_name,
        start_point=start_point,
        seed=seed,
        save_path=save_results_path,
        model_name="LVMOGP_SSVI"
    )

    print(f"\nBO COMPLETED")
    print(f"Final training set size: {bo_metrics['final_train_size']}")
    print(f"Surfaces optimized: {bo_metrics['surfaces_optimized']}")
    print(f"Final NLPD: {bo_metrics['nlpd_values'][-1]:.4f}")
    print(f"Final RMSE: {bo_metrics['rmse_values'][-1]:.4f}")
    print(f"Final Regret: {bo_metrics['regret_values'][-1]:.4f}")
    print(f"Results saved to: {save_results_path}")
    print(f"Notebook-compatible CSV: {csv_path}")
