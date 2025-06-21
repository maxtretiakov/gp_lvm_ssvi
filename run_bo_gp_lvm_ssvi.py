#!/usr/bin/env python
"""
Run GP-LVM SSVI with Bayesian Optimization loop from a YAML config.

Example:
$ python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/original_bo_ssvi_config.yaml --seed 0 --pct_train 50

This script runs a BO loop using your GP-LVM SSVI model and the specified train/test split.
"""
import argparse
from pathlib import Path
import torch
import pandas as pd
import dataclasses
import datetime
import json
from sklearn.decomposition import PCA

from src.config import load_full_config
from src.helpers import initialize_latents_and_z
from bo_loop import bayesian_optimization_loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML config with gp_ssvi + bo sections")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pct_train", type=int, default=50)
    args = parser.parse_args()

    cfg = load_full_config(args.config)
    gp_cfg = cfg.gp_ssvi
    bo_cfg = cfg.bo

    work_dir = Path.cwd()

    # Load full data
    data = pd.read_csv(work_dir / "data" / "data.csv", index_col=[0])

    # Load train/test splits
    train_file = work_dir / f"data/cross_validation/seed_{args.seed}_pct_train_{args.pct_train}/train.txt"
    with train_file.open("r") as f:
        train_locs = [line.strip() for line in f]

    test_file = work_dir / f"data/cross_validation/seed_{args.seed}_pct_train_{args.pct_train}/test.txt"
    with test_file.open("r") as f:
        test_locs = [line.strip() for line in f]

    train_df = data[data['PrimerPairReporterBPGC'].isin(train_locs)]
    test_df = data[data['PrimerPairReporterBPGC'].isin(test_locs)]

    # Target: Value column
    Y = torch.tensor(train_df['Value'].values, dtype=torch.float64, device=gp_cfg.device).unsqueeze(-1)

    # === PCA for latents ===
    Q = gp_cfg.q_latent
    all_values = pd.concat([train_df, test_df])['Value'].values.reshape(-1, 1)
    pca = PCA(n_components=Q)
    all_latents_np = pca.fit_transform(all_values)

    train_latents = torch.tensor(all_latents_np[:len(train_df)], dtype=torch.float64, device=gp_cfg.device)
    test_latents = torch.tensor(all_latents_np[len(train_df):], dtype=torch.float64, device=gp_cfg.device)

    # Init latents and Z on train
    init_latents_and_z_dict = initialize_latents_and_z(Y, gp_cfg)
    init_latents_and_z_dict["mu_x"] = train_latents  # override with PCA

    # Acquisition grid: real test latents
    acquisition_grid = test_latents

    # Oracle: pick true Value from test_df matching the latent
    test_Y = torch.tensor(test_df['Value'].values, dtype=torch.float64, device=gp_cfg.device).unsqueeze(-1)

    def oil_oracle(x_latent):
        diffs = acquisition_grid - x_latent
        dists = torch.norm(diffs, dim=1)
        idx = torch.argmin(dists)
        return test_Y[idx]

    # Run BO loop
    Y_final, mu_x_final, log_s2x_final = bayesian_optimization_loop(
        Y,
        init_latents_and_z_dict,
        gp_cfg,
        K_steps=bo_cfg.bo_steps,
        acquisition_grid=acquisition_grid,
        reinit_Z=False,
        oracle_fn=oil_oracle
    )

    # Save
    RESULTS_ROOT = work_dir / "gp_lvm_bo_run_results"
    config_name = args.config.stem
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    save_results_path = RESULTS_ROOT / f"bo_results_{config_name}_{timestamp}"
    save_results_path.mkdir(parents=True, exist_ok=True)

    with open(save_results_path / f"config_used_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    torch.save({
        "Y_final": Y_final.cpu(),
        "mu_x_final": mu_x_final.cpu(),
        "log_s2x_final": log_s2x_final.cpu()
    }, save_results_path / f"bo_loop_output_{timestamp}.pt")

    print(f"BO loop finished. Results saved to {save_results_path}")
