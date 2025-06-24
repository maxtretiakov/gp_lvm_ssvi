#!/usr/bin/env python
"""
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.bayesian_optimization.config_helper import load_full_config
from src.bayesian_optimization.bo_gp_lvm_ssvi import bayesian_optimization_loop
from src.bayesian_optimization.targets_helper import load_targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="YAML config with gp_ssvi + bo sections")
    args = parser.parse_args()

    # Load config
    cfg = load_full_config(args.config)
    gp_cfg = cfg.gp_ssvi
    bo_cfg = cfg.bo

    work_dir = Path.cwd()

    # Load full data
    data = pd.read_csv(work_dir / "data" / "data.csv", index_col=[0])

    # Use seed, test_name, start_point, pct_train from config
    seed = bo_cfg.seed
    pct_train = bo_cfg.pct_train
    test_name = bo_cfg.test_name
    start_point = bo_cfg.start_point

    print(f"Config: seed={seed}, test_name={test_name}, start_point={start_point}")

    # Load train/test split
    train_file = work_dir / f"data/bayes_opt/seed_{seed}_{test_name}_{start_point}/train.txt"
    test_file = work_dir / f"data/bayes_opt/seed_{seed}_{test_name}_{start_point}/test.txt"

    with train_file.open() as f:
        train_locs = [line.strip() for line in f]
    with test_file.open() as f:
        test_locs = [line.strip() for line in f]

    train_df = data[data['PrimerPairReporterBPGC'].isin(train_locs)]
    test_df = data[data['PrimerPairReporterBPGC'].isin(test_locs)]

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Load targets
    targets = load_targets(work_dir)

    # Target for training
    Y = torch.tensor(train_df['Value'].values, dtype=torch.float64, device=gp_cfg.device).unsqueeze(-1)

    # PCA for latent space
    Q = gp_cfg.q_latent
    N_train = len(train_df)
    N_test = len(test_df)

    train_latents = torch.randn(N_train, Q, dtype=torch.float64, device=gp_cfg.device) * 0.1
    test_latents  = torch.randn(N_test,  Q, dtype=torch.float64, device=gp_cfg.device)
    
    # Init Z using KMeans
    kmeans = KMeans(n_clusters=gp_cfg.inducing.n_inducing, random_state=gp_cfg.inducing.seed)
    Z = torch.tensor(
        kmeans.fit(train_latents.cpu().numpy()).cluster_centers_,
        dtype=torch.float64,
        device=gp_cfg.device
    )

    # Init dict for BO loop
    init_latents_and_z_dict = {
        "mu_x": train_latents,
        "log_s2x": torch.zeros_like(train_latents),
        "Z": Z
    }

    # Acquisition grid ===
    acquisition_grid = test_latents

    # Use only the first surface for demonstration
    # Here we choose one surface for which we perform BO
    ppr = targets['PrimerPairReporter'].iloc[0]
    print(f"Optimizing surface: {ppr}")

    # Oracle returns true value for closest latent
    def oracle_fn(x_latent):
        diffs = acquisition_grid - x_latent
        dists = torch.norm(diffs, dim=1)
        idx = torch.argmin(dists)
        y_val = test_df['Value'].iloc[idx.item()]
        return torch.tensor([y_val], device=gp_cfg.device, dtype=torch.float64)

    # Run BO loop
    results = bayesian_optimization_loop(
        Y=Y,
        init_latents_z_dict=init_latents_and_z_dict,
        config=gp_cfg,
        K_steps=bo_cfg.bo_steps,
        acquisition_grid=acquisition_grid,
        reinit_Z=True,
        oracle_fn=oracle_fn,
        test_df=test_df,
        targets=targets,
        ppr=ppr
    )

    # Save results
    RESULTS_ROOT = work_dir / "gp_lvm_bo_run_results"
    config_name = args.config.stem
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    save_results_path = RESULTS_ROOT / f"bo_results_{config_name}_{timestamp}"
    save_results_path.mkdir(parents=True, exist_ok=True)

    with open(save_results_path / f"config_used_{timestamp}.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    torch.save({
        "Y_final": results["Y_final"].cpu(),
        "mu_x_final": results["mu_x_final"].cpu(),
        "log_s2x_final": results["log_s2x_final"].cpu(),
        "chosen_indices": results["chosen_indices"],
        "ei_values": [ei.numpy().tolist() for ei in results["ei_values"]],
        "nlpd_values": results["nlpd_values"],
        "rmse_values": results["rmse_values"],
        "regret_values": results["regret_values"]
    }, save_results_path / f"bo_loop_output_{timestamp}.pt")

    print(f"BO loop finished. Results saved to {save_results_path}")
