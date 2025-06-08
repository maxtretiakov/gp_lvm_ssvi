import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import root_mean_squared_error

def save_metrics_json(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {path}")


def compute_nlpd(y_true: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> float:
    """
    Compute average Negative Log Predictive Density (NLPD) for Gaussian outputs.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth values (N, D)
    mean : torch.Tensor
        Predictive mean (N, D)
    variance : torch.Tensor
        Predictive variance (N, D), should include noise variance

    Returns
    -------
    float
        Average NLPD across all points and dimensions
    """
    eps = 1e-6
    var_clamped = variance.clamp_min(eps)
    term1 = 0.5 * torch.log(2 * torch.pi * var_clamped)
    term2 = 0.5 * ((y_true - mean) ** 2) / var_clamped
    nlpd_matrix = term1 + term2  # (N, D)
    return nlpd_matrix.mean().item()


def evaluate_gp_lvm_model_metrics(
        results_dict: dict,
        Y: torch.Tensor,
    ) -> dict:
    """
    Evaluate GP-LVM model by computing RMSE and NLPD on all data (no train/test split).

    Parameters
    ----------
    results_dict : dict
        Dictionary containing predictive mean and variance.
    Y : torch.Tensor
        Full dataset (N, D)

    Returns
    -------
    dict
        Dictionary with RMSE and NLPD on full data
    """
    Y = Y.cpu()
    pred_mean = results_dict["predictive_mean"]
    pred_var = results_dict["predictive_variance"]

    rmse = root_mean_squared_error(Y.numpy(), pred_mean.numpy())
    nlpd = compute_nlpd(Y, pred_mean, pred_var)

    return {
        "rmse": rmse,
        "nlpd": nlpd
    }
