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
    train_idx: np.ndarray,
    test_idx: np.ndarray
) -> dict:
    """
    Evaluate GP-LVM model by computing RMSE and NLPD on train/test split.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing predictive mean and variance.
    Y : torch.Tensor
        Full dataset (N, D)
    train_idx : np.ndarray
        Indices for training data
    test_idx : np.ndarray
        Indices for testing data

    Returns
    -------
    dict
        Dictionary with train/test RMSE and NLPD
    """
    Y = Y.cpu()
    pred_mean = results_dict["predictive_mean"]
    pred_var = results_dict["predictive_variance"]

    rmse_train = root_mean_squared_error(Y[train_idx].numpy(), pred_mean[train_idx].numpy())
    rmse_test = root_mean_squared_error(Y[test_idx].numpy(), pred_mean[test_idx].numpy())

    nlpd_train = compute_nlpd(Y[train_idx], pred_mean[train_idx], pred_var[train_idx])
    nlpd_test = compute_nlpd(Y[test_idx], pred_mean[test_idx], pred_var[test_idx])

    return {
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "nlpd_train": nlpd_train,
        "nlpd_test": nlpd_test
    }
