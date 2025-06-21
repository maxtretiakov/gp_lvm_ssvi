import torch
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
from src.ssvigplvm import train_gp_lvm_ssvi


def k_se(x, z, log_sf2, log_alpha):
    sf2 = torch.exp(log_sf2)
    alpha = torch.exp(log_alpha)
    diff = x.unsqueeze(-2) - z
    return sf2 * torch.exp(-0.5 * (diff ** 2 * alpha).sum(-1))


def expected_improvement(mean, var, y_best):
    std = np.sqrt(np.maximum(var.cpu().numpy(), 1e-9))
    mean_np = mean.cpu().numpy()
    y_best_np = y_best.cpu().numpy()
    Z = (mean_np - y_best_np) / std
    ei_np = (mean_np - y_best_np) * norm.cdf(Z) + std * norm.pdf(Z)
    ei = torch.from_numpy(ei_np).to(mean.device)
    return ei


def add_new_data_point(Y, y_new, mu_x, log_s2x, init_method="prior"):
    Q = mu_x.shape[1]
    if init_method == "prior":
        mu_new = torch.zeros(Q, device=Y.device)
        log_s2x_new = torch.zeros(Q, device=Y.device)
    elif init_method == "random":
        mu_new = torch.randn(Q, device=Y.device) * 0.1
        log_s2x_new = torch.zeros(Q, device=Y.device)
    else:
        raise ValueError(f"Unknown init_method: {init_method}")
    Y = torch.cat([Y, y_new.view(1, -1)], dim=0)
    mu_x = torch.cat([mu_x, mu_new.view(1, -1)], dim=0)
    log_s2x = torch.cat([log_s2x, log_s2x_new.view(1, -1)], dim=0)
    mu_x.requires_grad = True
    log_s2x.requires_grad = True
    return Y, mu_x, log_s2x


def reinitialize_Z(mu_x, config):
    n_inducing = config.inducing.n_inducing
    Z_np = KMeans(n_clusters=n_inducing, random_state=config.inducing.seed)\
        .fit(mu_x.detach().cpu().numpy()).cluster_centers_
    Z = torch.tensor(Z_np, device=mu_x.device, dtype=torch.float64)
    return Z


def bayesian_optimization_loop(Y, init_latents_z_dict, config,
                               K_steps=5, acquisition_grid=None,
                               reinit_Z=False, oracle_fn=None):
    mu_x = init_latents_z_dict["mu_x"]
    log_s2x = init_latents_z_dict["log_s2x"]
    Z = init_latents_z_dict["Z"]
    DEV = config.device_resolved()

    chosen_indices = []
    ei_values = []

    for k in range(K_steps):
        print(f"\nBO step {k+1}/{K_steps}")

        init_dict = {"mu_x": mu_x, "log_s2x": log_s2x, "Z": Z}
        results = train_gp_lvm_ssvi(config, Y, init_dict)

        mu_x = results["mu_x"].to(DEV)
        log_s2x = results["log_s2x"].to(DEV)
        Z = results["Z"].to(DEV)
        m_u = results["m_u"].to(DEV)
        C_u = results["C_u"].to(DEV)
        log_sf2 = torch.tensor(results["log_sf2"], device=DEV)
        log_alpha = results["log_alpha"].to(DEV)

        K_MM = k_se(Z, Z, log_sf2, log_alpha) + 1e-6 * torch.eye(Z.shape[0], device=DEV)
        K_inv = torch.linalg.inv(K_MM)
        A = k_se(acquisition_grid, Z, log_sf2, log_alpha) @ K_inv

        pred_mean = A @ m_u.T
        pred_var = torch.stack([
            (A @ (C_u[d] @ C_u[d].T) * A).sum(-1)
            for d in range(m_u.shape[0])
        ], dim=1)

        pred_mean_scalar = pred_mean.mean(dim=1)
        pred_var_scalar = pred_var.mean(dim=1)

        y_best = Y.max()
        ei = expected_improvement(pred_mean_scalar, pred_var_scalar, y_best)

        # Mask already chosen
        if chosen_indices:
            mask = torch.zeros_like(ei)
            mask[chosen_indices] = 1
            ei = ei.masked_fill(mask.bool(), float("-inf"))

        idx_best = torch.argmax(ei)
        x_next_latent = acquisition_grid[idx_best]

        print(f"Next latent idx: {idx_best}  value: {x_next_latent}")
        print(f"Max EI: {ei[idx_best].item():.4f}")

        if oracle_fn is None:
            raise ValueError("You must pass oracle_fn for real data BO loop.")
        else:
            y_next = oracle_fn(x_next_latent)

        print(f"Oracle output: {y_next}")

        Y, mu_x, log_s2x = add_new_data_point(Y, y_next, mu_x, log_s2x, init_method="prior")

        if reinit_Z:
            Z = reinitialize_Z(mu_x, config)

        print(f"Updated dataset size: N = {Y.size(0)}")

        chosen_indices.append(idx_best.item())
        ei_values.append(ei.detach().cpu())

    return {
        "Y_final": Y,
        "mu_x_final": mu_x,
        "log_s2x_final": log_s2x,
        "chosen_indices": chosen_indices,
        "ei_values": ei_values
    }
