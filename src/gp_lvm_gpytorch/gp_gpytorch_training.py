import torch
import numpy as np
from tqdm import trange
from pathlib import Path
import json
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO

from src.gp_lvm_gpytorch.gp_gpytorch_model import bGPLVM
from src.gp_lvm_gpytorch.gp_gpytorch_dataclasses import BGPLVMConfig


def load_init_x(init_x_config, N, latent_dim):
    if init_x_config.method == "custom":
        path = Path(init_x_config.custom_path)
        with open(path, "r") as f:
            data = json.load(f)
        mu = np.asarray(data["init_mu_x"], dtype=np.float32)
        log_sigma2 = np.asarray(data["init_log_sigma2_x"], dtype=np.float32)
    else:
        mu = np.random.randn(N, latent_dim).astype(np.float32)
        log_sigma2 = np.log(np.ones((N, latent_dim), dtype=np.float32))
    return {"init_mu_x": mu, "init_log_sigma2_x": log_sigma2}


def train_bgplvm(cfg: BGPLVMConfig, Y: torch.Tensor, init_latents_and_z_dict: dict):
    torch.set_default_dtype(torch.float64)
    device = cfg.device_resolved()
    Y = Y.to(device)

    N, data_dim = Y.shape
    latent_dim = cfg.q_latent
    n_inducing = cfg.inducing.n_inducing

    model = bGPLVM(N, data_dim, latent_dim, n_inducing, init_latents_and_z_dict).to(device)
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape).to(device)
    mll = VariationalELBO(likelihood, model, num_data=N)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=cfg.optimizer.lr)

    iters = 4 if cfg.training.smoke_test else cfg.training.total_iters
    batch_size = cfg.training.batch_size
    loss_list, iters_list   = [], []
    snapshots = {}

    iterator = trange(iters, leave=True)
    for i in iterator:
        batch_index = model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample = model.sample_latent_variable()
        sample_batch = sample[batch_index]
        output_batch = model(sample_batch)
        loss = -mll(output_batch, Y[batch_index].T).sum()
        loss_list.append(loss.item())
        iterator.set_description(f"Loss: {loss.item():.2f}, Iter: {i}")
        loss.backward()
        optimizer.step()
        
        iters_list.append(i)
        if i % 2000 == 0 and i > 0:
            with torch.no_grad():
                q_u = model.q_u_dist
                log_s2x = 2 * model.q_x_std.log().detach().cpu()
                latent_mu = model.X.q_mu
                dist = model(latent_mu)
                snapshot = {
                    "mu_x": latent_mu.detach().cpu().clone(),
                    "log_alpha": -2 * model.covar_module.base_kernel.lengthscale.log().squeeze().detach().cpu().clone(),
                    "elbo_iters": iters_list.copy(),
                    "elbo_vals": [-v for v in loss_list],
                    "Z": model.inducing_inputs[0].detach().cpu().clone(),
                    "m_u": q_u.variational_mean.detach().cpu().clone(),
                    "C_u": q_u.chol_variational_covar.detach().cpu().clone(),
                    "log_sf2": model.covar_module.outputscale.log().item(),
                    "log_s2x": log_s2x,
                    "predictive_mean": dist.mean.T.detach().cpu().clone(),
                    "predictive_variance": dist.variance.T.detach().cpu().clone(),
                }
                snapshots[i] = snapshot
        
    with torch.no_grad():
        q_u = model.q_u_dist
        log_s2x = 2 * model.q_x_std.log().detach().cpu()
        latent_mu = model.X.q_mu
        dist = model(latent_mu)  # (batch_shape=D)
        
    results_dict = {
    "mu_x": model.X.q_mu.detach().cpu(),  # (N, Q)
    "log_alpha": -2 * model.covar_module.base_kernel.lengthscale.log().squeeze().detach().cpu(),  # (Q,)
    "elbo_iters": iters_list,
    "elbo_vals": [-v for v in loss_list],
    "Z": model.inducing_inputs[0].detach().cpu(),
    "m_u": q_u.variational_mean.detach().cpu(),
    "C_u": q_u.chol_variational_covar.detach().cpu().clone(),  
    "log_sf2": model.covar_module.outputscale.log().item(),  # log(signal variance)
    "log_s2x": log_s2x,
    "predictive_mean": dist.mean.T.cpu(),      # (N, D)
    "predictive_variance": dist.variance.T.cpu(),  # (N, D)
    "snapshots": snapshots,
    }

    return results_dict