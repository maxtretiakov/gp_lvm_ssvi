import json
import torch
import numpy as np
from typing import Union
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.gp_dataclasses import GPSSVIConfig
from src.gp_lvm_gpy.gpy_dataclasses import BGPLVMConfig

def initialize_latents_and_z(Y: torch.Tensor, model_cfg: GPSSVIConfig | BGPLVMConfig) -> dict:
    DEV = model_cfg.device
    
    N = Y.size(0)
    D = Y.size(1)
    n_inducing = model_cfg.inducing.n_inducing
    Q = D
    
    Y_np = Y.detach().cpu().numpy()
    
    # latent space params init
    if model_cfg.init.method.lower() == "default":
        print("Default x dist params init used")
        Y_std = (Y_np - Y_np.mean(0)) / Y_np.std(0)
        mu_x = torch.tensor(PCA(Q).fit_transform(Y_std),
                            device=DEV)  # (N, Q)
        log_s2x = torch.full_like(mu_x, -2.0)  # (N, Q)
    else:
        print("Custom json x dist params init used")
        if not model_cfg.init.custom_path:
            raise ValueError("init.custom_path was not specified in the config")
        with open(model_cfg.init.custom_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            
        mu_x = torch.tensor(np.asarray(obj["mu_x"]),
                            device=DEV, dtype=torch.float64)
        log_s2x = torch.tensor(np.asarray(obj["log_s2x"]),
                               device=DEV, dtype=torch.float64)
        
    assert mu_x.shape[1] == Q
    
    # inducing points init
    if model_cfg.inducing.selection == "perm":
        if model_cfg.inducing.seed is not None:
            torch.manual_seed(model_cfg.inducing.seed)
        perm = torch.randperm(N, device=DEV)[:n_inducing]
        Z = mu_x.detach()[perm].clone()
    elif model_cfg.inducing.selection == "kmeans":
        Z_np = KMeans(n_clusters=n_inducing, random_state=model_cfg.inducing.seed)\
                .fit(mu_x.detach().cpu().numpy()).cluster_centers_
        Z = torch.tensor(Z_np, device=DEV, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown selection: {model_cfg.inducing.selection!r}")

    return {
        "mu_x": mu_x,
        "log_s2x": log_s2x,
        "Z": Z
    }
