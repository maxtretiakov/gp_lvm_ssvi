"""
Pure-PyTorch implementation of the Latent-Variable Multi-Output GP
trained with Stochastic Structured Variational Inference (SSVI).

Public API intentionally matches the original GPflow class so that
existing code can switch models with one line:

    gp = LVMOGP_SSVI_Torch(...)
    gp.ssvi_train(config)          # <-- training
    mu, var = gp.predict_y((X_mean, X_var))

Only an ARD RBF kernel is supported – the same assumption the SSVI loop
makes internally.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import numpy as np


from src.lvmogp.gp_lvm_ssvi_core import train_lvmogp_ssvi
from src.gp_dataclasses    import GPSSVIConfig

# global settings & tiny helpers
torch.set_default_dtype(torch.float64)
JITTER  = 5e-6
MAX_EXP = 60.0


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Clamp-then-exp to avoid overflow and underflow."""
    return torch.exp(torch.clamp(x, min=-MAX_EXP, max=MAX_EXP))


def _chol_safe(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically safe Cholesky (or eigen-fallback)."""
    eye = torch.eye(mat.size(-1), device=mat.device, dtype=mat.dtype)
    try:
        return torch.linalg.cholesky(mat + eps * eye)
    except RuntimeError:                      # not PD – use eig fallback
        eig_val, eig_vec = torch.linalg.eigh(mat)
        eig_val = torch.clamp(eig_val, min=eps)
        return eig_vec @ torch.diag_embed(torch.sqrt(eig_val))

# main class
class LVMOGP_SSVI_Torch:
    """
    Latent-variable multi-output GP with SSVI training – PyTorch edition.

    Parameters mimic the GPflow constructor *exactly* so existing
    pipelines (Bayesian optimisation, plotting, …) remain untouched.
    """

    # --------------------------  constructor  ------------------------- #
    def __init__(
        self,
        data                : torch.Tensor,     # (N , D_out)
        X_data              : torch.Tensor,     # (N , D_x )
        X_data_fn           : torch.Tensor,     # (N , 1)  – function idx
        H_data_mean         : torch.Tensor,     # (num_fns , Q)
        H_data_var          : torch.Tensor,     # (num_fns , Q)
        kernel                     = None,      # kept for API parity
        num_inducing_variables: Optional[int] = None,
        inducing_variable         = None,
        H_prior_mean              = None,       # unused (same as GPflow)
        H_prior_var               = None,
        device : Optional[torch.device] = None,
    ):
        # device handling
        self.DEV = device or X_data.device

        # store raw data
        self.Y      = data.to(self.DEV)           # (N , D_out)
        self.X_obs  = X_data.to(self.DEV)         # (N , D_x )
        self.fn_idx = X_data_fn.to(self.DEV).long().squeeze()  # (N,) - function indices

        self.N , self.D = self.Y.shape
        self.D_x        = self.X_obs.shape[1]
        self.num_fns, self.Q = H_data_mean.shape

        # latent H initial values (per function)
        self.H_mean    = H_data_mean.to(self.DEV).clone()          # (num_fns, Q)
        self.H_var     = H_data_var.to(self.DEV).clone()           # (num_fns, Q)

        # inducing inputs in FULL latent space (D_x + Q)
        if inducing_variable is None:
            if num_inducing_variables is None:
                raise ValueError("Specify `num_inducing_variables` or supply `inducing_variable`.")
            # Create full input by concatenating X with H[fn_idx]
            H_per_point = self.H_mean[self.fn_idx]  # (N, Q)
            X_full = torch.cat([self.X_obs, H_per_point], dim=1)  # (N, D_x+Q)
            Z = X_full[torch.randperm(self.N)[:num_inducing_variables]]
        else:
            Z = inducing_variable
        self.Z = Z.to(self.DEV).clone()   # (M, D_x+Q)
        self.M = self.Z.shape[0]

        # container for trained quantities
        self.results : Dict[str, torch.Tensor] | None = None

    # training
    def ssvi_train(self, config: GPSSVIConfig) -> Dict[str, torch.Tensor]:
        """Run the external optimiser and cache its result dict."""
        # build initial variational parameters for LVMOGP SSVI
        init_H_dict = {
            "H_mean": self.H_mean,  # (num_fns, Q)
            "H_var": self.H_var,    # (num_fns, Q)
            "Z": self.Z             # (M, D_x+Q)
        }

        # Run LVMOGP SSVI training
        self.results = train_lvmogp_ssvi(config, self.Y, self.X_obs, self.fn_idx.unsqueeze(-1), init_H_dict)
        return self.results

    # helpers
    def _unpack(self):
        """Convenience to pull tensors from the training dict."""
        if self.results is None:
            raise RuntimeError("Call .ssvi_train() before prediction.")

        r = self.results
        return (
            r["H_mean"].to(self.DEV),                         # (num_fns, Q)
            r["H_log_s2"].to(self.DEV),                       # (num_fns, Q)
            r["Z"].to(self.DEV),                              # (M, D_x+Q)
            torch.tensor(r["log_sf2"], device=self.DEV),
            r["log_alpha"].to(self.DEV),                      # (D_x+Q,)
            torch.tensor(r["log_beta_inv"], device=self.DEV),
            r["m_u"].to(self.DEV),                            # (D, M)
            r["C_u"].to(self.DEV),                            # (D, M, M)
        )

    def _get_full_input(self, X_data, X_data_fn, H_mean_vals, H_log_s2_vals=None):
        """
        Create full input by concatenating X with H[fn_idx]
        X_data: (N*, D_x)
        X_data_fn: (N*,) - function indices  
        H_mean_vals: (num_fns, Q)
        Returns: X_full_mean (N*, D_x+Q), X_full_var (N*, D_x+Q) if H_log_s2_vals provided
        """
        H_per_point = H_mean_vals[X_data_fn]  # (N*, Q)
        X_full_mean = torch.cat([X_data, H_per_point], dim=1)  # (N*, D_x+Q)
        
        if H_log_s2_vals is not None:
            H_var_per_point = torch.exp(H_log_s2_vals[X_data_fn])  # (N*, Q)
            X_var_fixed = torch.full_like(X_data, 1e-8)  # X is observed, very small variance
            X_full_var = torch.cat([X_var_fixed, H_var_per_point], dim=1)  # (N*, D_x+Q)
            return X_full_mean, X_full_var
        else:
            return X_full_mean

    @staticmethod
    def _kernel(x, z, log_sf2, log_alpha):
        sf2   = _safe_exp(log_sf2)
        alpha = _safe_exp(log_alpha)                      # (D_x+Q,)
        diff  = x.unsqueeze(-2) - z                       # (...,1,D_x+Q)-(M,D_x+Q)
        return sf2 * _safe_exp(-0.5 * (diff**2 * alpha).sum(-1))

    @staticmethod
    def _psi(mu, s2, Z, log_sf2, log_alpha):
        """Closed-form psi-statistics for an ARD RBF with diagonal s2."""
        sf2   = _safe_exp(torch.clamp(log_sf2, -8.0,  8.0))
        alpha = _safe_exp(torch.clamp(log_alpha, -20.0, 20.0))

        psi0  = mu.new_full((mu.size(0),), sf2.item())         # (B,)

        d1    = alpha * s2 + 1.0
        # numerically-stable: log-prod → exp(sum log)
        log_c1 = -0.5 * torch.log(d1).sum(-1, keepdim=True)
        c1 = torch.exp(log_c1)                                   # (B,1)
        diff  = mu.unsqueeze(1) - Z                           # (B,M,D_x+Q)
        psi1  = sf2 * c1 * _safe_exp(
                    -0.5 * ((alpha * diff**2) / d1.unsqueeze(1)).sum(-1))

        d2    = 1.0 + 2.0 * alpha * s2
        log_c2 = -0.5 * torch.log(d2).sum(-1, keepdim=True)
        c2 = torch.exp(log_c2)                                   # (B,1)
        ZZ    = Z.unsqueeze(1) - Z.unsqueeze(0)               # (M,M,D_x+Q)
        dist  = (alpha * ZZ**2).sum(-1)
        mid   = 0.5 * (Z.unsqueeze(1) + Z.unsqueeze(0))
        mu_c  = (mu.unsqueeze(1).unsqueeze(1) - mid)**2
        expo  = -0.25 * dist - (alpha * mu_c / d2.unsqueeze(1).unsqueeze(1)).sum(-1)
        psi2  = sf2**2 * c2.unsqueeze(-1) * _safe_exp(expo)
        return psi0, psi1, psi2                              # (B,) (B,M) (B,M,M)


    @torch.no_grad()
    def predict_f(
        self,
        Xnew: Tuple[torch.Tensor, torch.Tensor],       # (X_new, X_new_fn) or (X_full_mean, X_full_var)
        *,                                             # keyword-only flags
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return GPflow-style predictive mean and *diagonal* variance.

        For LVMOGP, Xnew can be:
        1. (X_new, X_new_fn) where X_new is (N*, D_x) and X_new_fn is (N*,) function indices
        2. (X_full_mean, X_full_var) where both are (N*, D_x+Q) - full input representation
        """
        # CAN BE FIXED IF NEEDED
        if full_cov or full_output_cov:
            raise NotImplementedError("Only diagonal predictive cov is implemented.")

        # 2. Unpack trained parameters
        (H_mu_tr, H_log_s2_tr,
        Z, log_sf2, log_alpha,
        log_beta_inv, m_u, C_u) = self._unpack()

        # 3. Handle different input formats
        Xnew_mean, Xnew_var = Xnew
        if Xnew_mean.shape[1] == self.D_x:
            # Format 1: (X_new, X_new_fn)
            X_new = Xnew_mean                                      # (N*, D_x)
            X_new_fn = Xnew_var.long().squeeze()                   # (N*,) - function indices
            mu_full, s2_full = self._get_full_input(X_new, X_new_fn, H_mu_tr, H_log_s2_tr)
        else:
            # Format 2: (X_full_mean, X_full_var)
            mu_full, s2_full = Xnew_mean, Xnew_var               # (N*, D_x+Q)

        # 4. Psi-statistics for the new points
        psi0, psi1, psi2 = self._psi(mu_full, s2_full, Z, log_sf2, log_alpha)

        # 5. Posterior predictive moments
        Kuu      = self._kernel(Z, Z, log_sf2, log_alpha)
        Kinv     = torch.cholesky_inverse(_chol_safe(Kuu + JITTER *
                                                    torch.eye(self.M, device=self.DEV)))
        A        = psi1 @ Kinv                                  # (N*, M)
        mean_f   = A @ m_u.T                                    # (N*, D)

        # ---- diagonal predictive variance --------------------------------
        Sigma_u  = C_u @ C_u.transpose(-1, -2)                  # (D, M, M)
        var_f    = torch.stack(
                    [(A @ Sigma_u[d] * A).sum(-1) for d in range(self.D)], 1
                )                                            # (N*, D)
        var_f    = var_f + psi0.unsqueeze(1) - (psi2 * Kinv).sum((-2, -1)).unsqueeze(1)

        return mean_f, var_f

    def predict_y(
        self,
        Xnew: Tuple[torch.Tensor, torch.Tensor],
        *,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add learned homoscedastic Gaussian noise to `predict_f`."""
        mu_f, var_f = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        
        # Add noise variance
        if self.results is not None:
            noise = _safe_exp(torch.tensor(self.results["log_beta_inv"], device=self.DEV))
            return mu_f, var_f + noise
        else:
            raise RuntimeError("Call .ssvi_train() before prediction.")

    # optional – not implemented
    def predict_f_point(self, *args, **kwargs):
        raise NotImplementedError("Point predictions are not provided.")

    # Additional method for compatibility with GPflow interface
    def fill_Hs(self, X_data=None, X_data_fn=None):
        """
        Compatibility method that creates full input like the original GPflow implementation.
        Returns (X_full_mean, X_full_var) by concatenating X with H[fn_idx].
        """
        if self.results is None:
            # Use initial values
            H_mean_vals = self.H_mean
            H_var_vals = self.H_var
        else:
            # Use trained values
            H_mean_vals = self.results["H_mean"].to(self.DEV)
            H_var_vals = torch.exp(self.results["H_log_s2"].to(self.DEV))
        
        if X_data is None:
            X_data = self.X_obs
        if X_data_fn is None:
            X_data_fn = self.fn_idx
        
        return self._get_full_input(X_data, X_data_fn, H_mean_vals, torch.log(H_var_vals))


