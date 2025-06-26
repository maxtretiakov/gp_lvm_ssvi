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


from src.lvmogp.gp_lvm_ssvi_core import train_gp_lvm_ssvi
from src.gp_dataclasses    import GPSSVIConfig

# global settings & tiny helpers
torch.set_default_dtype(torch.float64)
JITTER  = 5e-6
MAX_EXP = 60.0


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Clamp-then-exp to avoid overflow."""
    return torch.exp(torch.clamp(x, max=MAX_EXP))


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
        H_data_mean         : torch.Tensor,     # (N , Q   )
        H_data_var          : torch.Tensor,     # (N , Q   )
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
        self.fn_idx = X_data_fn.to(self.DEV)      # kept for completeness

        self.N , self.D = self.Y.shape
        self.D_x        = self.X_obs.shape[1]
        self.Q          = H_data_mean.shape[1]

        # latent H initial values
        self.H_mean    = H_data_mean.to(self.DEV).clone()          # (N,Q)
        self.H_log_s2  = torch.log(H_data_var.to(self.DEV))        # (N,Q)

        # inducing inputs in latent space
        if inducing_variable is None:
            if num_inducing_variables is None:
                raise ValueError("Specify `num_inducing_variables` or supply `inducing_variable`.")
            Z = self.H_mean[torch.randperm(self.N)[:num_inducing_variables]]
        else:
            Z = inducing_variable
        self.Z = Z.to(self.DEV).clone()   # (M,Q)
        self.M = self.Z.shape[0]

        # static mask: observed dims are fixed during SSVI
        mask             = torch.ones(self.D_x + self.Q, dtype=torch.float64, device=self.DEV)
        mask[:self.D_x]  = 0.0
        self.static_mask = mask

        # container for trained quantities
        self.results : Dict[str, torch.Tensor] | None = None

    # training
    def ssvi_train(self, config: GPSSVIConfig) -> Dict[str, torch.Tensor]:
        """Run the external optimiser and cache its result dict."""
        # build initial variational parameters
        mu_x0   = torch.cat([self.X_obs, self.H_mean], 1)
        log_s20 = torch.cat(
            [torch.full_like(self.X_obs, -10.0), self.H_log_s2], 1
        )
        init = {"mu_x": mu_x0, "log_s2x": log_s20, "Z": self.Z}

        # pass the "keep-observed-dims-fixed" mask
        config.static_mask = self.static_mask

        self.results = train_gp_lvm_ssvi(config, self.Y, init)
        return self.results

    # helpers
    def _unpack(self):
        """Convenience to pull tensors from the training dict."""
        if self.results is None:
            raise RuntimeError("Call .ssvi_train() before prediction.")

        r     = self.results
        mu_x  = r["mu_x"].to(self.DEV)                    # (N,D_x+Q)
        H_mu  = mu_x[:, self.D_x:]                        # (N,Q)
        return (
            H_mu,                                         # posterior mean of H
            r["log_s2x"].to(self.DEV)[:, self.D_x:],      # log-var of   H
            r["Z"].to(self.DEV),                          # inducing Z
            torch.tensor(r["log_sf2"], device=self.DEV),
            r["log_alpha"].to(self.DEV),
            torch.tensor(r["log_beta_inv"], device=self.DEV),
            r["m_u"].to(self.DEV),
            r["C_u"].to(self.DEV),
        )

    @staticmethod
    def _kernel(x, z, log_sf2, log_alpha):
        sf2   = _safe_exp(log_sf2)
        alpha = _safe_exp(log_alpha)                      # (Q,)
        diff  = x.unsqueeze(-2) - z                       # (...,1,Q)-(M,Q)
        return sf2 * _safe_exp(-0.5 * (diff**2 * alpha).sum(-1))

    @staticmethod
    def _psi(mu, s2, Z, log_sf2, log_alpha):
        """Closed-form psi-statistics for an ARD RBF with diagonal s2."""
        sf2   = _safe_exp(torch.clamp(log_sf2, -8.0,  8.0))
        alpha = _safe_exp(torch.clamp(log_alpha, -20.0, 20.0))

        psi0  = mu.new_full((mu.size(0),), sf2.item())         # (B,)

        d1    = alpha * s2 + 1.0
        c1    = d1.rsqrt().prod(-1, keepdim=True)
        diff  = mu.unsqueeze(1) - Z                           # (B,M,Q)
        psi1  = sf2 * c1 * _safe_exp(
                    -0.5 * ((alpha * diff**2) / d1.unsqueeze(1)).sum(-1))

        d2    = 1.0 + 2.0 * alpha * s2
        c2    = d2.rsqrt().prod(-1, keepdim=True)
        ZZ    = Z.unsqueeze(1) - Z.unsqueeze(0)               # (M,M,Q)
        dist  = (alpha * ZZ**2).sum(-1)
        mid   = 0.5 * (Z.unsqueeze(1) + Z.unsqueeze(0))
        mu_c  = (mu.unsqueeze(1).unsqueeze(1) - mid)**2
        expo  = -0.25 * dist - (alpha * mu_c / d2.unsqueeze(1).unsqueeze(1)).sum(-1)
        psi2  = sf2**2 * c2.unsqueeze(-1) * _safe_exp(expo)
        return psi0, psi1, psi2                              # (B,) (B,M) (B,M,M)


    @torch.no_grad()
    def predict_f(
        self,
        Xnew: Tuple[torch.Tensor, torch.Tensor],       # (mu_full , var_full)
        *,                                             # keyword-only flags
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return GPflow-style predictive mean and *diagonal* variance.

        Xnew  = (X_mean_full , X_var_full) where each tensor has shape
                (N*, D_x + Q).  Only the diagonal predictive covariance
                is implemented – exactly like `gpflow.models.GPR.predict_f()`.
        """
        # CAN BE FIXED IF NEEDED
        if full_cov or full_output_cov:
            raise NotImplementedError("Only diagonal predictive cov is implemented.")

        # 2. Unpack trained parameters
        (H_mu_tr, H_log_s2_tr,        # not used here, but returned for completeness
        Z, log_sf2, log_alpha,
        log_beta_inv, m_u, C_u) = self._unpack()

        # 3. Split observed & latent parts of Xnew
        Xnew_mean_full, Xnew_var_full = Xnew                    # (N*, D_x+Q)
        mu_H = Xnew_mean_full[:, self.D_x:]                    # (N*, Q)
        s2_H = Xnew_var_full[:,  self.D_x:]                    # (N*, Q)

        # 4. Psi-statistics for the new points
        psi0, psi1, psi2 = self._psi(mu_H, s2_H, Z, log_sf2, log_alpha)

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
        noise = _safe_exp(torch.tensor(self.results["log_beta_inv"], device=self.DEV))
        return mu_f, var_f + noise

    # optional – not implemented
    def predict_f_point(self, *args, **kwargs):
        raise NotImplementedError("Point predictions are not provided.")


