import math, tarfile, urllib.request, numpy as np, matplotlib.pyplot as plt, torch
from pathlib import Path
from tqdm import trange
from sklearn.decomposition import PCA
import datetime
import json
from dataclasses import asdict
from torch.func import vmap 

torch.set_default_dtype(torch.float64)

from src.gp_dataclasses import GPSSVIConfig
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def train_lvmogp_ssvi(config: GPSSVIConfig, Y: torch.Tensor, X_data: torch.Tensor, 
                      X_data_fn: torch.Tensor, init_H_dict: dict) -> dict:
    """
    Train LVMOGP with SSVI.
    
    Args:
        config: SSVI configuration
        Y: Output data (N, D)
        X_data: Observed input data (N, D_x) - FIXED during training
        X_data_fn: Function indices (N, 1) - which function each point belongs to
        init_H_dict: Initial latent variables {"H_mean": (num_fns, Q), "H_var": (num_fns, Q), "Z": (M, D_x+Q)}
    """
    # --------------------------- misc -----------------------------------
    DEV = config.device_resolved()
    DEBUG = config.debug
    LR_H, LR_HYP, LR_ALPHA = config.lr.x, config.lr.hyp, config.lr.alpha  # reuse lr.x for H
    BATCH, T_TOTAL = config.training.batch_size, config.training.total_iters
    INNER0 = config.training.inner_iters.start
    INNER = config.training.inner_iters.after
    JITTER, MAX_EXP = 5e-6, 60.0,
    CLIP_QUAD, GR_CLIP =  1e6, 20.0
    BOUNDS = {"log_sf2": (-8.0, 8.0), "log_alpha": (-20.0, 20.0),
              "log_beta": (-8.0, 5.0), "log_s2H": (-10.0, 10.0)}
    NUM_U_SAMPLES = config.num_u_samples_per_iter
    print(f"num_u_samples_per_iter: {NUM_U_SAMPLES}")

    rho = lambda t, t0=config.rho.t0, k=0.6: (t0 + t) ** (-config.rho.k)  # SVI step size
    safe_exp = lambda x: torch.exp(torch.clamp(x, max=MAX_EXP))

    def cholesky_safe(mat, eps=1e-6):  # mat (., M, M)
        eye = torch.eye(mat.size(-1), device=mat.device, dtype=mat.dtype)
        try:
            return torch.linalg.cholesky(mat + eps * eye)  # (., M, M)
        except RuntimeError:
            eig_val, eig_vec = torch.linalg.eigh(mat)  # (., M), (., M, M)
            eig_val = torch.clamp(eig_val, min=eps)
            return eig_vec @ torch.diag_embed(torch.sqrt(eig_val))  # (., M, M)

    # --------------------------- data -----------------------------------
    Y = Y.to(DEV)
    X_data = X_data.to(DEV)  # (N, D_x) - FIXED
    X_data_fn = X_data_fn.to(DEV).long().squeeze()  # (N,) - function indices
    N, D = Y.shape  # N=846, D=12
    D_x = X_data.shape[1]
    Q = config.q_latent
    num_fns = int(X_data_fn.max().item()) + 1  # number of functions
    
    print(f"Data: N={N}, D={D}, D_x={D_x}, Q={Q}, num_fns={num_fns}")

    # ----------------------- latent variables H per function ---------------------------
    # H_mean: (num_fns, Q) - mean of latent variables for each function
    # H_log_s2: (num_fns, Q) - log variance of latent variables for each function
    H_mean = init_H_dict["H_mean"].to(device=DEV, dtype=torch.float64).detach().clone().requires_grad_()
    H_log_s2 = torch.log(init_H_dict["H_var"]).to(device=DEV, dtype=torch.float64).detach().clone().requires_grad_()
    
    def get_full_input(X_data, X_data_fn, H_mean_vals, H_log_s2_vals=None):
        """
        Create full input by concatenating X with H[fn_idx]
        X_data: (N, D_x)
        X_data_fn: (N,) - function indices  
        H_mean_vals: (num_fns, Q)
        Returns: X_full_mean (N, D_x+Q), X_full_var (N, D_x+Q) if H_log_s2_vals provided
        """
        H_per_point = H_mean_vals[X_data_fn]  # (N, Q)
        X_full_mean = torch.cat([X_data, H_per_point], dim=1)  # (N, D_x+Q)
        
        if H_log_s2_vals is not None:
            H_var_per_point = torch.exp(H_log_s2_vals[X_data_fn])  # (N, Q)
            X_var_fixed = torch.full_like(X_data, 1e-8)  # X is observed, very small variance
            X_full_var = torch.cat([X_var_fixed, H_var_per_point], dim=1)  # (N, D_x+Q)
            return X_full_mean, X_full_var
        else:
            return X_full_mean

    # ------------------- kernel & inducing inputs -----------------------
    M = config.inducing.n_inducing
    Z = init_H_dict["Z"].to(device=DEV, dtype=torch.float64).detach().clone().requires_grad_()  # (M, D_x+Q)
    
    snr = config.init_signal_to_noise_ratio
    print(f"snr: {snr}")
    sf2_init = float(Y.var().item())
    noise_var_init = sf2_init / snr
    
    log_sf2 = torch.tensor(math.log(sf2_init), device=DEV, requires_grad=True)
    log_alpha = (torch.full((D_x + Q,), -2.0, device=DEV)  # ARD for full input dimension
                 + 0.1 * torch.randn(D_x + Q, device=DEV)
                 ).requires_grad_()  # (D_x+Q,)
    log_beta_inv = torch.tensor(math.log(noise_var_init), device=DEV, requires_grad=True)  # ()

    def k_se(x, z, log_sf2_val, log_alpha_val):
        sf2 = safe_exp(log_sf2_val)  # ()
        alpha = safe_exp(log_alpha_val)  # (D_x+Q,)
        diff = x.unsqueeze(-2) - z  # (., |x|, |z|, D_x+Q)
        return sf2 * safe_exp(-0.5 * (diff ** 2 * alpha).sum(-1))  # (., |x|, |z|)

    noise_var = lambda: safe_exp(log_beta_inv)  # ()

    def update_K_and_inv():
        Kmat = k_se(Z, Z,
                    log_sf2.clamp(*BOUNDS["log_sf2"]),
                    log_alpha.clamp(*BOUNDS["log_alpha"])) \
               + JITTER * torch.eye(M, device=DEV)  # (M, M)
        L = cholesky_safe(Kmat)  # (M, M)
        return Kmat, torch.cholesky_inverse(L)  # (M, M), (M, M)

    K_MM, K_inv = update_K_and_inv()  # (M, M), (M, M)

    # ------------------------- q(U) block-diag --------------------------
    m_u = torch.zeros(D, M, device=DEV)  # (D, M)
    C_u = torch.eye(M, device=DEV).expand(D, M, M).clone()  # (D, M, M)

    def Sigma_u(C_u):
        return C_u @ C_u.transpose(-1, -2)  # (D, M, M)

    def sample_U_batch(m_u: torch.Tensor, C_u: torch.Tensor, S: int) -> torch.Tensor:
        eps   = torch.randn(S, *m_u.shape, device=m_u.device).unsqueeze(-1)   # (S,D,M,1)
        C_exp = C_u.unsqueeze(0).expand(S, -1, -1, -1)                        # (S,D,M,M)
        return m_u.unsqueeze(0) + (C_exp @ eps).squeeze(-1)                   # (S,D,M)

    def natural_from_moment(m_u, C_u):
        Sigma = C_u @ C_u.transpose(-1, -2)
        Lambda = -0.5 * torch.linalg.inv(Sigma)  # (D, M, M)
        h = (-2.0 * Lambda @ m_u.unsqueeze(-1)).squeeze(-1)  # (D, M)
        return h, Lambda

    def set_from_natural(h_new, Lambda_new, m_u, C_u, eps=1e-8):
        for d in range(D):
            Lam_d = 0.5 * (Lambda_new[d] + Lambda_new[d].transpose(0, 1))  # (M,M)
            eig_val, eig_vec = torch.linalg.eigh(Lam_d)  # (M,), (M, M)
            eig_val = torch.minimum(eig_val, torch.full_like(eig_val, -eps))
            S_d = torch.linalg.inv((eig_vec * (-2.0 * eig_val)) @ eig_vec.T)  # (M, M)
            C_u[d] = cholesky_safe(S_d, eps)  # (M, M)
            m_u[d] = S_d @ h_new[d]  # (M,)

    Lambda_prior = (-0.5 * K_inv).expand(D, M, M).clone()  # (D, M, M)

    # --------------------- psi statistics -------------------------------
    def compute_psi(mu, s2):
        """
        mu : (B, D_x+Q) - full input mean
        s2 : (B, D_x+Q) - full input variance  
        Returns:
            psi0 : (B,)
            psi1 : (B, M)
            psi2 : (B, M, M)
        """
        sf2 = safe_exp(log_sf2.clamp(*BOUNDS["log_sf2"]))  # ()
        alpha = safe_exp(log_alpha.clamp(*BOUNDS["log_alpha"]))  # (D_x+Q,)

        psi0 = torch.full((mu.size(0),), sf2.item(), device=DEV)  # (B,)

        d1 = alpha * s2 + 1.0  # (B, D_x+Q)
        # numerically-stable: log-prod → exp(sum log)
        log_c1 = -0.5 * torch.log(d1).sum(-1, keepdim=True)
        c1 = torch.exp(log_c1)                                   # (B,1)
        diff = mu.unsqueeze(1) - Z  # (B, M, D_x+Q)
        psi1 = sf2 * c1 * safe_exp(-0.5 * ((alpha * diff ** 2) / d1.unsqueeze(1)).sum(-1))  # (B, M)

        d2 = 1.0 + 2.0 * alpha * s2  # (B, D_x+Q)
        c2 = d2.rsqrt().prod(-1, keepdim=True)  # (B, 1)
        ZZ = Z.unsqueeze(1) - Z.unsqueeze(0)  # (M, M, D_x+Q)
        dist = (alpha * ZZ ** 2).sum(-1)  # (M, M)
        mid = 0.5 * (Z.unsqueeze(1) + Z.unsqueeze(0))  # (M, M, D_x+Q)
        mu_c = (mu.unsqueeze(1).unsqueeze(1) - mid) ** 2  # (B, M, M, D_x+Q)
        expo = -0.25 * dist - (alpha * mu_c / d2.unsqueeze(1).unsqueeze(1)).sum(-1)  # (B, M, M)
        psi2 = sf2 ** 2 * c2.unsqueeze(-1) * safe_exp(expo)  # (B,M,M)
        return psi0, psi1, psi2

    # ---------- KL(q(U) || p(U)) ----------
    def compute_kl_u(m_u, C_u, K_MM, K_inv):
        """KL between q(U) and p(U)"""
        Sigma = C_u @ C_u.transpose(-1, -2)  # (D, M, M)
        L_K = torch.linalg.cholesky(K_MM)  # (M, M)
        logdet_K = 2.0 * torch.log(torch.diagonal(L_K)).sum()  # scalar
        diag_C = torch.diagonal(C_u, dim1=-2, dim2=-1).clamp_min(1e-12)  # (D, M)
        logdet_Sigma = 2.0 * torch.sum(torch.log(diag_C), dim=1)  # (D,)
        trace_term = (K_inv.unsqueeze(0) * Sigma).sum(dim=(-2, -1))  # (D,)
        quad_term = (m_u @ K_inv * m_u).sum(dim=1)  # (D,)
        kl_per_d = 0.5 * (trace_term + quad_term - M + logdet_K - logdet_Sigma)
        return kl_per_d.sum()  # scalar

    # ---------- KL(q(H) || p(H)) - ONLY over H, not X ----------
    def compute_kl_H(H_mean, H_log_s2):
        """
        KL divergence between q(H) and p(H) = N(0, I)
        H_mean: (num_fns, Q)
        H_log_s2: (num_fns, Q) 
        """
        H_s2 = torch.exp(H_log_s2)  # (num_fns, Q)
        kl = 0.5 * ((H_s2 + H_mean ** 2) - H_log_s2 - 1.0)  # (num_fns, Q)
        return kl.sum()  # scalar

    # --------------------------- local step ------------------------------
    def local_step(idx, U_sample, Sigma_det, update_beta, dbg=False):
        """
        idx         : (B,)        mini-batch indices
        U_sample    : (D, M)      sample of inducing outputs
        Sigma_det   : (D, M, M)   detached covariance of q(U)
        update_beta : bool        whether noise is trainable here
        Returns     : scalar ELBO , r (B,D,M) , Q (B,D,M,M)
        """
        # Get full input for this batch
        mu_full, s2_full = get_full_input(X_data[idx], X_data_fn[idx], H_mean, H_log_s2)  # (B, D_x+Q)
        B = mu_full.size(0)

        psi0, psi1, psi2 = compute_psi(mu_full, s2_full)  # (B,), (B,M), (B,M,M)
        A = psi1 @ K_inv  # (B, M)

        if dbg and DEBUG:
            print("Shapes  A", A.shape, "psi1", psi1.shape, "psi2", psi2.shape)

        f_mean = A @ U_sample.T  # (B, D)
        var_f = torch.stack([(A @ Sigma_det[d] * A).sum(-1) for d in range(D)], 1)  # (B, D)

        noise = noise_var() if update_beta else noise_var().detach()  # ()
        tr_term = (psi2 * K_inv).sum((-2, -1))  # (B,)
        sigma2 = torch.clamp(noise + psi0 - tr_term, 1e-6, 1e3)  # (B,)
        sigma2_unsq = sigma2.unsqueeze(-1)  # (B,1)

        # ---------------- r ---------------------------------------------
        Y_div = Y[idx] / sigma2_unsq  # (B, D)
        r = Y_div.unsqueeze(-1) * A.unsqueeze(1)  # (B, D, M)

        # ---------------- Q ---------------------------------------------
        A_exp = A.unsqueeze(1)  # (B, 1, M)
        outer = A_exp.transpose(-1, -2) * A_exp  # (B, 1, M, M)
        outer = outer.unsqueeze(1)  # (B, 1, M, M)
        outer = outer.expand(B, D, M, M)  # (B, D, M, M)
        Q_mat = (-0.5 / sigma2_unsq).unsqueeze(-1).unsqueeze(-1) * outer  # (B, D, M, M)

        quad = ((Y[idx] - f_mean) ** 2 + var_f) / sigma2_unsq  # (B, D)
        quad = torch.clamp(quad, max=CLIP_QUAD)
        log_like = (-0.5 * math.log(2.0 * math.pi)
                    - 0.5 * sigma2.log().unsqueeze(-1)
                    - 0.5 * quad).sum(-1)  # (B,)
        
        # KL for H only (not for X which is observed)
        kl_H_batch = compute_kl_H(H_mean[X_data_fn[idx]], H_log_s2[X_data_fn[idx]]) / N  # scale by total N
        
        ll_mean  = log_like.mean()       
        klH_mean = kl_H_batch
        elbo_mean = ll_mean - klH_mean
        return elbo_mean, ll_mean, klH_mean, r.detach(), Q_mat.detach()

    # --------------------------- optimizers ------------------------------
    opt_H = torch.optim.Adam([H_mean, H_log_s2], lr=LR_H)
    opt_hyp = torch.optim.Adam([log_sf2, log_beta_inv, Z], lr=LR_HYP)
    opt_alpha = torch.optim.Adam([log_alpha], lr=LR_ALPHA, betas=(0.5, 0.8), eps=1e-8)
    
    H_mean_prev = H_mean.detach().clone()
    Z_prev = Z.detach().clone()
    alpha_prev = log_alpha.detach().clone()
    
    snapshots = {}
    iters, full_elbo_hist, local_elbo_hist, ll_hist, klH_hist, klu_hist = [], [], [], [], [], []
    
    for t in trange(1, T_TOTAL + 1, ncols=100):
        Sigma_det = Sigma_u(C_u).detach()  # (D, M, M)
        idx = torch.randint(0, N, (BATCH,), device=DEV)  # (B,)

        # ----- inner loop: update latent H ------------------------------
        inner_iters = INNER0 if t <= 50 else INNER
        for _ in range(inner_iters):
            opt_H.zero_grad(set_to_none=True)
            U_smpls = sample_U_batch(m_u, C_u, NUM_U_SAMPLES)
            elbo_vec = vmap(lambda u: local_step(idx, u, Sigma_det, False)[0])(U_smpls)
            local_elbo_batch_mean_H = elbo_vec.mean()
            loss_H = -local_elbo_batch_mean_H * N
            loss_H.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_([H_mean, H_log_s2], GR_CLIP)
            opt_H.step()
            with torch.no_grad():
                H_log_s2.clamp_(*BOUNDS["log_s2H"])

        # ----- update kernel hyper-params and q(U) ----------------------
        opt_hyp.zero_grad(set_to_none=True)
        opt_alpha.zero_grad(set_to_none=True)
        K_MM, K_inv = update_K_and_inv()
        U_smpls = sample_U_batch(m_u, C_u, NUM_U_SAMPLES)
        elbo_vec, ll_vec, klH_vec, r_stack, Q_stack = vmap(
            lambda u: local_step(idx, u, Sigma_u(C_u), True)
        )(U_smpls)
        local_elbo_batch_mean = elbo_vec.mean()
        ll_b_mean   = ll_vec.mean()
        klH_b_mean  = klH_vec.mean()
        r_b_mean    = r_stack.mean(0).sum(0)
        Q_b_mean    = Q_stack.mean(0).sum(0)
        diff_U_avg = (U_smpls - m_u).mean(0)  
        local_elbo = local_elbo_batch_mean * N
        kl_u = compute_kl_u(m_u, C_u, K_MM, K_inv)
        full_elbo = local_elbo - kl_u
        loss_hyp = -full_elbo
        loss_hyp.backward()

        if t % 50 == 0:
            with torch.no_grad():
                # 1) full ELBO and KL terms
                LL_batch  = (ll_b_mean  * N).item()
                KLH_batch = (klH_b_mean * N).item()
                kl_u_val  = kl_u.item()
                full_elbo = LL_batch - KLH_batch - kl_u_val
                print(f"\nBATCH FULL ELBO @ {t:3d}: {full_elbo:.4e}  "
                    f"LL={LL_batch:.4e}  KL_H={KLH_batch:.4e}  KL_U={kl_u_val:.4e}")

                # 2) noise vs signal
                sf2   = safe_exp(log_sf2)
                noise = noise_var()
                print(f"    sf2={sf2:.3e}, noise={noise:.3e}, ratio={noise/sf2:.3f}")

                # 3) hyperparameter values
                print(f"    log_sf2={log_sf2.item():.3f}, "
                    f"log_beta={log_beta_inv.item():.3f}, "
                    f"log_alpha min/max={log_alpha.min():.3f}/{log_alpha.max():.3f}")

                # 4) gradient norms for each block
                def grad_norm(params):
                    grads = [p.grad.flatten() for p in params if p.grad is not None]
                    return torch.cat(grads).norm().item() if grads else 0.0
                print(f"    grad_norm_H   ={grad_norm([H_mean, H_log_s2]):.2e}, "
                    f"grad_norm_hyp ={grad_norm([log_sf2, Z, log_beta_inv]):.2e}, "
                    f"grad_norm_alpha={grad_norm([log_alpha]):.2e}")

                # 5) parameter update norms
                def step_norm(new, old):
                    return (new - old).norm().item()
                print(f"    step_norm_H   ={step_norm(H_mean, H_mean_prev):.2e}, "
                    f"step_norm_hyp ={step_norm(Z, Z_prev):.2e}, "
                    f"step_norm_alpha={step_norm(log_alpha, alpha_prev):.2e}")

                # 6) inducing point distance stats
                dZ = torch.pdist(Z)
                print(f"    Z_distances min={dZ.min():.2e}, max={dZ.max():.2e}")

                # update previous copies for next log
                H_mean_prev.copy_(H_mean)
                Z_prev.copy_(Z)
                alpha_prev.copy_(log_alpha)

        opt_hyp.step()
        opt_alpha.step()

        # ----- natural-gradient step for q(U) --------------------------
        with torch.no_grad():
            for par, key in ((log_sf2, "log_sf2"),
                             (log_alpha, "log_alpha"),
                             (log_beta_inv, "log_beta")):
                par.clamp_(*BOUNDS[key])

            K_MM, K_inv = update_K_and_inv()  # (M, M), (M, M)
            Lambda_prior.copy_((-0.5 * K_inv).expand_as(Lambda_prior))  # (D, M, M)

            h_nat, Lambda_nat = natural_from_moment(m_u, C_u)  # (D,M), (D,M,M)
            r_tilde = r_b_mean + 2.0 * (Q_b_mean @ diff_U_avg.unsqueeze(-1)).squeeze(-1)

            lr = rho(t)
            scale = N / idx.size(0)
            h_new = (1.0 - lr) * h_nat + lr * scale * r_tilde  # (D, M)
            Lambda_new = ((1.0 - lr) * Lambda_nat
                          + lr * (Lambda_prior + scale * Q_b_mean))  # (D, M, M)
            set_from_natural(h_new, Lambda_new, m_u, C_u)

        # ----- monitoring ----------------------------------------------
        if t % 25 == 0 or t == 1:
            with torch.no_grad():
                U_smpls_full = sample_U_batch(m_u, C_u, NUM_U_SAMPLES)
                elbo_vec, ll_vec, klH_vec, *_ = vmap(
                    lambda u: local_step(torch.arange(N, device=DEV), u, Sigma_u(C_u), False)
                )(U_smpls_full) 
                local_elbo_full_n_mean = elbo_vec.mean()
                LL_full   = (ll_vec.mean()  * N).item()
                KLH_full  = (klH_vec.mean() * N).item()
                kl_u_full = compute_kl_u(m_u, C_u, K_MM, K_inv).item()
                full_elbo = LL_full - KLH_full - kl_u_full
            iters.append(t)
            full_elbo_hist.append(full_elbo)
            local_elbo_hist.append(local_elbo_full_n_mean.item())
            ll_hist.append(LL_full)
            klH_hist.append(KLH_full)
            klu_hist.append(kl_u_full)
            print(f"\nDATASET FULL ELBO @ {t:3d}: {full_elbo:.4e}  "
                  f"LL={LL_full:.4e}  KL_H={KLH_full:.4e}  KL_U={kl_u_full:.4e}")
            
        if t % 250 == 0:
            with torch.no_grad():
                # For prediction, get full input
                mu_full_pred = get_full_input(X_data, X_data_fn, H_mean)  # (N, D_x+Q)
                A = k_se(mu_full_pred, Z, log_sf2, log_alpha) @ K_inv  # (N, M)
                predictive_mean_snap = A @ m_u.T  # (N, D)
                predictive_variance_snap = torch.stack([(A @ Sigma_u(C_u)[d] * A).sum(-1) for d in range(D)], dim=1)  # (N, D)
                iters_snapshot = iters.copy()
                snapshot = {
                    "H_mean": H_mean.detach().cpu().clone(),
                    "H_log_s2": H_log_s2.detach().cpu().clone(),
                    "Z": Z.detach().cpu().clone(),
                    "log_sf2": log_sf2.item(),
                    "log_alpha": log_alpha.detach().cpu().clone(),
                    "log_beta_inv": log_beta_inv.item(),
                    "m_u": m_u.detach().cpu().clone(),
                    "C_u": C_u.detach().cpu().clone(),
                    "elbo_iters": iters_snapshot,
                    "elbo_vals": full_elbo_hist.copy(), 
                    "local_elbo_vals": local_elbo_hist.copy(),
                    "ll_vals":   ll_hist.copy(),
                    "klH_vals":  klH_hist.copy(),
                    "klu_vals":  klu_hist.copy(),
                    "predictive_mean": predictive_mean_snap.detach().cpu().clone(),
                    "predictive_variance": predictive_variance_snap.detach().cpu().clone(),
                }
                snapshots[t] = snapshot

    results_dict = {
        "H_mean": H_mean.detach().cpu(),  # (num_fns, Q)
        "H_log_s2": H_log_s2.detach().cpu(),  # (num_fns, Q)
        "Z": Z.detach().cpu(),  # (M, D_x+Q)
        "log_sf2": log_sf2.item(),
        "log_alpha": log_alpha.detach().cpu(),  # (D_x+Q,)
        "log_beta_inv": log_beta_inv.item(),
        "m_u": m_u.detach().cpu(),  # (D, M)
        "C_u": C_u.detach().cpu(),  # (D, M, M)
        "elbo_iters": iters,
        "elbo_vals": full_elbo_hist,
        "local_elbo_vals": local_elbo_hist,
        "ll_vals":   ll_hist,
        "klH_vals":  klH_hist,
        "klu_vals":  klu_hist,
        "snapshots": snapshots
    }
    
    with torch.no_grad():
        # Final prediction
        mu_full_final = get_full_input(X_data, X_data_fn, H_mean)  # (N, D_x+Q)
        A = k_se(mu_full_final, Z, log_sf2, log_alpha) @ K_inv  # (N, M)
        predictive_mean = A @ m_u.T  # (N, D)
        predictive_variance = torch.stack([(A @ Sigma_u(C_u)[d] * A).sum(-1) for d in range(D)], dim=1)  # (N, D)

    results_dict["predictive_mean"] = predictive_mean.cpu()
    results_dict["predictive_variance"] = predictive_variance.cpu()
    
    return results_dict