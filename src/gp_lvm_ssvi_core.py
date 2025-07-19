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
from src.local_variable_optimisation import local_step, optimize_latents
from src.inducing_points import sample_U_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]



def train_gp_lvm_ssvi(config: GPSSVIConfig, Y: torch.Tensor, init_latents_z_dict: dict) -> dict:

    # --------------------------- misc -----------------------------------
    DEV = config.device_resolved()
    DEBUG = config.debug
    LR_X, LR_HYP, LR_ALPHA = config.lr.x, config.lr.hyp, config.lr.alpha
    BATCH, T_TOTAL = config.training.batch_size, config.training.total_iters
    INNER0 = config.training.inner_iters.start
    INNER = config.training.inner_iters.after
    JITTER, MAX_EXP = 5e-6, 60.0,
    CLIP_QUAD, GR_CLIP =  1e6, 20.0
    BOUNDS = {"log_sf2": (-8.0, 8.0), "log_alpha": (-20.0, 20.0),
              "log_beta": (-8.0, 5.0), "log_s2x": (-10.0, 10.0)}
    NUM_U_SAMPLES = config.num_u_samples_per_iter
    print(f"num_u_samples_per_iter: {NUM_U_SAMPLES}")

    rho = lambda t, t0=config.rho.t0, k=0.6: (t0 + t) ** (-config.rho.k)  # SVI step size
    safe_exp = lambda x: torch.exp(torch.clamp(x, min=-MAX_EXP, max=MAX_EXP))


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
    N, D = Y.shape  # N=846, D=12
    Q = config.q_latent

    # ----------------------- latent variables ---------------------------
    mu_x = init_latents_z_dict["mu_x"].to(device=DEV, dtype=torch.float64).detach().clone().requires_grad_()
    log_s2x = init_latents_z_dict["log_s2x"].to(device=DEV, dtype=torch.float64).detach().clone().requires_grad_()


    # ------------------- kernel & inducing inputs -----------------------
    M = config.inducing.n_inducing
    Z = init_latents_z_dict["Z"].to(device=DEV, dtype=torch.float64).detach().clone().requires_grad_()        

    snr = config.init_signal_to_noise_ratio
    print(f"snr: {snr}")
    sf2_init = float(Y.var().item())
    noise_var_init = sf2_init / snr
    
    log_sf2 = torch.tensor(math.log(sf2_init), device=DEV, requires_grad=True)
    log_alpha = (torch.full((Q,), -2.0, device=DEV)
                 + 0.1 * torch.randn(Q, device=DEV)
                 ).requires_grad_()  # (Q,)
    log_beta_inv = torch.tensor(math.log(noise_var_init), device=DEV, requires_grad=True)  # ()


    def k_se(x, z, log_sf2_val, log_alpha_val):
        sf2 = safe_exp(log_sf2_val)  # ()
        alpha = safe_exp(log_alpha_val)  # (Q,)
        diff = x.unsqueeze(-2) - z  # (., |x|, |z|, Q)
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





    # ---------- KL(q(U) || p(U)) ----------
    def compute_kl_u(m_u, C_u, K_MM, K_inv):
        """
        KL between the variational posterior q(U)=Prod_d N(m_d, Σ_d)
        and the GP prior p(U)=Prod_d N(0, K_MM).

        Returns
        -------
        kl : torch.Tensor scalar
            Differentiable w.r.t. kernel hyper-parameters (log_sf2, log_alpha, Z,…).
            q(U) parameters (m_u, C_u) are treated as constants here – they are
            updated by the natural-gradient block that follows.
        """
        # Sigma_d  –  full covariances of q(u_d)
        Sigma = C_u @ C_u.transpose(-1, -2)  # (D, M, M)

        # log|K_MM|
        L_K = torch.linalg.cholesky(K_MM)  # (M, M)
        logdet_K = 2.0 * torch.log(torch.diagonal(L_K)).sum()  # scalar

        # log|Sigma_d|
        diag_C = torch.diagonal(C_u, dim1=-2, dim2=-1).clamp_min(1e-12)  # (D, M)
        logdet_Sigma = 2.0 * torch.sum(torch.log(diag_C), dim=1)  # (D,)

        # tr(K^(-1) Sigma_d)
        trace_term = (K_inv.unsqueeze(0) * Sigma).sum(dim=(-2, -1))  # (D,)

        # m_d^T K^(-1) m_d
        quad_term = (m_u @ K_inv * m_u).sum(dim=1)  # (D,)

        kl_per_d = 0.5 * (trace_term + quad_term - M + logdet_K - logdet_Sigma)
        return kl_per_d.sum()  # scalar


    # --------------------------- optimizers ------------------------------
    opt_x = torch.optim.Adam([mu_x, log_s2x], lr=LR_X)
    opt_hyp = torch.optim.Adam([log_sf2, log_beta_inv, Z], lr=LR_HYP)

    opt_alpha = torch.optim.Adam(
        [log_alpha],
        lr=LR_ALPHA,
        betas=(0.5, 0.8),
        eps=1e-8
    )
    
    mu_x_prev   = mu_x.detach().clone()
    Z_prev      = Z.detach().clone()
    alpha_prev  = log_alpha.detach().clone()
    
    snapshots = {}
    iters, full_elbo_hist, local_elbo_hist, ll_hist, klx_hist, klu_hist = [], [], [], [], [], []
    for t in trange(1, T_TOTAL + 1, ncols=100):
        Sigma_det = Sigma_u(C_u).detach()  # (D, M, M)
        idx = torch.randint(0, N, (BATCH,), device=DEV)  # (B,)

        # ----- inner loop: update latent X ------------------------------
        inner_iters = INNER0 if t <= 50 else INNER
        optimize_latents(inner_iters=inner_iters, opt_x=opt_x, Y=Y, K_inv=K_inv, noise_var=noise_var, m_u=m_u, C_u=C_u, Sigma_det=Sigma_det, idx=idx, Z=Z, DEV=DEV, log_sf2=log_sf2, log_alpha=log_alpha, mu_x=mu_x, log_s2x=log_s2x, NUM_U_SAMPLES=NUM_U_SAMPLES, GR_CLIP=GR_CLIP, LR_X=LR_X)

        # ----- update kernel hyper-params and q(U) ----------------------
        opt_hyp.zero_grad(set_to_none=True)
        opt_alpha.zero_grad(set_to_none=True)
        K_MM, K_inv = update_K_and_inv()
        U_smpls = sample_U_batch(m_u, C_u, NUM_U_SAMPLES)
        elbo_vec, ll_vec, klx_vec, r_stack, Q_stack = vmap(
            lambda u: local_step(idx=idx, U_sample=u, Sigma_det=Sigma_u(C_u), update_beta=True, mu_x_batch=mu_x[idx], log_s2x_batch=log_s2x[idx], Y=Y, K_inv=K_inv, noise_var=noise_var, log_sf2=log_sf2, log_alpha=log_alpha, Z=Z, DEV=DEV)
        )(U_smpls)
        local_elbo_batch_mean = elbo_vec.mean()
        ll_b_mean   = ll_vec.mean()
        klx_b_mean  = klx_vec.mean()
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
                KLx_batch = (klx_b_mean * N).item()
                kl_u_val  = kl_u.item()
                full_elbo = LL_batch - KLx_batch - kl_u_val
                print(f"\nBATCH FULL ELBO @ {t:3d}: {full_elbo:.4e}  "
                    f"LL={LL_batch:.4e}  KL_X={KLx_batch:.4e}  KL_U={kl_u_val:.4e}")

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
                print(f"    grad_norm_x   ={grad_norm([mu_x, log_s2x]):.2e}, "
                    f"grad_norm_hyp ={grad_norm([log_sf2, Z, log_beta_inv]):.2e}, "
                    f"grad_norm_alpha={grad_norm([log_alpha]):.2e}")

                # 5) parameter update norms
                def step_norm(new, old):
                    return (new - old).norm().item()
                print(f"    step_norm_x   ={step_norm(mu_x, mu_x_prev):.2e}, "
                    f"step_norm_hyp ={step_norm(Z, Z_prev):.2e}, "
                    f"step_norm_alpha={step_norm(log_alpha, alpha_prev):.2e}")

                # 6) inducing point distance stats
                dZ = torch.pdist(Z)
                print(f"    Z_distances min={dZ.min():.2e}, max={dZ.max():.2e}")

                # update previous copies for next log
                mu_x_prev.copy_(mu_x)
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
                full_idx = torch.arange(N, device=DEV)
                elbo_vec, ll_vec, klx_vec, *_ = vmap(
                    lambda u: local_step(idx=full_idx, U_sample=u, Sigma_det=Sigma_u(C_u), update_beta=False, mu_x_batch=mu_x[full_idx], log_s2x_batch=log_s2x[full_idx], Y=Y, K_inv=K_inv, noise_var=noise_var, log_sf2=log_sf2, log_alpha=log_alpha, Z=Z, DEV=DEV)
                )(U_smpls_full) 
                local_elbo_full_n_mean = elbo_vec.mean()
                LL_full   = (ll_vec.mean()  * N).item()
                KLx_full  = (klx_vec.mean() * N).item()
                kl_u_full = compute_kl_u(m_u, C_u, K_MM, K_inv).item()
                full_elbo = LL_full - KLx_full - kl_u_full
            iters.append(t)
            full_elbo_hist.append(full_elbo)
            local_elbo_hist.append(local_elbo_full_n_mean.item())
            ll_hist.append(LL_full)
            klx_hist.append(KLx_full)
            klu_hist.append(kl_u_full)
            print(f"\nDATASET FULL ELBO @ {t:3d}: {full_elbo:.4e}  "
                  f"LL={LL_full:.4e}  KL_X={KLx_full:.4e}  KL_U={kl_u_full:.4e}")
            
        if t % 250 == 0:
            with torch.no_grad():
                A = k_se(mu_x, Z, log_sf2, log_alpha) @ K_inv  # (N, M)
                predictive_mean_snap = A @ m_u.T  # (N, D)
                predictive_variance_snap = torch.stack([(A @ Sigma_u(C_u)[d] * A).sum(-1) for d in range(D)], dim=1)  # (N, D)
                iters_snapshot = iters.copy()
                snapshot = {
                    "mu_x": mu_x.detach().cpu().clone(),
                    "log_s2x": log_s2x.detach().cpu().clone(),
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
                    "klx_vals":  klx_hist.copy(),
                    "klu_vals":  klu_hist.copy(),
                    "predictive_mean": predictive_mean_snap.detach().cpu().clone(),
                    "predictive_variance": predictive_variance_snap.detach().cpu().clone(),
                }
                snapshots[t] = snapshot

    results_dict = {
    "mu_x": mu_x.detach().cpu(),  # (N, Q)
    "log_s2x": log_s2x.detach().cpu(),  # (N, Q)
    "Z": Z.detach().cpu(),  # (M, Q)
    "log_sf2": log_sf2.item(),
    "log_alpha": log_alpha.detach().cpu(),  # (Q,)
    "log_beta_inv": log_beta_inv.item(),
    "m_u": m_u.detach().cpu(),  # (D, M)
    "C_u": C_u.detach().cpu(),  # (D, M, M)
    "elbo_iters": iters,
    "elbo_vals": full_elbo_hist,
    "local_elbo_vals": local_elbo_hist,
    "ll_vals":   ll_hist,
    "klx_vals":  klx_hist,
    "klu_vals":  klu_hist,
    "snapshots": snapshots
    }
    
    with torch.no_grad():
        A = k_se(mu_x, Z, log_sf2, log_alpha) @ K_inv  # (N, M)
        predictive_mean = A @ m_u.T  # (N, D)
        predictive_variance = torch.stack([(A @ Sigma_u(C_u)[d] * A).sum(-1) for d in range(D)], dim=1)  # (N, D)

    results_dict["predictive_mean"] = predictive_mean.cpu()
    results_dict["predictive_variance"] = predictive_variance.cpu()
    
    return results_dict