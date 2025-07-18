import torch 
import math
from torch import vmap
from src.inducing_points import sample_U_batch
from pdb import set_trace as st

#todo: move all of these values either to one file or 
MAX_EXP = 60
CLIP_QUAD = 1e6

BOUNDS = {"log_sf2": (-8.0, 8.0), "log_alpha": (-20.0, 20.0),
              "log_beta": (-8.0, 5.0), "log_s2x": (-10.0, 10.0)}


safe_exp = lambda x: torch.exp(torch.clamp(x, min=-MAX_EXP, max=MAX_EXP)) #todo: pull out files into src/utils.py

def optimize_latents(inner_iters, opt_x, Y, K_inv, noise_var, m_u, C_u, Sigma_det, idx, Z, DEV,log_sf2, log_alpha, BATCH_SIZE, NUM_LATENT_DIMS, NUM_U_SAMPLES, GR_CLIP):
    N = Y.shape[0]
    mu_x_batch = torch.zeros((BATCH_SIZE, NUM_LATENT_DIMS), device=DEV, requires_grad=True)  # (B, Q)
    log_s2x_batch = torch.zeros((BATCH_SIZE, NUM_LATENT_DIMS), device=DEV, requires_grad=True)  # (B, Q)
    for _ in range(inner_iters):
        opt_x.zero_grad(set_to_none=True)
        U_smpls = sample_U_batch(m_u, C_u, NUM_U_SAMPLES)
        elbo_vec = vmap(lambda u: local_step(idx=idx, U_sample=u, Sigma_det=Sigma_det, update_beta=False, mu_x_batch=mu_x_batch, log_s2x_batch=log_s2x_batch, Y=Y, K_inv=K_inv, noise_var=noise_var, log_sf2=log_sf2, log_alpha=log_alpha, Z=Z, DEV=DEV)[0])(U_smpls)
        local_elbo_batch_mean_x = elbo_vec.mean()
        loss_x = -local_elbo_batch_mean_x * N
        st()
        loss_x.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_([mu_x_batch, log_s2x_batch], GR_CLIP)
        opt_x.step()
        with torch.no_grad():
            log_s2x_batch.clamp_(*BOUNDS["log_s2x"])



    # --------------------- psi statistics -------------------------------
def compute_psi(mu, s2, log_sf2, log_alpha, Z, DEV):
    """
    mu : (B, Q)
    s2 : (B, Q)
    Returns:
        psi0 : (B,)
        psi1 : (B, M)
        psi2 : (B, M, M)
    """
    sf2 = safe_exp(log_sf2.clamp(*BOUNDS["log_sf2"]))  # ()
    alpha = safe_exp(log_alpha.clamp(*BOUNDS["log_alpha"]))  # (Q,)

    psi0 = torch.full((mu.size(0),), sf2.item(), device=DEV)  # (B,)

    d1 = alpha * s2 + 1.0  # (B, Q)
    # numerically-stable: log-prod to exp(sum log)
    log_c1 = -0.5 * torch.log(d1).sum(-1, keepdim=True)
    c1 = torch.exp(log_c1)                                   # (B,1)
    diff = mu.unsqueeze(1) - Z  # (B, M, Q)
    psi1 = sf2 * c1 * safe_exp(-0.5 * ((alpha * diff ** 2) / d1.unsqueeze(1)).sum(-1))  # (B, M)

    d2 = 1.0 + 2.0 * alpha * s2  # (B, Q)
    log_c2 = -0.5 * torch.log(d2).sum(-1, keepdim=True)
    c2 = torch.exp(log_c2)                                   # (B,1)
    ZZ = Z.unsqueeze(1) - Z.unsqueeze(0)  # (M, M, Q)
    dist = (alpha * ZZ ** 2).sum(-1)  # (M, M)
    mid = 0.5 * (Z.unsqueeze(1) + Z.unsqueeze(0))  # (M, M, Q)
    mu_c = (mu.unsqueeze(1).unsqueeze(1) - mid) ** 2  # (B, M, M, Q)
    expo = -0.25 * dist - (alpha * mu_c / d2.unsqueeze(1).unsqueeze(1)).sum(-1)  # (B, M, M)
    psi2 = sf2 ** 2 * c2.unsqueeze(-1) * safe_exp(expo)  # (B,M,M)

    return psi0, psi1, psi2

#TODO: find out what is happening to the gradients of mu_x and log_s2x
# --------------------------- local step ------------------------------
def local_step(idx, U_sample, Sigma_det, update_beta, 
               mu_x_batch, log_s2x_batch, Y, K_inv, noise_var, log_sf2, log_alpha, Z, DEV,
               dbg=False):
    """
    idx         : (B,)        mini-batch indices
    U_sample    : (D, M)      sample of inducing outputs
    Sigma_det   : (D, M, M)   detached covariance of q(U)
    update_beta : bool        whether noise is trainable here
    Returns     : scalar ELBO , r (B,D,M) , Q (B,D,M,M)
    """
    D, M = U_sample.size()  # (D, M)
    B = idx.size(0)  # (B,)


    # mu = mu_x[idx]  # (B, Q)
    s2 = log_s2x_batch.exp()  # (B, Q)

    psi0, psi1, psi2 = compute_psi(mu=mu_x_batch, s2=s2, log_sf2=log_sf2, log_alpha=log_alpha, Z=Z, DEV=DEV)  # (B,), (B,M), (B,M,M)
    A = psi1 @ K_inv  # (B, M)

    if dbg:
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
    kl_x = 0.5 * ((s2 + mu_x_batch ** 2) - s2.log() - 1.0).sum(-1)  # (B,)
    ll_mean  = log_like.mean()       
    klx_mean = kl_x.mean()           
    elbo_mean = ll_mean - klx_mean
    return elbo_mean, ll_mean, klx_mean, r.detach(), Q_mat.detach()
