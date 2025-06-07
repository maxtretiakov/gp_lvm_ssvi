from __future__ import annotations
from dataclasses import dataclass
import json, numpy as np, torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

try:
    import umap
except ImportError:
    umap = None


@dataclass
class InitGenConfig:
    method: str                     # pca | prior | random | isomap | umap
    q_latent: int
    var_init: float = -2.0
    random_scale: float = 0.1
    seed: int | None = None
    n_neighbors: int = 15
    min_dist: float = 0.1


# ───────────────── helpers ─────────────────
def _std(x): return (x - x.mean(0)) / x.std(0)

def _to_np(t): return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

def _mk(mu, log_s2, dev):
    mu_x = torch.as_tensor(mu, dtype=torch.float64, device=dev).requires_grad_()
    log_s2x = torch.full_like(mu_x, log_s2).requires_grad_()
    return mu_x, log_s2x


# ───────────────── concrete methods ─────────────────
def _pca(Y, q_latent, dev, c):    return _mk(PCA(q_latent).fit_transform(_std(_to_np(Y))), c.var_init, dev)

def _prior(Y, q_latent, dev, c):  return torch.zeros((Y.shape[0], q_latent), dtype=torch.float64,
                                            device=dev, requires_grad=True), \
                               torch.zeros((Y.shape[0], q_latent), dtype=torch.float64,
                                           device=dev, requires_grad=True)
                               
def _random(Y, q_latent, dev, c):
    if c.seed is not None:
        torch.manual_seed(c.seed); np.random.seed(c.seed)
    mu = c.random_scale * np.random.randn(Y.shape[0], q_latent)
    return _mk(mu, c.var_init, dev)

def _isomap(Y, q_latent, dev, c):
    mu = Isomap(n_neighbors=c.n_neighbors, n_components=q_latent)\
         .fit_transform(_std(_to_np(Y)))
    return _mk(mu, c.var_init, dev)

def _umap(Y, q_latent, dev, c):
    if umap is None:
        raise RuntimeError("pip install umap-learn")
    mu = umap.UMAP(n_components=q_latent, n_neighbors=c.n_neighbors,
                   min_dist=c.min_dist).fit_transform(_std(_to_np(Y)))
    return _mk(mu, c.var_init, dev)


_ROUTER = {"pca": _pca, "prior": _prior, "random": _random,
           "isomap": _isomap, "umap": _umap}


def build_latents(Y: torch.Tensor, q_latent: int, dev, cfg: InitGenConfig):
    fn = _ROUTER.get(cfg.method.lower())
    if fn is None:
        raise ValueError(f"unknown init.method '{cfg.method}'")
    mu_x, log_s2x = fn(Y, q_latent, dev, cfg)
    assert mu_x.shape[1] == q_latent, f"Expected q_latent={q_latent}, got {mu_x.shape[1]}"
    return mu_x, log_s2x


def save_json(mu_x, log_s2x, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"mu_x": mu_x.detach().cpu().tolist(),
                   "log_s2x": log_s2x.detach().cpu().tolist()}, f)
