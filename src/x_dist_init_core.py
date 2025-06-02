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


# ───────────────── dataclass для ГЕНЕРАТОРА ─────────────────
@dataclass
class InitGenConfig:
    method: str                     # pca | prior | random | isomap | umap
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
def _pca(Y, q, d, c):    return _mk(PCA(q).fit_transform(_std(_to_np(Y))), c.var_init, d)
def _prior(Y, q, d, c):  return torch.zeros((Y.shape[0], q), dtype=torch.float64,
                                            device=d, requires_grad=True), \
                               torch.zeros((Y.shape[0], q), dtype=torch.float64,
                                           device=d, requires_grad=True)
def _random(Y, q, d, c):
    if c.seed is not None:
        torch.manual_seed(c.seed); np.random.seed(c.seed)
    mu = c.random_scale * np.random.randn(Y.shape[0], q)
    return _mk(mu, c.var_init, d)
def _isomap(Y, q, d, c):
    mu = Isomap(n_neighbors=c.n_neighbors, n_components=q)\
         .fit_transform(_std(_to_np(Y)))
    return _mk(mu, c.var_init, d)
def _umap(Y, q, d, c):
    if umap is None:
        raise RuntimeError("pip install umap-learn")
    mu = umap.UMAP(n_components=q, n_neighbors=c.n_neighbors,
                   min_dist=c.min_dist).fit_transform(_std(_to_np(Y)))
    return _mk(mu, c.var_init, d)


_ROUTER = {"pca": _pca, "prior": _prior, "random": _random,
           "isomap": _isomap, "umap": _umap}


def build_latents(Y: torch.Tensor, q: int, dev, cfg: InitGenConfig):
    fn = _ROUTER.get(cfg.method.lower())
    if fn is None:
        raise ValueError(f"unknown init.method '{cfg.method}'")
    return fn(Y, q, dev, cfg)


def save_json(mu_x, log_s2x, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"mu": mu_x.detach().cpu().tolist(),
                   "s2": log_s2x.detach().cpu().tolist()}, f)
