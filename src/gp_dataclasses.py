from dataclasses import dataclass
import torch
from typing import Optional, Literal


@dataclass
class InducingConfig:
    n_inducing: int                         # Number of inducing points (M)
    selection: Literal["perm", "kmeans"]    # Strategy for selecting Z
    seed: Optional[int] = None              # Random seed (for reproducibility)
    
@dataclass
class InitXDistSsvi:
    """
    Used in train_gp_lvm_ssvi:
        - method: "default" | "custom"
        - custom_path: path to JSON, if method == "custom"
    """
    method: str = "default"
    custom_path: Optional[str] = None

@dataclass
class LR:
    x: float
    hyp: float


@dataclass
class InnerIters:
    start: int
    after: int
    switch: int


@dataclass
class Training:
    batch_size: int
    total_iters: int
    inner_iters: InnerIters


@dataclass
class Rho:
    t0: float
    k: float


@dataclass
class GPSSVIConfig:
    device: str
    debug: bool
    lr: LR
    training: Training
    jitter: float
    max_exp: float
    rho: Rho
    inducing: InducingConfig
    q_latent: int
    init: InitXDistSsvi

    def device_resolved(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)