from dataclasses import dataclass, field
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
    alpha: float


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
    jitter: float
    max_exp: float
    q_latent: int
    lr: LR = field(default_factory=LR)
    rho: Rho = field(default_factory=Rho)
    training: Training = field(default_factory=Training)
    inducing: InducingConfig = field(default_factory=InducingConfig)
    init_latent_dist: InitXDistSsvi = field(default_factory=InitXDistSsvi)

    def device_resolved(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Config auto selected device:", self.device)
        return torch.device(self.device)