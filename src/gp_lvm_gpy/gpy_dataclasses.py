from dataclasses import dataclass, field
from typing import Optional, Literal
import torch

@dataclass
class InducingConfig:
    n_inducing: int                         # Number of inducing points (M)
    selection: Literal["perm", "kmeans"]    # Strategy for selecting Z
    seed: Optional[int] = None              # Random seed (for reproducibility)
    
@dataclass
class InitX:
    """
    Used for X dist params init
        - method: "default" | "custom"
        - custom_path: path to JSON, if method == "custom"
    """
    method: str = "default"  # "default" | "custom"
    custom_path: Optional[str] = None 

@dataclass
class OptimizerConfig:
    lr: float = 0.01  # learning rate

@dataclass
class TrainingConfig:
    batch_size: int = 100
    total_iters: int = 10000
    smoke_test: bool = False 

@dataclass
class BGPLVMConfig:
    inducing: InducingConfig
    q_latent: int
    device: str = "auto"  # "cuda", "cpu", or "auto"
    debug: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    init_latent_dist: InitX = field(default_factory=InitX)

    def device_resolved(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Config auto selected device:", self.device)
        return torch.device(self.device)
