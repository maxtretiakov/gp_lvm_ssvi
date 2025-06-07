from dataclasses import dataclass
from typing import Optional
import torch

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
class ModelConfig:
    latent_dim: Optional[int] = None  # if None - will be data_dim
    n_inducing: int = 25
    pca: bool = False  # using PCA

@dataclass
class BGPLVMConfig:
    device: str = "auto"  # "cuda", "cpu", or "auto"
    debug: bool = False
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    init_latent_dist: InitX = InitX()

    def device_resolved(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
