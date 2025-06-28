#!/usr/bin/env python
"""
Run bGPLVM training from a YAML config.

oil dataset:
$ python run_gp_lvm_gpytorch.py --config gp_lvm_gpytorch_configs/original_gp_gpytorch_config.yaml

swiss roll dataset:
$ python run_gp_lvm_gpytorch.py --config gp_lvm_gpytorch_configs/swiss_roll_gp_gpytorch_config.yaml
"""
import argparse, yaml, dataclasses, torch
from typing import Any, get_type_hints
from pathlib import Path
import json
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

from src.gp_lvm_gpytorch.gp_gpytorch_dataclasses import BGPLVMConfig
from src.gp_lvm_gpytorch.gp_gpytorch_training import train_bgplvm
from src.data_loaders.oil_data_loader import load_Y
from src.data_loaders.swiss_roll_loader import load_swiss_roll_data
from src.oil_dataset_plot_core import load_oil_fractions, plot_oil_dataset_gp_lvm_results
from src.helpers import initialize_latents_and_z
from src.evaluate_gp_metrics import evaluate_gp_lvm_model_metrics, save_metrics_json
from src.save_results_utils import save_all_results


def _to_dataclass(cls, src: Any):
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    if not isinstance(src, dict):
        raise TypeError(f"Expected dict to populate {cls}, got {type(src)}")

    type_hints = get_type_hints(cls)
    kwargs = {}

    for fld in dataclasses.fields(cls):
        key = fld.name
        if key not in src:
            # Skip fields that have default values
            if fld.default is not dataclasses.MISSING or fld.default_factory is not dataclasses.MISSING:
                continue
            else:
                raise ValueError(f"Missing '{key}' in config")

        val = src[key]
        typ = type_hints[key]

        if dataclasses.is_dataclass(typ) and isinstance(val, dict):
            kwargs[key] = _to_dataclass(typ, val)
        else:
            try:
                # Handle basic type conversion
                if typ in (int, float, str, bool):
                    kwargs[key] = typ(val)
                else:
                    kwargs[key] = val
            except Exception:
                kwargs[key] = val

    return cls(**kwargs)


def load_config(path: Path | str) -> BGPLVMConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_dataclass(BGPLVMConfig, raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("CONFIG:",cfg)

    PROJECT_ROOT = Path(__file__).resolve().parent
    
    # Load data based on dataset type
    if cfg.dataset.type == "oil":
        oil_data_path = PROJECT_ROOT / "oil_data"
        Y, labels = load_Y(oil_data_path, cfg.device)
        fractions = load_oil_fractions(oil_data_path)
        dataset_type = "oil"
    elif cfg.dataset.type == "swiss_roll":
        Y, labels = load_swiss_roll_data(
            n_samples=cfg.dataset.n_samples,
            noise=cfg.dataset.noise,
            random_state=cfg.dataset.random_state if cfg.dataset.random_state is not None else 42,
            device=cfg.device
        )
        fractions = labels  # For Swiss Roll, use colors as "fractions"
        dataset_type = "swiss_roll"
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.type}")
    
    print(f"Loaded {dataset_type} dataset with shape: {Y.shape}")
    
    init_latents_and_z_dict = initialize_latents_and_z(Y, cfg)    
    train_results_dict = train_bgplvm(cfg, Y, init_latents_and_z_dict)
    
    metrics = evaluate_gp_lvm_model_metrics(train_results_dict, Y)
    
    RESULTS_ROOT = PROJECT_ROOT / "gp_lvm_gpytorch_run_results"
    config_name = args.config.stem
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    save_results_path = RESULTS_ROOT / f"results_{config_name}_{timestamp}"
    
    save_results_path.mkdir(parents=True, exist_ok=True)
    
    with open(save_results_path / f"config_used_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    save_all_results(
        train_results_dict,
        labels,
        fractions,
        Y, 
        metrics,
        config_name,
        save_results_path,
        dataset_type
    )