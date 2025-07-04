#!/usr/bin/env python
"""
Run GP-LVM SSVI from a YAML config.

$ python run_gp_lvm_ssvi.py --config ssvi_configs/original_ssvi_config.yaml
"""
from typing import get_type_hints
import argparse, yaml, torch
import dataclasses
import datetime
from pathlib import Path
from typing import Any, Dict
import numpy as np
from sklearn.model_selection import train_test_split

from src.gp_lvm_ssvi_core import train_gp_lvm_ssvi
from src.gp_dataclasses import *
from src.data_loaders.oil_data_loader import load_Y
from src.oil_dataset_plot_core import load_oil_fractions, plot_oil_dataset_gp_lvm_results
from src.helpers import initialize_latents_and_z
from src.evaluate_gp_metrics import evaluate_gp_lvm_model_metrics, save_metrics_json


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
            raise ValueError(f"Missing '{key}' in config")

        val = src[key]
        typ = type_hints[key]

        if dataclasses.is_dataclass(typ) and isinstance(val, dict):
            kwargs[key] = _to_dataclass(typ, val)
        else:
            try:
                kwargs[key] = typ(val)
            except Exception:
                kwargs[key] = val  # fallback if conversion fails

    return cls(**kwargs)



def load_config(path: Path | str) -> GPSSVIConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_dataclass(GPSSVIConfig, raw)


def start_gp_lvm_ssvi_training(cfg: GPSSVIConfig):
    torch.set_default_dtype(torch.float64)

    train_gp_lvm_ssvi(cfg)

    print("Mocked training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(cfg)
    
    PROJECT_ROOT = Path(__file__).resolve().parent
    oil_data_path = PROJECT_ROOT / "oil_data"

    Y, labels = load_Y(oil_data_path, cfg.device)
    fractions = load_oil_fractions(oil_data_path)
    
    # train/test split
    N = Y.shape[0]
    train_idx, test_idx = train_test_split(np.arange(N), test_size=0.05, random_state=42)
    Y_train = Y[train_idx]
    
    init_latents_and_z_dict = initialize_latents_and_z(Y_train, cfg)
    train_results_dict = train_gp_lvm_ssvi(cfg, Y_train, init_latents_and_z_dict)
    
    metrics = evaluate_gp_lvm_model_metrics(train_results_dict, Y_train)

    RESULTS_ROOT = PROJECT_ROOT / "gp_lvm_ssvi_run_results"
    config_name = args.config.stem
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    save_results_path = RESULTS_ROOT / f"results_{config_name}_{timestamp}"

    plot_oil_dataset_gp_lvm_results(train_results_dict, labels[train_idx], fractions[train_idx], save_results_path)

    metrics_path = save_results_path / f"{config_name}_metrics.json"
    save_metrics_json(metrics, metrics_path)
    
    torch.save(train_results_dict, save_results_path / "trained_model_dict.pt")