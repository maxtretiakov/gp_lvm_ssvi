#!/usr/bin/env python
"""
Run bGPLVM training from a YAML config.

$ python run_gp_lvm_gpy.py --config gpy_configs/original_gpy_config.yaml
"""
import argparse, yaml, dataclasses, torch
from typing import Any, get_type_hints
from pathlib import Path
import datetime

from src.gp_lvm_gpy.gpy_dataclasses import BGPLVMConfig
from src.gp_lvm_gpy.gpy_training import train_bgplvm
from src.data_loaders.oil_data_loader import load_Y
from src.oil_dataset_plot_core import load_oil_fractions, plot_oil_dataset_gp_lvm_results
from src.helpers import initialize_latents_and_z


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
    print(cfg)

    PROJECT_ROOT = Path(__file__).resolve().parent
    oil_data_path = PROJECT_ROOT / "oil_data"

    Y, labels = load_Y(oil_data_path, cfg.device)
    fractions = load_oil_fractions(oil_data_path)
    
    init_latents_and_z_dict = initialize_latents_and_z(Y, cfg)    
    train_results_dict = train_bgplvm(cfg, Y, init_latents_and_z_dict)
    

    RESULTS_ROOT = PROJECT_ROOT / "gp_lvm_gpy_run_results"
    config_name = args.config.stem
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    save_results_path = RESULTS_ROOT / f"results_{config_name}_{timestamp}"
    
    plot_oil_dataset_gp_lvm_results(train_results_dict, labels, fractions, save_results_path)