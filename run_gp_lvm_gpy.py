#!/usr/bin/env python
"""
Run bGPLVM training from a YAML config.

$ python run_gp_lvm_gpy.py --config configs/bgplvm_config.yaml
"""
import argparse, yaml, dataclasses, torch
from typing import Any, get_type_hints
from pathlib import Path

from src.gp_lvm_gpy.gpy_dataclasses import BGPLVMConfig
from src.gp_lvm_gpy.gpy_training import train_bgplvm
from src.data_loaders.oil_data_loader import load_Y


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

    Y = load_Y(cfg.device)
    train_results_dict = train_bgplvm(cfg, Y)
