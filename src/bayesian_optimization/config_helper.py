import yaml
import dataclasses
from pathlib import Path
from typing import Any, get_type_hints

from src.gp_dataclasses import GPSSVIConfig, BOConfig

@dataclasses.dataclass
class RunConfig:
    seed: int
    pct_train: int
    test_name: str
    start_point: str  # only '0_point_start' or 'centre'

@dataclasses.dataclass
class FullConfig:
    gp_ssvi: GPSSVIConfig
    bo: BOConfig
    run: RunConfig  

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

def load_gp_ssvi_config(path: Path | str) -> GPSSVIConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_dataclass(GPSSVIConfig, raw)

def load_full_config(path: Path | str) -> FullConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_dataclass(FullConfig, raw)
