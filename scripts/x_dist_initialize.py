#!/usr/bin/env python
"""
Example:
    python scripts/x_dist_initialize.py \
           --method random --seed 42 \
           --out x_dist_init_inputs/oil_latents.json
"""
import argparse, tarfile, urllib.request, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT))

from src.x_dist_init_core import InitGenConfig, build_latents, save_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "oil_data"

def load_oil():
    DATA.mkdir(exist_ok=True)
    url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    arc = DATA / "3PhData.tar.gz"
    if not arc.exists():
        urllib.request.urlretrieve(url, arc)
    with tarfile.open(arc) as tar:
        tar.extract("DataTrn.txt", path=DATA)
    return torch.tensor(np.loadtxt(DATA / "DataTrn.txt"), dtype=torch.float64)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   help="pca | prior | random | isomap | umap")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    Y = load_oil()
    cfg = InitGenConfig(method=args.method, seed=args.seed)
    mu_x, log_s2x = build_latents(Y, Y.shape[1], torch.device("cpu"), cfg)
    save_json(mu_x, log_s2x, args.out)
    print("Saved ->", args.out)

if __name__ == "__main__":
    main()
