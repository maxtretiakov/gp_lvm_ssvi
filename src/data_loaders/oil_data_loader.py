import urllib.request
import tarfile
import numpy as np
import torch
from pathlib import Path

def load_Y(root: Path, device: str = "auto") -> torch.Tensor:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device.lower()

    root.mkdir(exist_ok=True)

    data_file = root / "DataTrn.txt"
    label_file = root / "DataTrnLbls.txt"
    archive_file = root / "3PhData.tar.gz"

    if not data_file.exists() or not label_file.exists():
        url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
        print(f"Downloading dataset from {url} ...")
        urllib.request.urlretrieve(url, archive_file)
        with tarfile.open(archive_file) as tar:
            tar.extract("DataTrn.txt", path=root)
            tar.extract("DataTrnLbls.txt", path=root)
        print("Dataset extracted.")

    Y_np = np.loadtxt(data_file)  # (N, D)
    Y = torch.tensor(Y_np, dtype=torch.float64, device=device)
    labels_np = np.loadtxt(label_file).astype(int)
    labels = torch.tensor(labels_np, dtype=torch.float64, device=device)
    return Y, labels
