import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

def load_oil_fractions(data_dir: Path) -> np.ndarray:
    frctn_path = data_dir / "DataTrnFrctns.txt"
    archive_path = data_dir / "3PhData.tar.gz"

    if not frctn_path.exists():
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        with tarfile.open(archive_path) as tar:
            if "DataTrnFrctns.txt" not in tar.getnames():
                raise FileNotFoundError("DataTrnFrctns.txt not found in the archive.")
            tar.extract("DataTrnFrctns.txt", path=data_dir)

    w = np.loadtxt(frctn_path)  # (N, 2)
    w3 = 1.0 - w.sum(axis=1, keepdims=True)  # (N, 1)
    w_full = np.concatenate((w, w3), axis=1)  # (N, 3)

    return w_full


def plot_oil_dataset_gp_lvm_results(results: dict, labels, fractions, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)

    mu_x = results["mu_x"]
    log_alpha = results["log_alpha"]
    iters = results["elbo_iters"]
    elbo_vals = results["elbo_vals"]

    # ELBO and 2D latent
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(iters, elbo_vals, "-o")
    plt.grid(ls=":")
    plt.xlabel("iteration")
    plt.ylabel("ELBO")
    plt.title("ELBO trajectory")

    plt.subplot(1, 2, 2)
    prec = torch.exp(log_alpha)
    idx = torch.topk(prec, k=2).indices
    mu_vis = mu_x[:, idx]
    plt.scatter(mu_vis[:, 0], mu_vis[:, 1], c=labels.cpu().float(), cmap="brg", s=14)
    plt.gca().set_aspect("equal")
    plt.title(f"latent dims {idx[0].item()} & {idx[1].item()}")
    plt.xlabel("latent dim 1")
    plt.ylabel("latent dim 2")
    plt.tight_layout()

    path_main = save_path / "elbo_and_latents.png"
    plt.savefig(path_main, dpi=300)
    print(f"Saved plot to: {path_main}")
    plt.close()

    # Inverse lengthscale barplot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(torch.arange(len(log_alpha)), torch.exp(log_alpha))
    ax1.set_title("Inverse Lengthscale with SE-ARD kernel")
    ax1.grid(True, axis='y', ls=':')
    fig1.tight_layout()
    path_invlen = save_path / "inv_lengthscale.png"
    fig1.savefig(path_invlen, dpi=300)
    print(f"Saved inverse lengthscale plot to: {path_invlen}")
    plt.close(fig1)

    # Latent space by fractions
    fig2, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(iters, elbo_vals, "-o")
    axs[0].grid(ls=":")
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("ELBO")
    axs[0].set_title("ELBO trajectory")

    axs[1].scatter(mu_x[:, 0], mu_x[:, 1], c=fractions, s=14)
    axs[1].set_aspect("equal")
    axs[1].set_title("latent space")
    axs[1].set_xlabel("mu_1")
    axs[1].set_ylabel("mu_2")
    fig2.tight_layout()

    path_frac = save_path / "latent_space_by_fractions.png"
    fig2.savefig(path_frac, dpi=300)
    print(f"Saved latent space plot to: {path_frac}")
    plt.close(fig2)
    
    if "predictive_variance" in results:
        var = results["predictive_variance"].mean(dim=1)  # (N,)
        labels_np = labels.cpu().numpy()
        unique_labels = np.unique(labels_np)
        grouped_vars = [var[labels_np == label].numpy() for label in unique_labels]

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.boxplot(grouped_vars, labels=[f"Class {int(c)}" for c in unique_labels])
        ax3.set_title("Predictive variance by class")
        ax3.set_ylabel("mean predictive variance")
        ax3.set_xlabel("Class label")
        ax3.grid(ls=":")
        fig3.tight_layout()

        path_varbox = save_path / "predictive_variance_boxplot.png"
        fig3.savefig(path_varbox, dpi=300)
        print(f"Saved predictive variance boxplot to: {path_varbox}")
        plt.close(fig3)
