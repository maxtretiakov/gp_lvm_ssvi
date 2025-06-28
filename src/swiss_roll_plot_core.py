import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
from mpl_toolkits.mplot3d import Axes3D

def plot_swiss_roll_3d_original(Y: torch.Tensor, colors: torch.Tensor, save_path: Path, filename: str = "swiss_roll_3d_original.png"):
    """Plot the original 3D Swiss Roll data."""
    save_path.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    Y_np = Y.cpu().numpy()
    colors_np = colors.cpu().numpy()
    
    scatter = ax.scatter(Y_np[:, 0], Y_np[:, 1], Y_np[:, 2], c=colors_np, cmap='viridis', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('Original Swiss Roll Data (3D)')
    plt.colorbar(scatter)
    
    out_file = save_path / filename
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved 3D Swiss Roll plot to: {out_file}")
    plt.close(fig)

def plot_swiss_roll_latent_2d(mu_x: torch.Tensor, colors: torch.Tensor, save_path: Path, filename: str = "swiss_roll_latent_2d.png"):
    """Plot the learned 2D latent representation."""
    save_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mu_x_np = mu_x.cpu().numpy()
    colors_np = colors.cpu().numpy()
    
    scatter = ax.scatter(mu_x_np[:, 0], mu_x_np[:, 1], c=colors_np, cmap='viridis', s=10)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('Learned 2D Latent Representation')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter)
    
    out_file = save_path / filename
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved 2D latent plot to: {out_file}")
    plt.close(fig)

def plot_swiss_roll_latent_dimensions(mu_x: torch.Tensor, colors: torch.Tensor, save_path: Path, max_dims: int = 6):
    """Plot multiple latent dimensions in pairs."""
    save_path.mkdir(parents=True, exist_ok=True)
    
    mu_x_np = mu_x.cpu().numpy()
    colors_np = colors.cpu().numpy()
    
    n_dims = min(mu_x.shape[1], max_dims)
    n_pairs = n_dims // 2
    
    if n_pairs > 0:
        fig, axes = plt.subplots(2, n_pairs, figsize=(5*n_pairs, 10))
        if n_pairs == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_pairs):
            dim1, dim2 = 2*i, 2*i+1
            
            # Top row: scatter plot
            scatter = axes[0, i].scatter(mu_x_np[:, dim1], mu_x_np[:, dim2], 
                                       c=colors_np, cmap='viridis', s=10)
            axes[0, i].set_xlabel(f'Latent Dimension {dim1+1}')
            axes[0, i].set_ylabel(f'Latent Dimension {dim2+1}')
            axes[0, i].set_title(f'Latent Dims {dim1+1} vs {dim2+1}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Bottom row: marginal distributions
            axes[1, i].hist(mu_x_np[:, dim1], bins=30, alpha=0.7, label=f'Dim {dim1+1}')
            axes[1, i].hist(mu_x_np[:, dim2], bins=30, alpha=0.7, label=f'Dim {dim2+1}')
            axes[1, i].set_xlabel('Value')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_title(f'Marginal Distributions')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        out_file = save_path / "swiss_roll_latent_dimensions.png"
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"Saved latent dimensions plot to: {out_file}")
        plt.close(fig)

def plot_single_curve_from_results(
    results: dict,
    x_key: str,
    y_key: str,
    label: str,
    save_path: Path,
    filename: str,
    title: str = "Curve",
):
    """Generic helper to plot a single curve from a results dict."""
    save_path.mkdir(parents=True, exist_ok=True)

    x = results[x_key]
    y = results[y_key]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, label=label, marker="o")
    ax.set_xlabel(x_key)
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.grid(ls=":")
    ax.legend()
    fig.tight_layout()

    out_file = save_path / filename
    fig.savefig(out_file, dpi=300)
    print(f"Saved curve plot to: {out_file}")
    plt.close(fig)

def plot_swiss_roll_extra_curves(results_dict: dict, save_dir: Path):
    """Plot extra diagnostic curves for Swiss Roll results."""
    plot_single_curve_from_results(
        results_dict,
        x_key="elbo_iters",
        y_key="ll_vals",
        label="Log Likelihood",
        save_path=save_dir,
        filename="log_lik_iters.png",
        title="Log Likelihood over iterations",
    )

    plot_single_curve_from_results(
        results_dict,
        x_key="elbo_iters",
        y_key="klx_vals",
        label="KL_X",
        save_path=save_dir,
        filename="klx_iters.png",
        title="KL_X over iterations",
    )

    plot_single_curve_from_results(
        results_dict,
        x_key="elbo_iters",
        y_key="klu_vals",
        label="KL_U",
        save_path=save_dir,
        filename="klu_iters.png",
        title="KL_U over iterations",
    )

def plot_swiss_roll_gp_lvm_results(results: dict, Y: torch.Tensor, colors: torch.Tensor, save_path: Path):
    """Plot comprehensive Swiss Roll GP-LVM results."""
    save_path.mkdir(parents=True, exist_ok=True)

    mu_x = results["mu_x"]
    log_alpha = results["log_alpha"]
    iters = results["elbo_iters"]
    elbo_vals = results["elbo_vals"]

    # Main results plot: ELBO and 2D latent
    plt.figure(figsize=(15, 5))

    # ELBO trajectory
    plt.subplot(1, 3, 1)
    plt.plot(iters, elbo_vals, "-o")
    plt.grid(ls=":")
    plt.xlabel("iteration")
    plt.ylabel("ELBO")
    plt.title("ELBO trajectory")

    # Original 3D data
    ax2 = plt.subplot(1, 3, 2, projection='3d')
    Y_np = Y.cpu().numpy()
    colors_np = colors.cpu().numpy()
    ax2.scatter(Y_np[:, 0], Y_np[:, 1], Y_np[:, 2], c=colors_np, cmap='viridis', s=10)
    ax2.set_title("Original 3D Swiss Roll")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # 2D latent representation
    plt.subplot(1, 3, 3)
    prec = torch.exp(log_alpha)
    idx = torch.topk(prec, k=2).indices
    mu_vis = mu_x[:, idx]
    plt.scatter(mu_vis[:, 0], mu_vis[:, 1], c=colors.cpu().numpy(), cmap='viridis', s=10)
    plt.gca().set_aspect("equal")
    plt.title(f"Latent space (dims {idx[0].item()+1} & {idx[1].item()+1})")
    plt.xlabel("latent dim 1")
    plt.ylabel("latent dim 2")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

    path_main = save_path / "swiss_roll_main_results.png"
    plt.savefig(path_main, dpi=300, bbox_inches='tight')
    print(f"Saved main results plot to: {path_main}")
    plt.close()

    # Inverse lengthscale barplot
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(torch.arange(len(log_alpha)), torch.exp(log_alpha))
    ax1.set_title("Inverse Lengthscale with SE-ARD kernel")
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("Inverse Lengthscale")
    ax1.grid(True, axis='y', ls=':')
    fig1.tight_layout()
    path_invlen = save_path / "inv_lengthscale.png"
    fig1.savefig(path_invlen, dpi=300)
    print(f"Saved inverse lengthscale plot to: {path_invlen}")
    plt.close(fig1)

    # Additional visualizations
    plot_swiss_roll_3d_original(Y, colors, save_path)
    plot_swiss_roll_latent_2d(mu_x, colors, save_path)
    plot_swiss_roll_latent_dimensions(mu_x, colors, save_path)

    # Predictive variance analysis if available
    if "predictive_variance" in results:
        var = results["predictive_variance"]
        if var.ndim == 2:
            var = var.mean(dim=1)
        elif var.ndim == 1:
            pass
        else:
            raise ValueError(f"Unexpected shape for predictive_variance: {var.shape}")

        var_np = var.cpu().numpy()

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.scatter(range(len(var_np)), var_np, c=colors.cpu().numpy(), cmap='viridis', s=10)
        ax3.set_title("Predictive variance across data points")
        ax3.set_ylabel("Mean predictive variance")
        ax3.set_xlabel("Data point index")
        ax3.grid(ls=":")
        fig3.tight_layout()

        path_varscatter = save_path / "predictive_variance_scatter.png"
        fig3.savefig(path_varscatter, dpi=300)
        print(f"Saved predictive variance scatter plot to: {path_varscatter}")
        plt.close(fig3) 