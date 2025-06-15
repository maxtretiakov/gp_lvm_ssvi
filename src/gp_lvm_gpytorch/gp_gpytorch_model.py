from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from matplotlib import pyplot as plt
from tqdm.notebook import trange
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
import torch
import numpy as np


def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))


class bGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, init_latents_z_dict):
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        Z = init_latents_z_dict["Z"].to(dtype=torch.float64).detach().clone()
        self.inducing_inputs = Z.unsqueeze(0).expand(data_dim, -1, -1).clone()
        assert self.inducing_inputs.shape == (data_dim, n_inducing, latent_dim), \
            f"Expected inducing_inputs shape {(data_dim, n_inducing, latent_dim)}, got {self.inducing_inputs.shape}"

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))

        mu_init = init_latents_z_dict["mu_x"].to(dtype=torch.float64).detach().clone()
        log_sigma2_init = init_latents_z_dict["log_s2x"].to(dtype=torch.float64).detach().clone()
        assert mu_init.shape == (n, latent_dim),         "init_latents_z_dict['mu_x'] has wrong shape"
        assert log_sigma2_init.shape == (n, latent_dim), "init_latents_z_dict['log_s2x'] has wrong shape"
        std_init = (log_sigma2_init.exp()).sqrt()
        X_init = torch.nn.Parameter(mu_init)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
        X._variational_std = torch.nn.Parameter(std_init)  # override std

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        self.q_u_dist = q_u
        self.q_x_std = X._variational_std

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        
        print(f"[bGPLVM] Inducing shape: {self.inducing_inputs.shape} | mu_x shape: {mu_init.shape}")

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)