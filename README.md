# GP-LVM SSVI

This repository implements a **Bayesian Gaussian Process Latent Variable Model (GP-LVM)** trained using **Structured Stochastic Variational Inference (SSVI)**, inspired by the paper by Hoffman & Blei (2015).

---

## 🚀 Quick Start

### 1. 📦 Install Requirements
Ensure you have **Python 3.11** installed.

Install dependencies:
```bash
pip install -r requirements.txt
```


### 2. ▶️ Run SSVI-based GP-LVM (main model)

```bash
python run_gp_lvm_ssvi.py --config configs/original_ssvi_config.yaml
```


### 3. ▶️ Run GP-LVM (classic version using GPy)

```bash
python run_gp_lvm_gpy.py
```


### 4. 🔧 Custom initialization of latent space (optional)

By default, latent variables `mu_x` and `log_s2x` are initialized via **PCA**.

You can generate custom initialization using different methods (`random`, `pca`, `prior`, `isomap`, `umap`) via:

```bash
python scripts/x_dist_initialize.py \
       --method random --seed 42 \
       --out x_dist_init_inputs/oil_latents.json
```

Then update your config file:
```yaml
# Inside your config.yaml
init:
  method: custom
  custom_path: ./x_dist_init_inputs/oil_latents.json
```

Both models (`run_gp_lvm_ssvi.py` and `run_gp_lvm_gpy.py`) support this custom latent initialization.

---
