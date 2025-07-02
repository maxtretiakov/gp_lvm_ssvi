# GP-LVM SSVI

This repository implements a **Bayesian Gaussian Process Latent Variable Model (GP-LVM)** trained using **Structured Stochastic Variational Inference (SSVI)**, inspired by the paper by Hoffman & Blei (2015).


## 🚀 Quick Start

### 1. 📦 Install Requirements

Please make sure you have **Python 3.11** installed.

Install dependencies:

```bash
pip install -r requirements.txt
```


### 2. ▶️ Run GP-LVM SSVI (main model)

Run a single experiment with a specific config file:

```bash
python run_gp_lvm_ssvi.py --config ssvi_configs/original_ssvi_config.yaml
```


### 3. ▶️ Run GP-LVM (classic version using GpyTorch)

```bash
python run_gp_lvm_gpytorch.py --config gp_lvm_gpytorch_configs/original_gp_lvm_gpytorch_config.yaml
```


### 4. 🔁 Run Multiple Configs from Folder

To automatically run all configuration files (`.yaml`) from a folder:

#### ✅ SSVI experiments

```bash
bash jobs/run_ssvi_configs_in_the_folder.sh ssvi_configs
```

#### ✅ GpyTorch experiments

```bash
bash jobs/run_gp_lvm_gpytorch_configs_in_the_folder.sh gp_gpytorch_configs
```

These scripts iterate over all YAML files in the given folder and run each experiment sequentially.


### 5. 🧬 Run Biological Experiments (PCR Data)

For **Bayesian Optimization** and **Cross Validation** experiments with PCR/biological data:

```bash
# Bayesian optimization experiments
python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/original_bo_ssvi_config.yaml

# Cross validation experiments  
python run_cv_gp_lvm_ssvi.py --config cv_ssvi_configs/original_cv_ssvi_config.yaml
```

See **[LVMOGP_SSVI_BIOEXP_README.md](LVMOGP_SSVI_BIOEXP_README.md)** for complete biological experiments documentation.


### 6. 🔧 Custom Initialization of Latent Space (optional)

By default, latent variables `mu_x` and `log_s2x` are initialized via **PCA**.

You can generate custom initialization using different methods (`random`, `pca`, `prior`, `isomap`, `umap`) via:

```bash
python scripts/x_dist_initialize.py \Add commentMore actions
       --method random \
       --q_latent 12 \
       --seed 42 \
       --out x_dist_init_inputs/oil_latents.json
```

Then update your config file:

```yaml
# Inside your config.yaml
init:
  method: custom
  custom_path: ./x_dist_init_inputs/oil_latents.json
```

Both models (`run_gp_lvm_ssvi.py` and `run_gp_lvm_gpytorch.py`) support this custom latent initialization.
