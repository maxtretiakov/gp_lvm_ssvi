# LVMOGP Bayesian Optimization Pipeline

This directory contains a complete Bayesian Optimization pipeline that **exactly replicates** the experimental setup from the `useful_notebook.ipynb`. The only difference is that we use the LVMOGP-SSVI model instead of the baseline models.

## Overview

The pipeline implements:
- **"Learning Many Surfaces"** (`many_r`): Optimize all surfaces simultaneously
- **"One Surface Learning"** (`one_from_many_X_r`): Learn one surface while others are in training set
- **Expected Improvement** with target vector optimization (exact notebook implementation)
- **Metrics**: NLPD, RMSE, and Regret (matching notebook calculations)

## Files Structure

```
bo_ssvi_configs/
├── original_bo_ssvi_config.yaml      # Main comprehensive config
├── many_surfaces_bo_config.yaml      # For "learning many" experiments  
├── one_surface_bo_config.yaml        # For "one surface" experiments
└── quick_test_bo_config.yaml         # For quick testing/debugging

run_bo_gp_lvm_ssvi.py                 # Main script (replicated from notebook)
run_batch_experiments.py              # Batch runner for multiple experiments
```

## Quick Start

### 1. Single Experiment
```bash
# Test with quick config (few iterations, CPU)
python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/quick_test_bo_config.yaml

# Run "learning many surfaces" experiment
python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/many_surfaces_bo_config.yaml

# Run "one surface learning" experiment  
python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/one_surface_bo_config.yaml
```

### 2. Batch Experiments
```bash
# Quick test of batch system
python run_batch_experiments.py --quick_test

# Run multiple experiments systematically
python run_batch_experiments.py --config_base bo_ssvi_configs/original_bo_ssvi_config.yaml
```

## Configuration Parameters

### Core LVMOGP Parameters
- `q_latent: 12` - Latent dimension (matches notebook)
- `n_inducing: 64` - Number of inducing points
- `total_iters: 1000` - SSVI training iterations
- `device: cuda` - Use GPU (change to `cpu` if needed)

### BO Parameters
- `bo_steps: 15` - Number of BO iterations
- `test_name: many_r` - Experiment type
- `start_point: centre` - Starting point strategy
- `seed: 0` - Random seed for reproducibility

### Test Name Options
```yaml
# Learning many surfaces
test_name: many_r

# Learning specific surfaces
test_name: one_from_many_FP004-RP004-Probe_r
test_name: one_from_many_FP001-RP001x-EvaGreen_r
test_name: one_from_many_FP002-RP002x-EvaGreen_r
# ... (see config files for complete list)
```

## Results

Results are saved in **two formats** for maximum compatibility:

### 1. Internal JSON Format (Analysis)
Saved in `gp_lvm_bo_run_results/bo_results_<config>_<timestamp>/`:

- `config_used_<timestamp>.json` - Complete config used
- `bo_metrics_<timestamp>.json` - BO metrics (NLPD, RMSE, regret)
- `final_model_state_<timestamp>.pt` - Final model state
- `ei_values_<timestamp>.json` - Expected improvement values

```json
{
  "chosen_indices": [45, 123, 67, ...],
  "nlpd_values": [1.23, 1.15, 1.08, ...],
  "rmse_values": [0.45, 0.42, 0.38, ...], 
  "regret_values": [0.12, 0.09, 0.07, ...],
  "surfaces_optimized": ["FP004-RP004-Probe", ...]
}
```

### 2. Notebook-Compatible CSV Format (Comparison)
**Exact format matching notebook baseline results:**

- `bayes_opt_{start_point}_{test_name}_seed_{seed}.csv`
- **Columns**: `['BP', 'GC', 'PrimerPairReporter', 'r', 'r_mu', 'r_sig2', 'EI_z', 'model', 'iteration', 'error from target r', 'seed', ...]`
- **Direct Comparison**: Load alongside `results/bayes_opt_0_point_start_learning_many.csv` from notebook
- **Row Format**: Each row = one BO iteration data point

## Experimental Setup (Notebook Replication)

### 1. "Learning Many Surfaces" (`many_r`)
- **Setup**: All surfaces have some initial training data
- **BO Process**: Each iteration adds one point to the best surface
- **Surfaces**: All available surfaces optimized simultaneously
- **Config**: `many_surfaces_bo_config.yaml`

### 2. "One Surface Learning" (`one_from_many_X_r`)  
- **Setup**: All surfaces except X have full training data
- **BO Process**: Each iteration adds one point to surface X only
- **Surfaces**: Only the specified surface X is optimized
- **Config**: `one_surface_bo_config.yaml`

### 3. Starting Points
- `centre`: First point is the most central point on surface
- `0_point_start`: Model chooses first point (LVMOGP can do this)

## Model Differences from Notebook

| **Notebook Baselines** | **Our LVMOGP-SSVI** |
|------------------------|---------------------|
| Pure latent space GP | **[BP, GC] + H** where H is per-surface |
| Various implementations | **SSVI training** with structured variational inference |
| Different latent handling | **H per surface** (more interpretable) |

## Comparison with Baselines

Results are **fully compatible** with notebook baselines for direct comparison:

### Loading Results for Comparison
```python
# Load notebook baseline results  
baseline_df = pd.read_csv('results/bayes_opt_0_point_start_learning_many.csv', index_col=0)

# Load our LVMOGP-SSVI results
our_df = pd.read_csv('gp_lvm_bo_run_results/.../bayes_opt_centre_many_r_seed_0.csv', index_col=0)

# Compare directly - same column structure!
print("Baseline models:", baseline_df['model'].unique())  # ['MOGP', 'LMC', 'AvgGP', 'LVMOGP']
print("Our model:", our_df['model'].unique())            # ['LVMOGP_SSVI']
```

### Results Comparison
1. **Cross Validation**: Load `results/cross_validation.csv` from notebook
2. **Bayesian Opt**: Load `results/bayes_opt_*.csv` from notebook  
3. **Our Results**: Load `bayes_opt_*.csv` from our results directories
4. **Analysis**: Use identical plotting/analysis code from notebook

### Metrics Comparison
- **NLPD**: Lower is better (predictive uncertainty)
- **RMSE**: Lower is better (prediction accuracy)  
- **Regret**: Lower is better (optimization performance)
- **Error from Target**: Direct distance to known optimal values

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `batch_size`, `n_inducing`, or use `device: cpu`
2. **Long training time**: Reduce `total_iters` or `bo_steps` for testing
3. **Missing data files**: Ensure `data/data.csv` and targets CSV exist
4. **Surface not found**: Check surface names match exactly (case-sensitive)

### Debug Mode
```yaml
gp_ssvi:
  debug: true
  device: cpu
  training:
    total_iters: 20
bo:
  bo_steps: 2
```

## Next Steps

1. **Quick Test**: Start with `quick_test_bo_config.yaml`
2. **Single Run**: Try `many_surfaces_bo_config.yaml`  
3. **Batch Runs**: Use `run_batch_experiments.py`
4. **Compare Results**: Analyze against notebook baselines
5. **Publication**: Use results for model comparison

## Notes

- All configs use the **exact same data splits** as the notebook
- **Expected Improvement** uses identical target vector optimization
- **Metrics calculations** match the notebook implementations exactly
- **Results saved in two formats**: JSON (internal analysis) + CSV (notebook comparison)
- **Direct comparison**: CSV results can be loaded alongside baseline results
- **Column alignment**: Exact same structure as notebook baseline CSV files

The pipeline is ready for running the exact same benchmark experiments as the notebook.

### Result Compatibility Summary
- **Data Loading**: Identical to notebook  
- **BO Pipeline**: Exact replication  
- **Metrics**: Same calculations  
- **Results Format**: Compatible CSV + analysis JSON  
- **Direct Comparison**: Load our CSV with baseline CSV files 