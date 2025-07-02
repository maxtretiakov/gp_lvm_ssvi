# LVMOGP Model Evaluation Pipeline

This directory contains complete evaluation pipelines that **exactly replicate** the experimental setup from the `useful_notebook.ipynb`. The only difference is that we use the LVMOGP-SSVI model instead of the baseline models.

## Overview

The pipeline implements **both experimental setups** from the notebook:

### 1. Bayesian Optimization
- **"Learning Many Surfaces"** (`many_r`): Optimize all surfaces simultaneously
- **"One Surface Learning"** (`one_from_many_X_r`): Learn one surface while others are in training set
- **Expected Improvement** with target vector optimization (exact notebook implementation)
- **Metrics**: NLPD, RMSE, and Regret (matching notebook calculations)

### 2. Cross Validation
- **Predictive Performance Evaluation**: Test across different training set sizes (10%-95%)
- **Multiple Seeds**: Robust statistical evaluation across random splits
- **Standard Metrics**: RMSE and NLPD on train/test sets (exact notebook calculations)
- **Direct Comparison**: Results compatible with baseline models (MOGP, LMC, AvgGP, LVMOGP)

## Files Structure

```
# Bayesian Optimization
bo_ssvi_configs/
├── original_bo_ssvi_config.yaml      # Main comprehensive BO config
├── many_surfaces_bo_config.yaml      # For "learning many" experiments  
├── one_surface_bo_config.yaml        # For "one surface" experiments
└── quick_test_bo_config.yaml         # For quick BO testing/debugging

run_bo_gp_lvm_ssvi.py                 # BO script (replicated from notebook)
run_batch_experiments.py              # Batch BO runner for multiple experiments

# Cross Validation
cv_ssvi_configs/
├── original_cv_ssvi_config.yaml      # Main comprehensive CV config
└── quick_test_cv_config.yaml         # For quick CV testing/debugging

run_cv_gp_lvm_ssvi.py                 # CV script (replicated from notebook)
src/bayesian_optimization/cv_results_converter.py  # CV results converter
```

## Quick Start

### 1. Bayesian Optimization (Single Experiment)
```bash
# Test with quick config (few iterations, CPU)
python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/quick_test_bo_config.yaml

# Run "learning many surfaces" experiment
python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/many_surfaces_bo_config.yaml

# Run "one surface learning" experiment  
python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/one_surface_bo_config.yaml
```

### 2. Bayesian Optimization (Batch Experiments)
```bash
# Quick test of batch system
python run_batch_experiments.py --quick_test

# Run multiple experiments systematically
python run_batch_experiments.py --config_base bo_ssvi_configs/original_bo_ssvi_config.yaml
```

### 3. Cross Validation
```bash
# Quick test of CV pipeline (minimal seeds and train percentages)
python run_cv_gp_lvm_ssvi.py --config cv_ssvi_configs/quick_test_cv_config.yaml

# Full cross validation (matches notebook setup)
python run_cv_gp_lvm_ssvi.py --config cv_ssvi_configs/original_cv_ssvi_config.yaml

# Convert CV results to notebook format for comparison
python src/bayesian_optimization/cv_results_converter.py
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

#### Bayesian Optimization Results
**Exact format matching notebook baseline results:**

- `bayes_opt_{start_point}_{test_name}_seed_{seed}.csv`
- **Columns**: `['BP', 'GC', 'PrimerPairReporter', 'r', 'r_mu', 'r_sig2', 'EI_z', 'model', 'iteration', 'error from target r', 'seed', ...]`
- **Direct Comparison**: Load alongside `results/bayes_opt_0_point_start_learning_many.csv` from notebook
- **Row Format**: Each row = one BO iteration data point

#### Cross Validation Results
**Exact format matching notebook baseline results:**

- `cross_validation_lvmogp_ssvi.csv`
- **Columns**: `['seed', 'pct_train', 'param', 'no test points', 'no train points', 'lvm_ssvi_test_RMSE', 'lvm_ssvi_test_NLPD', 'lvm_ssvi_test_RMSE_z', 'lvm_ssvi_test_NLPD_z', ...]`
- **Direct Comparison**: Compatible with `results/cross_validation.csv` from notebook  
- **Row Format**: Each row = one CV fold (seed + train percentage combination)

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

#### Bayesian Optimization
```python
# Load notebook baseline results  
baseline_df = pd.read_csv('results/bayes_opt_0_point_start_learning_many.csv', index_col=0)

# Load our LVMOGP-SSVI results
our_df = pd.read_csv('gp_lvm_bo_run_results/.../bayes_opt_centre_many_r_seed_0.csv', index_col=0)

# Compare directly - same column structure!
print("Baseline models:", baseline_df['model'].unique())  # ['MOGP', 'LMC', 'AvgGP', 'LVMOGP']
print("Our model:", our_df['model'].unique())            # ['LVMOGP_SSVI']
```

#### Cross Validation
```python
# Load notebook baseline results
baseline_cv = pd.read_csv('results/cross_validation.csv', index_col=0)

# Load our LVMOGP-SSVI results 
our_cv = pd.read_csv('cross_validation_lvmogp_ssvi.csv', index_col=0)

# Combine for direct comparison
from src.bayesian_optimization.cv_results_converter import convert_cv_results_to_notebook_format
combined_cv = convert_cv_results_to_notebook_format(
    our_results_csv='cross_validation_lvmogp_ssvi.csv',
    baseline_results_csv='results/cross_validation.csv'
)

# Compare models: ['mo_indi', 'lmc', 'avg', 'lvm', 'lvm_ssvi']
```

### Results Comparison
1. **Cross Validation**: Load `results/cross_validation.csv` from notebook
2. **Bayesian Opt**: Load `results/bayes_opt_*.csv` from notebook  
3. **Our Results**: Load both CV and BO results from our directories
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
- **Bayesian Optimization**: Expected Improvement with identical target vector optimization
- **Cross Validation**: Predictive performance across identical train/test splits  
- **Metrics calculations** match the notebook implementations exactly
- **Results saved in two formats**: JSON (internal analysis) + CSV (notebook comparison)
- **Direct comparison**: CSV results can be loaded alongside baseline results
- **Column alignment**: Exact same structure as notebook baseline CSV files

The pipeline is ready for running the exact same benchmark experiments as the notebook.

### Complete Experiment Compatibility Summary
- **Data Loading**: Identical to notebook  
- **BO Pipeline**: Exact replication of acquisition/optimization experiments
- **CV Pipeline**: Exact replication of predictive performance experiments
- **Metrics**: Same calculations (RMSE, NLPD, Regret)
- **Results Format**: Compatible CSV + analysis JSON for both experiments
- **Direct Comparison**: Load our CSV files with baseline CSV files
- **Baseline Models**: Compare with MOGP, LMC, AvgGP, original LVMOGP 