# Learning Rate Tuning for LVMOGP-SSVI

This directory contains 27 configurations for comprehensive learning rate tuning of the LVMOGP-SSVI model.

## Overview

Instead of running the full parameter tuning suite with 3^4 = 81 configurations × 24 runs each (1,944 total runs), this setup provides a more efficient approach:

- **27 configurations** testing all learning rate combinations (3^3)
- **2000 iterations** per config for convergence analysis
- **40 BO steps** per config (as originally configured)
- **Single run** per config to analyze training logs

## Learning Rate Grid

The 27 configurations test all combinations of:
- `lr_x` (latent variables): [5e-4, 1e-3, 5e-3]
- `lr_hyp` (hyperparameters): [5e-4, 1e-3, 5e-3]
- `lr_alpha` (length scales): [1e-3, 5e-3, 1e-2]

### Configuration Files

All 27 combinations are systematically named as:
`lr_x_{LR_X}_hyp_{LR_HYP}_alpha_{LR_ALPHA}.yaml`

Examples:
- `lr_x_5e-4_hyp_5e-4_alpha_1e-3.yaml` - Most conservative
- `lr_x_1e-3_hyp_1e-3_alpha_5e-3.yaml` - Moderate baseline
- `lr_x_5e-3_hyp_5e-3_alpha_1e-2.yaml` - Most aggressive

The complete grid covers:
- **9 configs** with `lr_x=5e-4` (conservative latent learning)
- **9 configs** with `lr_x=1e-3` (moderate latent learning)
- **9 configs** with `lr_x=5e-3` (aggressive latent learning)

## Running the Experiments

### Option 1: Use Job Scripts (Recommended)

**Linux/Mac:**
```bash
cd jobs
./run_lr_tuning_configs.sh
```

**Windows (PowerShell):**
```powershell
cd jobs
.\run_lr_tuning_configs.ps1
```

**Windows (Command Prompt):**
```cmd
cd jobs
run_lr_tuning_configs.bat
```

### Option 2: Run Individual Configs

```bash
python run_bo_gp_lvm_ssvi.py --config bo_ssvi_configs/lr_tuning/lr_x_1e-3_hyp_1e-3_alpha_5e-3.yaml
```

## Logging Features

The job script automatically captures all training output to log files while still showing progress on the console:

### Log Directory Structure
```
lr_tuning_logs_YYYYMMDD_HHMMSS/
├── lr_x_5e-4_hyp_5e-4_alpha_1e-3_HHMMSS.txt
├── lr_x_5e-4_hyp_5e-4_alpha_5e-3_HHMMSS.txt
├── lr_x_5e-4_hyp_5e-4_alpha_1e-2_HHMMSS.txt
├── lr_x_5e-4_hyp_1e-3_alpha_1e-3_HHMMSS.txt
├── lr_x_5e-4_hyp_1e-3_alpha_5e-3_HHMMSS.txt
├── ... (27 total log files)
├── lr_x_5e-3_hyp_5e-3_alpha_5e-3_HHMMSS.txt
└── lr_x_5e-3_hyp_5e-3_alpha_1e-2_HHMMSS.txt
```

### Log Analysis

The logs are saved as text files (`.txt`) and can be viewed directly:

```bash
# List all log files
ls lr_tuning_logs_TIMESTAMP/

# View specific config logs
cat lr_tuning_logs_TIMESTAMP/lr_x_1e-3_hyp_1e-3_alpha_5e-3_*.txt

# Tail a running log
tail -f lr_tuning_logs_TIMESTAMP/lr_x_1e-3_hyp_1e-3_alpha_5e-3_*.txt

# Search for specific patterns (e.g., loss values)
grep "loss" lr_tuning_logs_TIMESTAMP/*.txt
```

## Expected Results

- **Training time**: ~20-30 minutes per config (depends on hardware)
- **Total time**: ~9-13.5 hours for all 27 configs
- **Output**: Results saved in `gp_lvm_bo_run_results/`
- **Logs**: Complete training logs saved in `lr_tuning_logs_TIMESTAMP/`
- **Console**: Real-time progress shown on console simultaneously

## Analysis

After running the experiments:

1. **Check convergence**: Analyze training logs to see which learning rates converge fastest
   ```bash
   # Search for loss patterns in logs
   grep "loss" lr_tuning_logs_TIMESTAMP/*.txt
   
   # View specific config logs
   cat lr_tuning_logs_TIMESTAMP/lr_x_1e-3_hyp_1e-3_alpha_5e-3_*.txt
   ```

2. **Compare performance**: Use `notebooks/useful_notebook.ipynb` to analyze final results

3. **Select best parameters**: Choose the configuration with best convergence/performance balance based on:
   - **Convergence speed** (from training logs)
   - **Final performance** (from result files)
   - **Training stability** (from loss curves in logs)

## Key Features

- **Comprehensive**: Complete 3×3×3 grid covering all learning rate combinations
- **Convergence focus**: 2000 iterations to see full training dynamics
- **Systematic exploration**: All interactions between lr_x, lr_hyp, and lr_alpha
- **Log analysis**: Focus on training logs rather than final metrics

## Notes

- All configs use the same experimental setup: `many_r` scenario with `centre` start point
- Other parameters (n_inducing=64, q_latent=5, etc.) are fixed to match the original bio experiments
- Results can be directly compared to find optimal learning rate combinations 