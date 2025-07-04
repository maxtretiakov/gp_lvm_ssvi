#!/bin/bash
# Job script to run all 27 LR tuning configs for LVMOGP-SSVI
# Each config runs once with 2000 iterations and 10 BO steps for convergence analysis
# Logs all output to files while showing on console

echo "=========================================="
echo "Starting LR Tuning Experiments"
echo "Total configs: 27"
echo "Iterations per config: 2000"
echo "BO steps per config: 10"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Move to the parent directory (project root)
cd "$SCRIPT_DIR/.."

# Create logs directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="lr_tuning_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"

# Array of config files - all 27 learning rate combinations
configs=(
    # lr_x = 5e-4
    "bo_ssvi_configs/lr_tuning/lr_x_5e-4_hyp_5e-4_alpha_1e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-4_hyp_5e-4_alpha_5e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-4_hyp_5e-4_alpha_1e-2.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-4_hyp_1e-3_alpha_1e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-4_hyp_1e-3_alpha_5e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-4_hyp_1e-3_alpha_1e-2.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-4_hyp_5e-3_alpha_1e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-4_hyp_5e-3_alpha_5e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-4_hyp_5e-3_alpha_1e-2.yaml"
    
    # lr_x = 1e-3
    "bo_ssvi_configs/lr_tuning/lr_x_1e-3_hyp_5e-4_alpha_1e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_1e-3_hyp_5e-4_alpha_5e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_1e-3_hyp_5e-4_alpha_1e-2.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_1e-3_hyp_1e-3_alpha_1e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_1e-3_hyp_1e-3_alpha_5e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_1e-3_hyp_1e-3_alpha_1e-2.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_1e-3_hyp_5e-3_alpha_1e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_1e-3_hyp_5e-3_alpha_5e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_1e-3_hyp_5e-3_alpha_1e-2.yaml"
    
    # lr_x = 5e-3
    "bo_ssvi_configs/lr_tuning/lr_x_5e-3_hyp_5e-4_alpha_1e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-3_hyp_5e-4_alpha_5e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-3_hyp_5e-4_alpha_1e-2.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-3_hyp_1e-3_alpha_1e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-3_hyp_1e-3_alpha_5e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-3_hyp_1e-3_alpha_1e-2.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-3_hyp_5e-3_alpha_1e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-3_hyp_5e-3_alpha_5e-3.yaml"
    "bo_ssvi_configs/lr_tuning/lr_x_5e-3_hyp_5e-3_alpha_1e-2.yaml"
)

# Track timing
start_time=$(date +%s)
successful_runs=0
failed_runs=0

echo "Starting experiments at $(date)"
echo ""

# Run each config
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    config_name=$(basename "$config" .yaml)
    
    # Create log file for this config
    LOG_FILE="$LOG_DIR/${config_name}_$(date +"%H%M%S").txt"
    
    echo "=========================================="
    echo "Running config $((i+1))/27: $config_name"
    echo "Config file: $config"
    echo "Log file: $LOG_FILE"
    echo "Time: $(date)"
    echo "=========================================="
    
    # Log the config info to the log file
    {
        echo "=========================================="
        echo "LR Tuning Experiment Log"
        echo "Config: $config_name"
        echo "Config file: $config"
        echo "Started at: $(date)"
        echo "=========================================="
        echo ""
    } > "$LOG_FILE"
    
    # Run the experiment with tee to capture output to both console and log file
    # Capture both stdout and stderr
    python run_bo_gp_lvm_ssvi.py --config "$config" 2>&1 | tee -a "$LOG_FILE"
    
    # Get the exit code from the python command (not tee)
    exit_code=${PIPESTATUS[0]}
    
    # Log completion info
    {
        echo ""
        echo "=========================================="
        echo "Completed at: $(date)"
        echo "Exit code: $exit_code"
        echo "=========================================="
    } >> "$LOG_FILE"
    
    # Check if successful
    if [ $exit_code -eq 0 ]; then
        echo "✓ SUCCESS: $config_name"
        echo "✓ Log saved to: $LOG_FILE"
        ((successful_runs++))
    else
        echo "✗ FAILED: $config_name"
        echo "✗ Log saved to: $LOG_FILE"
        ((failed_runs++))
    fi
    
    echo ""
done

# Summary
end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_min=$((runtime / 60))
runtime_sec=$((runtime % 60))

echo "=========================================="
echo "LR TUNING EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Total configs run: 27"
echo "Successful: $successful_runs"
echo "Failed: $failed_runs"
echo "Success rate: $(( successful_runs * 100 / 27 ))%"
echo "Total runtime: ${runtime_min}m ${runtime_sec}s"
echo "Completed at: $(date)"
echo ""

if [ $successful_runs -gt 0 ]; then
    echo "Successful configs:"
    for i in "${!configs[@]}"; do
        config="${configs[$i]}"
        config_name=$(basename "$config" .yaml)
        echo "  - $config_name"
    done
fi

echo ""
echo "Results will be saved in gp_lvm_bo_run_results/"
echo "Training logs saved in: $LOG_DIR/"
echo "Check the logs for convergence analysis and training progress."
echo ""
echo "Log files:"
for config in "${configs[@]}"; do
    config_name=$(basename "$config" .yaml)
    echo "  - $LOG_DIR/${config_name}_*.txt"
done
echo ""
echo "To analyze results, use notebooks/useful_notebook.ipynb"
echo "==========================================" 