#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=your_account_name
#SBATCH --cpus-per-task=32

source "<path/to/sc_venv_template>"

cd ..
cd src

# train
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" python train.py

export CUDA_VISIBLE_DEVICES=0,1,2,3      # Very important to make the GPUs visible
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"  # Get number of cpu per task in the script

source sc_venv_template/activate.sh      # Command to activate the environment
python run_gp_lvm_ssvi.py --config ssvi_configs/original_ssvi_config.yaml
