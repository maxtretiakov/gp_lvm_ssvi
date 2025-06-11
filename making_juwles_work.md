- Slurm command This command worked: 
```
salloc --partition=booster --gres=gpu:1 --ntasks=1 --account=hai_pac_bayes --time=01:00:00
```
info at https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/setup_editor/#test-openssh-proxyjump

- Make an envioron: https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/setup_environment/
--------
The bit above this actually didn't work
--------

- Actually worked 
```
srun --partition=booster --account=hai_pac_bayes --time=01:00:00 --pty bash
```
- python command: 
```
python gp_lvm_ssvi/run_gp_lvm_ssvi.py --config gp_lvm_ssvi/ssvi_configs/original_ssvi_config.yaml
```