#!/usr/bin/env python3
"""
Generate all 27 learning rate tuning configurations for LVMOGP-SSVI.
3^3 combinations: lr_x × lr_hyp × lr_alpha
"""

import os
import yaml
from pathlib import Path

# Learning rate values to test
lr_values = {
    'lr_x': [5e-4, 1e-3, 5e-3],
    'lr_hyp': [5e-4, 1e-3, 5e-3], 
    'lr_alpha': [1e-3, 5e-3, 1e-2]
}

# Base config template
base_config = {
    'gp_ssvi': {
        'device': 'cuda',
        'debug': False,
        'dataset': {
            'type': 'oil',
            'n_samples': 1000,
            'noise': 0.1,
            'random_state': None
        },
        'lr': {
            'x': 1e-3,      # Will be overridden
            'hyp': 1e-3,    # Will be overridden
            'alpha': 5e-3   # Will be overridden
        },
        'training': {
            'batch_size': 128,
            'total_iters': 2000,
            'inner_iters': {
                'start': 40,
                'after': 30,
                'switch': 50
            }
        },
        'init_latent_dist': {
            'method': 'default',
            'custom_path': ''
        },
        'inducing': {
            'n_inducing': 64,
            'selection': 'perm',
            'seed': 19
        },
        'jitter': 5e-6,
        'max_exp': 60.0,
        'rho': {
            't0': 100.0,
            'k': 0.6
        },
        'q_latent': 5,
        'init_signal_to_noise_ratio': 1.0,
        'num_u_samples_per_iter': 5
    },
    'bo': {
        'bo_steps': 40,
        'seed': 0,
        'pct_train': 50,
        'test_name': 'many_r',
        'start_point': 'centre'
    },
    'performance': {
        'gradient_checkpointing': False,
        'monitor_convergence': True,
        'convergence_tolerance': 1e-6,
        'early_stopping_patience': 50,
        'log_frequency': 10,
        'save_intermediate': False
    }
}

def generate_configs():
    """Generate all 27 learning rate configurations."""
    
    # Create output directory
    output_dir = Path('bo_ssvi_configs/lr_tuning')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing configs
    for existing_file in output_dir.glob('lr_x_*.yaml'):
        existing_file.unlink()
    
    configs_generated = []
    
    # Generate all combinations
    for lr_x in lr_values['lr_x']:
        for lr_hyp in lr_values['lr_hyp']:
            for lr_alpha in lr_values['lr_alpha']:
                
                # Create config for this combination
                config = base_config.copy()
                config['gp_ssvi'] = base_config['gp_ssvi'].copy()
                config['gp_ssvi']['lr'] = {
                    'x': lr_x,
                    'hyp': lr_hyp,
                    'alpha': lr_alpha
                }
                
                # Format learning rates for filename (replace scientific notation)
                def format_lr(lr):
                    if lr == 5e-4:
                        return "5e-4"
                    elif lr == 1e-3:
                        return "1e-3"
                    elif lr == 5e-3:
                        return "5e-3"
                    elif lr == 1e-2:
                        return "1e-2"
                    else:
                        return str(lr)
                
                # Generate filename
                filename = f"lr_x_{format_lr(lr_x)}_hyp_{format_lr(lr_hyp)}_alpha_{format_lr(lr_alpha)}.yaml"
                filepath = output_dir / filename
                
                # Add comment header
                config_with_header = {
                    '# LR Tuning Config': f"lr_x={lr_x}, lr_hyp={lr_hyp}, lr_alpha={lr_alpha}",
                    '# Single run with 2000 iterations for convergence analysis': None
                }
                config_with_header.update(config)
                
                # Save config
                with open(filepath, 'w') as f:
                    # Write header comments
                    f.write(f"# LR Tuning Config: lr_x={lr_x}, lr_hyp={lr_hyp}, lr_alpha={lr_alpha}\n")
                    f.write("# Single run with 2000 iterations for convergence analysis\n\n")
                    
                    # Write config
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                
                configs_generated.append({
                    'filename': filename,
                    'lr_x': lr_x,
                    'lr_hyp': lr_hyp,
                    'lr_alpha': lr_alpha
                })
    
    return configs_generated

if __name__ == "__main__":
    print("Generating learning rate tuning configurations...")
    print(f"Total combinations: {len(lr_values['lr_x']) * len(lr_values['lr_hyp']) * len(lr_values['lr_alpha'])}")
    
    configs = generate_configs()
    
    print(f"\nGenerated {len(configs)} configuration files:")
    for i, config in enumerate(configs, 1):
        print(f"{i:2d}. {config['filename']}")
        print(f"    lr_x={config['lr_x']}, lr_hyp={config['lr_hyp']}, lr_alpha={config['lr_alpha']}")
    
    print(f"\nConfigurations saved to: bo_ssvi_configs/lr_tuning/")
    print("Ready to run with: ./jobs/run_lr_tuning_configs.sh") 