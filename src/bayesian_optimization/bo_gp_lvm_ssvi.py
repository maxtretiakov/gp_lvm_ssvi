import torch
import numpy as np
import pandas as pd
from src.lvmogp.lvmogp_ssvi import LVMOGP_SSVI_Torch
from src.bayesian_optimization.expected_improvement import ExpectedImprovement
from src.bayesian_optimization.metrics_helper import get_nlpd, get_squared_error, get_regret


def prepare_lvmogp_data(train_df, test_df, targets, Q=2, device=None):
    """
    Prepare data for LVMOGP following the exact pipeline from the notebook.
    
    Args:
        train_df: Training dataframe with columns ['BP', 'GC', 'PrimerPairReporter', 'Value']
        test_df: Test dataframe for acquisition 
        targets: Targets dataframe with PrimerPairReporter and Target Rate
        Q: Latent dimension
        device: torch device
    
    Returns:
        Dict with all data needed for LVMOGP
    """
    device = device or torch.device('cpu')
    
    # Get all unique surfaces 
    all_pprs = pd.concat([train_df['PrimerPairReporter'], test_df['PrimerPairReporter']]).unique()
    ppr_to_idx = {ppr: idx for idx, ppr in enumerate(all_pprs)}
    num_surfaces = len(all_pprs)
    
    # Training data
    X_train = torch.tensor(train_df[['BP', 'GC']].values, dtype=torch.float64, device=device)
    Y_train = torch.tensor(train_df['Value'].values, dtype=torch.float64, device=device).unsqueeze(-1)
    fn_train = torch.tensor([ppr_to_idx[ppr] for ppr in train_df['PrimerPairReporter']], dtype=torch.long, device=device)
    
    # Test data for acquisition
    X_test = torch.tensor(test_df[['BP', 'GC']].values, dtype=torch.float64, device=device) 
    fn_test = torch.tensor([ppr_to_idx[ppr] for ppr in test_df['PrimerPairReporter']], dtype=torch.long, device=device)
    
    # Initialize H (latent variables per surface) - small random as in notebook  
    H_mean = torch.randn(num_surfaces, Q, dtype=torch.float64, device=device) * 0.1
    H_var = torch.ones(num_surfaces, Q, dtype=torch.float64, device=device) * 0.1
    
    # Prepare targets
    target_dict = {}
    for _, row in targets.iterrows():
        if row['PrimerPairReporter'] in ppr_to_idx:
            target_dict[row['PrimerPairReporter']] = row['Target Rate']
    
    return {
        'X_train': X_train,
        'Y_train': Y_train, 
        'fn_train': fn_train,
        'X_test': X_test,
        'fn_test': fn_test,
        'H_mean': H_mean,
        'H_var': H_var,
        'ppr_to_idx': ppr_to_idx,
        'idx_to_ppr': {idx: ppr for ppr, idx in ppr_to_idx.items()},
        'targets': target_dict,
        'num_surfaces': num_surfaces
    }


def add_new_data_point(train_data, x_new, y_new, ppr_new, ppr_to_idx):
    """Add a new data point to the training set."""
    device = train_data['X_train'].device
    
    # Convert new point
    x_new_tensor = torch.tensor(x_new, dtype=torch.float64, device=device).unsqueeze(0)
    y_new_tensor = torch.tensor([y_new], dtype=torch.float64, device=device).unsqueeze(-1)
    fn_new_tensor = torch.tensor([ppr_to_idx[ppr_new]], dtype=torch.long, device=device)
    
    # Concatenate
    train_data['X_train'] = torch.cat([train_data['X_train'], x_new_tensor], dim=0)
    train_data['Y_train'] = torch.cat([train_data['Y_train'], y_new_tensor], dim=0)
    train_data['fn_train'] = torch.cat([train_data['fn_train'], fn_new_tensor], dim=0)
    
    return train_data


def bayesian_optimization_loop(train_df, test_df, targets, config, 
                               K_steps=5, test_name="many_r", 
                               start_point="centre", device=None):
    """
    Run Bayesian optimization loop exactly as described in the notebook but using LVMOGP.
    
    Args:
        train_df: Initial training dataframe
        test_df: Test dataframe (acquisition pool)
        targets: Targets dataframe 
        config: SSVI config for LVMOGP
        K_steps: Number of BO steps
        test_name: Type of test (e.g., "many_r", "one_from_many_FP004-RP004-Probe_r")
        start_point: Starting point strategy ("centre" or "0_point_start")
        device: torch device
    """
    device = device or torch.device('cpu')
    
    # Prepare data for LVMOGP
    data_dict = prepare_lvmogp_data(train_df, test_df, targets, Q=config.q_latent, device=device)
    
    # Initialize Expected Improvement calculator (from notebook)
    ei_calculator = ExpectedImprovement()
    
    # Storage for results
    chosen_indices = []
    ei_values = []
    nlpd_values = []
    rmse_values = []
    regret_values = []
    pred_mean_history = []  # For notebook compatibility
    pred_var_history = []   # For notebook compatibility
    
    print(f"Starting BO with {len(train_df)} training points and {len(test_df)} test points")
    print(f"Test type: {test_name}, Start point: {start_point}")
    
    for k in range(K_steps):
        print(f"\nBO Step {k + 1}/{K_steps}")
        print(f"Current training set size: {len(data_dict['Y_train'])}")
        
        # Create and train LVMOGP model
        model = LVMOGP_SSVI_Torch(
            data=data_dict['Y_train'],
            X_data=data_dict['X_train'], 
            X_data_fn=data_dict['fn_train'].unsqueeze(-1),
            H_data_mean=data_dict['H_mean'],
            H_data_var=data_dict['H_var'],
            num_inducing_variables=config.inducing.n_inducing,
            device=device
        )
        
        # Train the model
        print("Training LVMOGP...")
        model.ssvi_train(config)
        print("Training complete.")
        
        # Make predictions on test set
        print("Making predictions...")
        pred_mean, pred_var = model.predict_y((data_dict['X_test'], data_dict['fn_test']))
        pred_mean = pred_mean.squeeze(-1).detach().cpu().numpy()  # (N_test,)
        pred_var = pred_var.squeeze(-1).detach().cpu().numpy()   # (N_test,)
        
        # Store predictions for notebook compatibility
        pred_mean_history.append(pred_mean.copy())
        pred_var_history.append(pred_var.copy())
        
        # Determine which surfaces to optimize based on test_name
        if test_name == "many_r":
            # Learning many surfaces - optimize all surfaces that have test points
            surfaces_to_optimize = test_df['PrimerPairReporter'].unique()
        else:
            # One from many - extract surface name from test_name
            # Format: "one_from_many_FP004-RP004-Probe_r"
            surface_name = test_name.replace("one_from_many_", "").replace("_r", "")
            surfaces_to_optimize = [surface_name] if surface_name in data_dict['targets'] else []
        
        print(f"Optimizing surfaces: {surfaces_to_optimize}")
        
        # Calculate EI for each surface and select best point
        best_ei = -float('inf')
        best_idx = None
        best_ppr = None
        
        for ppr in surfaces_to_optimize:
            if ppr not in data_dict['targets']:
                continue
                
            # Get test points for this surface
            surface_mask = test_df['PrimerPairReporter'] == ppr
            if not surface_mask.any():
                continue
                
            surface_indices = np.where(surface_mask)[0]
            
            # Get predictions for this surface
            surface_pred_mean = pred_mean[surface_mask]
            surface_pred_var = pred_var[surface_mask]
            
            # Calculate best_yet for this surface (from training data)
            surface_train_mask = data_dict['fn_train'] == data_dict['ppr_to_idx'][ppr]
            if surface_train_mask.any():
                surface_train_y = data_dict['Y_train'][surface_train_mask].cpu().numpy()
                target_rate = data_dict['targets'][ppr]
                best_yet = ei_calculator.BestYet(surface_train_y.flatten(), {'Target Rate': target_rate})
            else:
                # No points observed on this surface yet
                best_yet = 4.0  # As in notebook
            
            # Calculate EI for this surface
            pred_df = {
                'mu': surface_pred_mean,
                'sig2': surface_pred_var
            }
            target_dict = {'Target Rate': data_dict['targets'][ppr]}
            
            ei_surface = ei_calculator.EI(pred_df, target_dict, best_yet, ['r'])
            ei_surface[np.isnan(ei_surface)] = 0
            
            # Find best point on this surface
            max_ei_idx = np.argmax(ei_surface)
            max_ei_val = ei_surface[max_ei_idx]
            
            if max_ei_val > best_ei:
                best_ei = max_ei_val
                best_idx = surface_indices[max_ei_idx]
                best_ppr = ppr
        
        if best_idx is None:
            print("No valid acquisition point found!")
            break
            
        print(f"Selected point {best_idx} on surface {best_ppr} with EI = {best_ei:.4f}")
        
        # Get the selected point from test set
        selected_row = test_df.iloc[best_idx]
        x_new = [selected_row['BP'], selected_row['GC']]
        y_new = selected_row['Value']  # This is the "oracle" value
        
        print(f"Adding point: BP={x_new[0]:.3f}, GC={x_new[1]:.3f}, Value={y_new:.4f}")
        
        # Add new point to training data
        data_dict = add_new_data_point(data_dict, x_new, y_new, best_ppr, data_dict['ppr_to_idx'])
        
        # Store results
        chosen_indices.append(best_idx)
        ei_values.append(best_ei)  # Store the actual EI value that was selected
        
        # Calculate metrics for each optimized surface
        step_nlpd = []
        step_rmse = []
        step_regret = []
        
        for ppr in surfaces_to_optimize:
            if ppr not in data_dict['targets']:
                continue
                
            surface_mask = test_df['PrimerPairReporter'] == ppr
            if not surface_mask.any():
                continue
                
            surface_pred_mean = pred_mean[surface_mask]
            surface_pred_var = pred_var[surface_mask]
            ys_true = test_df.loc[surface_mask, 'Value'].values
            target_rate = data_dict['targets'][ppr]
            
            # Calculate metrics
            nlpd = get_nlpd(surface_pred_mean, surface_pred_var, ys_true)
            squared_error = get_squared_error(surface_pred_mean, ys_true)
            
            # Calculate best_yet for regret
            surface_train_mask = data_dict['fn_train'] == data_dict['ppr_to_idx'][ppr]
            if surface_train_mask.any():
                surface_train_y = data_dict['Y_train'][surface_train_mask].cpu().numpy()
                best_yet_regret = ei_calculator.BestYet(surface_train_y.flatten(), {'Target Rate': target_rate})
            else:
                best_yet_regret = 4.0
                
            regret = get_regret(ys_true, best_yet_regret, target_rate)
            
            step_nlpd.append(np.mean(nlpd))
            step_rmse.append(np.sqrt(np.mean(squared_error)))
            step_regret.append(np.min(regret))
        
        # Store average metrics across surfaces
        nlpd_values.append(np.mean(step_nlpd) if step_nlpd else 0.0)
        rmse_values.append(np.mean(step_rmse) if step_rmse else 0.0)
        regret_values.append(np.mean(step_regret) if step_regret else 0.0)
        
        print(f"Metrics - NLPD: {nlpd_values[-1]:.4f}, RMSE: {rmse_values[-1]:.4f}, Regret: {regret_values[-1]:.4f}")

    return {
        "final_train_data": data_dict,
        "chosen_indices": chosen_indices,
        "ei_values": ei_values,
        "nlpd_values": nlpd_values,
        "rmse_values": rmse_values,
        "regret_values": regret_values,
        "surfaces_optimized": surfaces_to_optimize,
        "pred_mean_history": pred_mean_history,
        "pred_var_history": pred_var_history
    }
