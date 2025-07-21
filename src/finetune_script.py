#!/usr/bin/env python3
"""
Compact script for finetuning a trained model with different loss and data.
"""

import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, List

def create_finetune_data(num_samples=1000, input_dim=20, output_dim=1, noise_level=0.1):
    """Create new data for finetuning with different distribution."""
    # Different data distribution for finetuning
    X = np.random.normal(0, 2, (num_samples, input_dim))  # Different std
    
    # Different target function (e.g., polynomial instead of linear)
    y = np.sum(X**2, axis=1, keepdims=True) + noise_level * np.random.randn(num_samples, 1)
    
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    
    return X_tensor, y_tensor
## TODO: lets ensure we pick the best model here and look at the loss curves for some examples/models
def finetune_model_regvar(input_model, u_eval_in, data_in, finetune_data_loader, orig_loss, finetune_lambda=1.0, lr=1e-4, num_epochs=20, device='cpu', verbose=False, save_best_model=False, save_path='best_finetune_model.pt', val_loader=None):
    """
    Finetune a pretrained model with new loss and data at given u_eval_in OR data_in at lambda =  finetune_lambda.
    
    Args:
        input_model: Pre-trained model
        u_eval_in: Flattened vector of gradients for regvar penalty, if not none, penalty for finetuning will be finetune_lambda * theta^T u_eval_in. One of u_eval_in or data_in must be provided.
        data_in: if not none, penalty for finetuning will be finetune_lambda * model(data_in).  Only one of u_eval_in OR data_in must be provided.
        finetune_data_loader: DataLoader with finetuning the input_model.
        orig_loss: Loss function to use for finetuning
        finetune_lambda : Regularization parameter for finetuning, Default is 1.0
        lr: Learning rate for finetuning
        num_epochs: Number of finetuning epochs
        device: Device to run on
        save_best_model: Whether to save the best model based on lowest validation loss
        save_path: Path to save the best model
        val_loader: Validation data loader (required if save_best_model=True)
    
    Returns:
        model: Finetuned model
        losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (empty if val_loader is None)
    """
    if data_in is not None and u_eval_in is not None:
        raise ValueError("Either data_in or u_eval_in must be provided, not both")
    if data_in is None and u_eval_in is None:
        raise ValueError("Either data_in or u_eval_in must be provided")
    if save_best_model and val_loader is None:
        raise ValueError("val_loader must be provided when save_best_model=True")
    if data_in is not None and verbose:
        print("Doing the finetuning with penalty of model(data_in)")
    elif verbose:
        print(f"Doing the finetuning with penalty of theta^T u_eval_in for finetune_lambda = {finetune_lambda}")

    # Move penalty-related tensors to the correct device once
    if data_in is not None:
        data_in = data_in.to(device)
    if u_eval_in is not None:
        # Detach because we never want to back-prop through u_eval_in itself
        u_eval_in = u_eval_in.to(device).detach()
        # Precompute slices of u_eval_in matching each parameter's shape for fast dot-product
        u_eval_slices = []
        pointer = 0
        for param in input_model.parameters():
            numel = param.numel()
            u_eval_slices.append(u_eval_in[pointer:pointer + numel].view_as(param))
            pointer += numel
    input_model_local = copy.deepcopy(input_model)
    model = input_model_local.to(device)
    model.train()
    
    # Use lower learning rate for finetuning
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    if verbose:
        print(f"Finetuning for {num_epochs} epochs with lr={lr}")
    # print(f"Using loss function: {orig_loss.__name__}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in finetune_data_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x).squeeze() # now matches the original training step
            if data_in is not None:
                penalty = model(data_in).sum()
            else:
                # Efficient dot-product without re-flattening
                penalty = sum((param * u_slice).sum() for param, u_slice in zip(model.parameters(), u_eval_slices))
            loss = orig_loss(outputs, batch_y) + finetune_lambda * penalty ## TODO: finetune_lambda is not scaling with batch size
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Validation phase (always compute if val_loader is provided)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                    outputs = model(batch_x).squeeze()
                    if data_in is not None:
                        penalty = model(data_in).sum()
                    else:
                        penalty = sum((param * u_slice).sum() for param, u_slice in zip(model.parameters(), u_eval_slices))
                    val_loss += (orig_loss(outputs, batch_y) + finetune_lambda * penalty).item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # Save best model based on lowest validation loss
            if save_best_model and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
            
            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {avg_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        elif verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
    
    # Load best model at the end if saving was enabled
    if save_best_model:
        model.load_state_dict(torch.load(save_path))
    
    if verbose: 
        print(f"Finetuning complete! Final train loss: {losses[-1]:.6f}")
        if val_losses:
            print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    return model, losses, val_losses


def variance_estimate_regvar(input_model, finetune_loader, u_eval, input_data, f_lambda_vec=None, num_finetune_epochs=25, device="cpu", verbose=False, method="u_based"):
    """ Estimate Regvar variance given input_data or u_eval
    """
    if method not in ["u_based", "data_based"]:
        raise ValueError("method must be either u_based or data_based")
    if f_lambda_vec is None:
        f_lambda_vec = [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    # Enable gradient computation for model parameters
    for param in input_model.parameters():
        param.requires_grad_(True)
    # Compute gradients directly with respect to model parameters at input_data 
    if method == "u_based":
        if u_eval is None:
            # Computing \grad_theta model(input_data) to use as u_eval
            input_model.eval()
            output = input_model(input_data)
            # Compute gradients with respect to each parameter and flatten them
            gradients = []
            for param in input_model.parameters():
                grad = torch.autograd.grad(
                    outputs=output.sum(),
                    inputs=param,
                    retain_graph=True,
                    create_graph=False
                )[0]
                gradients.append(grad.detach().flatten())
            u_eval = torch.cat(gradients)
            if verbose:
                print(f"u_eval shape: {u_eval.shape} (direct gradient as flattened vector)")
        data_for_func = None
    elif method == "data_based":
        if u_eval is not None:
            warnings.warn("u_eval must be None when method is data_based, setting u_eval to None")
            u_eval = None
        if input_data is None:
            raise ValueError("input_data must be provided when method is data_based")
        data_for_func = input_data
    # Looping over Hyper parameter choices
    finetuned_model_dict = {}
    for f_lambda in f_lambda_vec:
        if verbose:
            for name, param in input_model.named_parameters():
                print(f"{name}: {param.data.norm(p=2, dim=0).mean():.6f} (BEFORE finetuning with lambda={f_lambda})")
        finetuned_model, _ , _= finetune_model_regvar(
            input_model=input_model,
            u_eval_in=u_eval,
            data_in=data_for_func,
            finetune_data_loader=finetune_loader,
            orig_loss=nn.BCEWithLogitsLoss(),
            finetune_lambda=f_lambda,
            lr=1e-4,
            num_epochs=num_finetune_epochs,
            device=device,
            verbose=verbose
        )
        if verbose:
            for name, param in finetuned_model.named_parameters():
                print(f"{name}: {param.data.norm(p=2, dim=0).mean():.6f} (AFTER finetuning with lambda={f_lambda})")
        finetuned_model_dict[f_lambda] = finetuned_model
    # variance estimates at input_data
    regvar_u_est = {}
    hessian_inv_u = {}
    for f_lambda, model in finetuned_model_dict.items():
        if input_data is not None:
            regvar_u_est[f_lambda] = -(model(input_data) - input_model(input_data))/ f_lambda 
        flat_params_pretrained = torch.nn.utils.parameters_to_vector(input_model.parameters()).detach().clone()
        flat_params_finetuned = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        hessian_inv_u[f_lambda] = -(flat_params_finetuned - flat_params_pretrained) / f_lambda  ## TODO: What is the prior here? See James's paper
    return hessian_inv_u, regvar_u_est, finetuned_model_dict


def sketch_regvar_right(input_model, finetune_loader, right_sketch_size, f_lambda_vec=None, num_finetune_epochs=25, device="cpu", verbose=False):
    """
    Create sketch of the hessian with regvar as the oracle.
    Args:
        input_model: Pre-trained model
        finetune_loader: DataLoader with finetuning the input_model.
        right_sketch_size: Size of the right sketch matrix
        f_lambda_vec: List of lambda values to use for finetuning
        device: Device to run on
        verbose: Whether to print verbose output

    Returns:
        hessian_inv_right_sketch: Dict of [num_params, right_sketch_size] matrices for each lambda in f_lambda_vec
        model_by_lambda: Dict of finetuned models for each lambda in f_lambda_vec
        right_sketch_matrix: [right_sketch_size, num_params] matrix
    """
    if f_lambda_vec is None:
        f_lambda_vec = [0.005, 0.01, 0.02, 0.05, 0.1, .2]
    for param in input_model.parameters():
        param.requires_grad_(True)
    
    # Get number of parameters
    num_params = sum(p.numel() for p in input_model.parameters())
    if verbose:
        print(f"model has {num_params} parameters")
        print(f"creating right sketch matrix of size {right_sketch_size}")
    # Create right sketch matrix as [right_sketch_size, num_params]
    right_sketch_matrix = torch.randn(right_sketch_size, num_params, device=device) / np.sqrt(right_sketch_size)
    right_sketch_matrix = right_sketch_matrix.float()
    # Compute a regularized model for each right sketch vector
    hessian_inv_u_by_lambda: Dict[float, List[torch.Tensor]] = {f_lambda: [] for f_lambda in f_lambda_vec}
    model_by_lambda: Dict[float, List[nn.Module]] = {f_lambda: [] for f_lambda in f_lambda_vec}
    for sr in range(right_sketch_size):
        if sr % int(right_sketch_size / 10) == 0:
            print(f"Processing right sketch vector {sr+1}/{right_sketch_size}")
        # Use the flattened sketch vector directly - no conversion needed!
        u_eval_vector = right_sketch_matrix[sr, :]
        hessian_inv_u, _, finetuned_model_dict = variance_estimate_regvar(
            input_model=input_model,
            finetune_loader=finetune_loader,
            u_eval=u_eval_vector, 
            input_data=None,  # Set to none to use gaussian sketch
            f_lambda_vec=f_lambda_vec,
            num_finetune_epochs=num_finetune_epochs,
            device=device,
            verbose=verbose,
            method="u_based"
        )
        for f_lambda, model in finetuned_model_dict.items():
            hessian_inv_u_by_lambda[f_lambda].append(hessian_inv_u[f_lambda])
            model_by_lambda[f_lambda].append(model)
    hessian_inv_right_sketch = {}
    for f_lambda in f_lambda_vec:
        # Stack the list of tensors into a matrix [num_params, right_sketch_size]
        hessian_inv_right_sketch[f_lambda] = torch.stack(hessian_inv_u_by_lambda[f_lambda], dim=1)
    return hessian_inv_right_sketch, model_by_lambda, right_sketch_matrix


def sketch_regvar_left(left_sketch_size, hessian_inv_right_sketch, num_params, device):
    """ 
    Args:
        left_sketch_size: Size of the left sketch matrix
        hessian_inv_right_sketch: Dict of [num_params, right_sketch_size] matrices for each lambda in f_lambda_vec
        num_params: Number of parameters in the model
        device: Device to run on

    Returns:
        Q_mat: Dict of [left_sketch_size, num_params] matrices for each lambda in f_lambda_vec
        left_sketch_matrix: [left_sketch_size, num_params] matrix
    """
    left_sketch_matrix = torch.randn(left_sketch_size, num_params, device=device) / np.sqrt(left_sketch_size)
    Q_mat = {}
    
    for f_lambda, H_inv_right in hessian_inv_right_sketch.items():
        # Vectorized computation: Q = S_left^T @ H_inv_right
        Q_mat[f_lambda] = torch.matmul(left_sketch_matrix, H_inv_right)
    return Q_mat, left_sketch_matrix