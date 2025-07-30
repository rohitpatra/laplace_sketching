from math import sqrt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import gzip
import os
from utils import FlexibleMLP, generate_real_data_gradients, compute_estimated_variance_right_only, compute_estimated_variance
from mnist_example_simple import train_binary_model, create_binary_mnist_dataloaders, evaluate_binary_model,plot_training_losses, compute_hessian_binary
from finetune_script import sketch_regvar_right, sketch_regvar_left, variance_estimate_regvar, finetune_model_regvar, plot_finetune_losses
from typing import Dict, List


""" Tests that the right sketch is stable, by comparing H^-1 u_1 and H^-1 u_2 with H^-1 (u_1 + u_2)
"""

positive_class={0, 2, 4, 6, 8}
num_pretrain_epochs=7000
num_finetune_epochs=100
lr=0.0001
finetune_lr=0.0001
right_sketch_size=2
max_samples=1000
f_lambda_in=[0.05, .2]
save_results=True
verbose=True
prior_scale=0.0005
base_path = "/shared/public/sharing/laplace-sketching"
if not os.path.exists(os.path.join(base_path, "trained_models")):
    os.makedirs(os.path.join(base_path, "trained_models"))
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define model architecture
model_configs = [
    {
        'name': '2_layer_8_16_tanh_mnist',
        'hidden_dims': [8, 16],
        'dropout_rate': 0.0,
        'batch_norm': False,
        'activation': 'tanh'
    }
]
# Create binary MNIST dataloaders
print("Setting up binary MNIST classification...")
train_loader, test_loader = create_binary_mnist_dataloaders(
    batch_size=60000, 
    positive_class=positive_class
)
model = FlexibleMLP(input_dim=784, 
                    output_dim=1, 
                    hidden_dims=model_configs[0]['hidden_dims'], 
                    dropout_rate=model_configs[0]['dropout_rate'], 
                    batch_norm=model_configs[0]['batch_norm'],
                    activation=model_configs[0]['activation'])
print(f"model size is {sum(p.numel() for p in model.parameters())}")

# Create model checkpoint filename based on configuration
model_filename = f"{base_path}/trained_models/{model_configs[0]['name']}_positive_class_{positive_class}_epochs_{num_pretrain_epochs}_lr_{lr}_prior_scale_{prior_scale}_activation_{model_configs[0]['activation']}.pth"

# Check if model already exists
if os.path.exists(model_filename):
    print(f"Loading existing model from {model_filename}...")
    checkpoint = torch.load(model_filename, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    pretrained_model = model.to(device)
    
    # Load training metrics if available
    train_losses = checkpoint.get('train_losses', [])
    test_accuracies = checkpoint.get('test_accuracies', [])
    test_precisions = checkpoint.get('test_precisions', [])
    test_recalls = checkpoint.get('test_recalls', [])
    test_f1_scores = checkpoint.get('test_f1_scores', [])
    test_losses = checkpoint.get('test_losses', [])
    train_l2 = checkpoint.get('train_l2', [])
    batch_grad_norms = checkpoint.get('batch_grad_norms', [])
    epoch_grad_norms = checkpoint.get('epoch_grad_norms', [])
    print(f"Model loaded successfully! Training was completed with {len(train_losses)} epochs.")
else:
    print("Training new model...")
    pretrained_model, train_losses, test_accuracies, test_precisions, test_recalls, test_f1_scores, test_losses, train_l2, batch_grad_norms, epoch_grad_norms = train_binary_model(
        model,
        train_loader,
        test_loader,
        num_epochs=num_pretrain_epochs,
        lr=lr,
        device=device,  # ensure training happens on the chosen device
        track_grad_norm=True,
        best_model_path=f"best_model_positive_class_{positive_class}_epochs_{num_pretrain_epochs}_lr_{lr}.pt",
        prior_scale=prior_scale,
    )
    plot_training_losses(train_losses, test_losses, test_accuracies, epoch_grad_norms, train_l2, filename=f"Training_summary_{model_configs[0]['name']}_positive_class_{positive_class}_epochs_{num_pretrain_epochs}_lr_{lr}_prior_scale_{prior_scale}_activation_{model_configs[0]['activation']}.pdf")   
    # Save the trained model and metrics
    print(f"Saving model to {model_filename}...")
    checkpoint = {
        'model_state_dict': pretrained_model.state_dict(),
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'test_precisions': test_precisions,
        'test_recalls': test_recalls,
        'test_f1_scores': test_f1_scores,
        'test_losses': test_losses,
        'train_l2': train_l2,
        'batch_grad_norms': batch_grad_norms,
        'epoch_grad_norms': epoch_grad_norms,
        'model_config': model_configs[0],
        'training_params': {
            'num_epochs': num_pretrain_epochs,
            'lr': lr,
            'positive_class': positive_class,
            'prior_scale': prior_scale
        }
    }
    torch.save(checkpoint, model_filename)
    print("Model saved successfully!")
    # Final evaluation of the best model

# list of ratios of train_l2 and epoch_grad_norms
l2_grad_norm_ratio = [train_l2[i] / epoch_grad_norms[i]**2 for i in range(len(train_l2))]
print(f"l2_grad_norm_ratio: {l2_grad_norm_ratio}")

# find the best epoch
best_epoch = np.argmin(test_losses)
print(f"Best epoch: {best_epoch}, best train loss: {train_losses[best_epoch]:.4f}, best test loss: {test_losses[best_epoch]:.4f}, best test accuracy: {test_accuracies[best_epoch]:.2f}, best test precision: {test_precisions[best_epoch]:.2f}, best test recall: {test_recalls[best_epoch]:.2f}, best test f1 score: {test_f1_scores[best_epoch]:.2f}")


# ---- prepare the single data point --------------------------------------
first_x, _ = test_loader.dataset[0]          # Tensor shape [784]
first_x      = first_x.unsqueeze(0).to(device)  # shape [1,784] so model(x) works

# ---- choose your λ grid --------------------------------------------------
f_lambda_vec = [1e-6, 1e-3, 1e-1]              # or any list you like

# ---- run the RegVar finetuning --------------------
orig_loss = nn.BCEWithLogitsLoss(reduction='sum')
f_lambda = 1e-4
finetuned_model, losses, val_losses, penalty_values, loss_main, val_loss_main, best_epoch = finetune_model_regvar(
    pretrained_model,
    u_eval_in=None, 
    data_in=first_x,
    finetune_data_loader=train_loader,
    finetune_lambda=f_lambda,
    finetune_lr=finetune_lr,
    num_epochs=num_finetune_epochs,
    device=device,
    verbose=True,
    save_best_model=False,
    save_path=f"best_finetune_model_lambda_{f_lambda}.pt",
    val_loader=None,
    prior_scale=prior_scale
)

# look at the diff between the finetuned model and the pretrained model
flat_params_pretrained = torch.nn.utils.parameters_to_vector(pretrained_model.parameters()).detach().clone().to(device)
flat_params_finetuned = torch.nn.utils.parameters_to_vector(finetuned_model.parameters()).detach().clone().to(device)
diff = flat_params_finetuned - flat_params_pretrained
print(f"diff between finetuned and pretrained model when finetuned with lambda {f_lambda} for {num_finetune_epochs} epochs: {diff.norm()}")


# diff between finetuned and pretrained model when finetuned with lambda 0.0001 for 1000 epochs: 0.9403998851776123

num_finetune_epochs = 500
finetuned_model_500, losses_500, val_losses_500, penalty_values_500, loss_main_500, val_loss_main_500, best_epoch_500 = finetune_model_regvar(pretrained_model, u_eval_in=None, data_in=first_x, finetune_data_loader=train_loader, finetune_lambda=f_lambda, finetune_lr=finetune_lr, num_epochs=num_finetune_epochs, device=device, verbose=True, save_best_model=False, save_path=f"best_finetune_model_lambda_{f_lambda}_e500.pt", val_loader=None)

# look at the diff between the finetuned model and the pretrained model
flat_params_pretrained = torch.nn.utils.parameters_to_vector(pretrained_model.parameters()).detach().clone().to(device)
flat_params_finetuned = torch.nn.utils.parameters_to_vector(finetuned_model_500.parameters()).detach().clone().to(device)
diff = flat_params_finetuned - flat_params_pretrained
print(f"diff between finetuned and pretrained model when finetuned with lambda {f_lambda} for {num_finetune_epochs} epochs: {diff.norm()}")

print(flat_params_pretrained.norm(), flat_params_finetuned.norm())


(finetuned_model_500(first_x) - pretrained_model(first_x))/ f_lambda

(finetuned_model(first_x) - pretrained_model(first_x))/ f_lambda 


h_inv, flat_p, H, Hoff = compute_hessian_binary(
    pretrained_model,
    train_loader,
    device,
    diagonal_offset=0.0,
    prior_scale=prior_scale,
    profile=True,      # optional profiling
)

def grad_func_local(flat_params, new_inputs):
    param_dict = {}
    pointer = 0
    for name, param in pretrained_model.named_parameters():
        num_param = param.numel()
        param_dict[name] = flat_params[pointer:pointer + num_param].view_as(param)
        pointer += num_param
    outputs = torch.func.functional_call(pretrained_model, param_dict, new_inputs)
    return outputs
new_gradient = torch.autograd.functional.jacobian(
    lambda fp: grad_func_local(fp, first_x),
    flat_params
).squeeze(1)
oracle_var = torch.einsum('bi,ij,bj->b', new_gradient, hessian_inv, new_gradient)




# appendix


# ---- run the RegVar finetuning & variance estimation --------------------
h_inv_u, regvar_est, finetuned = variance_estimate_regvar(
    input_model       = pretrained_model,
    finetune_loader   = train_loader,         # or a smaller “finetune_loader” if you have one
    u_eval            = None,                 # <- we are in data_based mode
    input_data        = first_x,              # <- THIS is the first sample
    f_lambda_vec      = f_lambda_vec,
    num_finetune_epochs = num_finetune_epochs,
    device            = device,
    verbose           = True,
    method            = "data_based",         # <- crucial!
    finetune_lr       = finetune_lr,
)

# ---- regvar_est now contains your estimated variances -------------------
for lam in f_lambda_vec:
    print(f"λ={lam:>.4g}  RegVar estimate = {regvar_est[lam].item():.6f}")

with torch.no_grad():
    # u = ∇_θ model(first_x)  (one-liner with autograd)
    u = torch.autograd.grad(model_output := pretrained_model(first_x).sum(),
                            pretrained_model.parameters(), create_graph=False)
    u = torch.cat([g.flatten() for g in u])

    var_scalar = torch.dot(u, h_inv_u[lam])

