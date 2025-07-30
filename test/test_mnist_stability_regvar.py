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
from mnist_example_simple import train_binary_model, create_binary_mnist_dataloaders, evaluate_binary_model,plot_training_results
from finetune_script import sketch_regvar_right, sketch_regvar_left, variance_estimate_regvar, finetune_model_regvar, plot_finetune_losses
from typing import Dict, List


""" Tests that the right sketch is stable, by comparing H^-1 u_1 and H^-1 u_2 with H^-1 (u_1 + u_2)
"""

positive_class={0, 2, 4, 6, 8}
num_pretrain_epochs=6000
num_finetune_epochs=1000
lr=0.0001
finetune_lr=0.0001
right_sketch_size=2
max_samples=1000
f_lambda_in=[0.05, .2]
save_results=True
verbose=True
prior_scale=0.01
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
        'name': '2_layer_8_16_mnist',
        'hidden_dims': [8, 16],
        'dropout_rate': 0.0,
        'batch_norm': False
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
                    batch_norm=model_configs[0]['batch_norm'])
print(f"model size is {sum(p.numel() for p in model.parameters())}")

# Create model checkpoint filename based on configuration
model_filename = f"{base_path}/trained_models/{model_configs[0]['name']}_positive_class_{positive_class}_epochs_{num_pretrain_epochs}_lr_{lr}_prior_scale_{prior_scale}.pth"


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
    pretrained_model, train_losses, test_accuracies, test_precisions, test_recalls, test_f1_scores, test_losses, train_l2, batch_grad_norms, epoch_grad_norms = train_binary_model(model, train_loader, test_loader, num_epochs=num_pretrain_epochs, lr=lr, track_grad_norm=True, best_model_path=f"best_model_positive_class_{positive_class}_epochs_{num_pretrain_epochs}_lr_{lr}.pt", prior_scale=prior_scale)
    plot_training_losses(train_losses, test_losses, test_accuracies, epoch_grad_norms, filename=f"Training_summary_{model_configs[0]['name']}_positive_class_{positive_class}_epochs_{num_pretrain_epochs}_lr_{lr}_prior_scale_{prior_scale}.pdf")    
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

final_metrics = evaluate_binary_model(pretrained_model, test_loader, device)
print(f"\n Final Results for {model_configs[0]['name']}:")
print(f"   Accuracy: {final_metrics['accuracy']:.2f}%")
print(f"   Precision: {final_metrics['precision']:.2f}%")
print(f"   Recall: {final_metrics['recall']:.2f}%")
print(f"   F1 Score: {final_metrics['f1']:.2f}%")
print(f"   Test Loss: {final_metrics['loss']:.4f}")
print(f"   Model parameters: {pretrained_model.get_num_parameters():,}")
# Plot training results

# find the best epoch
best_epoch = np.argmin(test_losses)
print(f"Best epoch: {best_epoch}")
print(f"Best test loss: {test_losses[best_epoch]:.4f}")
print(f"Best test accuracy: {test_accuracies[best_epoch]:.2f}%")
print(f"Best test precision: {test_precisions[best_epoch]:.2f}%")
print(f"Best test recall: {test_recalls[best_epoch]:.2f}%")
print(f"Best test f1 score: {test_f1_scores[best_epoch]:.2f}%")
print(f"Best epoch grad norm: {epoch_grad_norms[best_epoch]:.4f}")


if not os.path.exists(os.path.join(base_path, "training_results")):
    os.makedirs(os.path.join(base_path, "training_results"))

# Only plot if we have training metrics (i.e., model was trained, not loaded)
if train_losses and test_accuracies and test_losses and epoch_grad_norms:
    plot_training_losses(train_losses, test_losses, test_accuracies, epoch_grad_norms, 
        filename=f"{base_path}/training_results/{model_configs[0]['name']}_positive_class_{positive_class}_epochs_{num_pretrain_epochs}_lr_{lr}_prior_scale_{prior_scale}_training_results.pdf"
    )
else:
    print("Skipping training plots - model was loaded from checkpoint.")

for name, param in pretrained_model.named_parameters():    
    print(f"{name}: {param.data.norm(p=2, dim=0).mean():.6f} (BEFORE finetuning)")

num_params = sum(p.numel() for p in pretrained_model.parameters())
right_sketch_matrix = torch.randn(2, num_params, device=device) / np.sqrt(right_sketch_size)
right_sketch_matrix = right_sketch_matrix.float()
right_sketch_matrix.shape
## add a third row to right_sketch_matrix that is the sum of the first two rows
sum_row = right_sketch_matrix[:2].sum(dim=0, keepdim=True)
right_sketch_matrix = torch.cat([right_sketch_matrix, sum_row], dim=0)
right_sketch_matrix.shape

f_lambda_in = [0.05]
hessian_inv_u_by_lambda: Dict[float, List[torch.Tensor]] = {f_lambda: [] for f_lambda in f_lambda_in}
model_by_lambda: Dict[float, List[nn.Module]] = {f_lambda: [] for f_lambda in f_lambda_in}
# for sr in range(right_sketch_matrix.shape[0]):
#     print(f"Processing right sketch vector {sr+1}/{right_sketch_size}")
#     # Use the flattened sketch vector directly - no conversion needed!
u_eval_vector = right_sketch_matrix[1, :]
finetuned_model_dict = {}
f_lambda = 5e-5
# for f_lambda in f_lambda_in:
if verbose:
    for name, param in pretrained_model.named_parameters():
        print(f"{name}: {param.data.norm(p=2, dim=0).mean():.6f} (BEFORE finetuning with lambda={f_lambda})")

finetuned_model, losses, val_losses, penalty_values, loss_values = finetune_model_regvar(
    input_model=pretrained_model,
    u_eval_in=u_eval_vector,
    data_in=None,
    finetune_data_loader=train_loader,
    finetune_lambda=f_lambda,
    finetune_lr=finetune_lr,
    num_epochs=num_finetune_epochs,
    device=device,
    verbose=verbose,
    save_best_model=True,
    val_loader=test_loader
)

if verbose:
    for name, param in finetuned_model.named_parameters():
        print(f"{name}: {param.data.norm(p=2, dim=0).mean():.6f} (AFTER finetuning with lambda={f_lambda})")
finetuned_model_dict[f_lambda] = finetuned_model
# variance estimates at input_data
plot_four_losses(losses, val_losses, penalty_values, loss_values, filename=f"{model_configs[0]['name']}_positive_class_{positive_class}_epochs_{num_pretrain_epochs}_lr_{lr}_finetune_epochs_{num_finetune_epochs}_lr_{finetune_lr}_finetune_losses.pdf")


hessian_inv_u = {}
    for f_lambda, model in finetuned_model_dict.items():
        flat_params_pretrained = torch.nn.utils.parameters_to_vector(pretrained_model.parameters()).detach().clone()
        flat_params_finetuned = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        hessian_inv_u[f_lambda] = -(flat_params_finetuned - flat_params_pretrained) / f_lambda  ## TODO: What is the prior here? See James's paper
    for f_lambda, model in finetuned_model_dict.items():
        hessian_inv_u_by_lambda[f_lambda].append(hessian_inv_u[f_lambda])
        model_by_lambda[f_lambda].append(model)
        print(f"Finetuned model for f_lambda {f_lambda} has {sum(p.numel() for p in model.parameters())} parameters")

hessian_inv_right_sketch = {f_lambda: torch.stack(hessian_inv_u_by_lambda[f_lambda], dim=1) for f_lambda in f_lambda_in}

hessian_inv_u_1 = hessian_inv_right_sketch[f_lambda_in[0]][:, 0]
hessian_inv_u_2 = hessian_inv_right_sketch[f_lambda_in[0]][:, 1]
hessian_inv_u_sum = hessian_inv_right_sketch[f_lambda_in[0]][:, 2]
# Compute and report the relative error for the linearity of H^{-1}
error_vec = (hessian_inv_u_1 + hessian_inv_u_2) - hessian_inv_u_sum
error_norm = torch.norm(error_vec, p=2)
sum_norm = torch.norm(hessian_inv_u_sum, p=2)
# Add a small epsilon to avoid division by zero
rel_error = error_norm / (sum_norm + 1e-12)
print(f"Relative error H^-1(u₁)+H^-1(u₂) ≈ H^-1(u₁+u₂): {rel_error:.6f}")

# hessian_inv_right_sketch, _, right_sketch_matrix = sketch_regvar_right(
#     pretrained_model, 
#     train_loader,
#     right_sketch_size=right_sketch_size,
#     f_lambda_vec=f_lambda_in,
#     num_finetune_epochs=num_finetune_epochs,
#     device=device,
#     verbose=verbose
# )

# # Compute Hessian of the best model on the training set
# hessian_matrix_offset_inv, flat_params, hessian_matrix, hessian_matrix_offset = compute_hessian_binary(model, train_loader, device, diagonal_offset = diagonal_offset)
# test gradients
flat_params = torch.nn.utils.parameters_to_vector(pretrained_model.parameters()).detach().clone()
num_params = flat_params.shape[0]
test_gradients, test_inputs = generate_real_data_gradients(pretrained_model, flat_params, test_loader, device, max_samples=max_samples)
# test_gradients is [max_samples, num_params]
# compute oracle regvar variance
oracle_vars = {f_lambda: torch.zeros(test_gradients.shape[0], device=device) for f_lambda in f_lambda_in}
for i in range(test_gradients.shape[0]):
    if i % 100 == 0:
        print(f"Computed oracle variance for sample {i} out of {test_gradients.shape[0]}")
    # Use the test gradient for this sample as u_eval
    u_eval_vector = test_gradients[i, :]
    _, oracle_regvar_dict, _ = variance_estimate_regvar(
        input_model=pretrained_model,
        finetune_loader=train_loader,
        u_eval=u_eval_vector,
        input_data= test_inputs[i, :],
        f_lambda_vec=f_lambda_in,
        num_finetune_epochs=num_finetune_epochs,
        device=device,
        verbose=verbose,
        method="u_based"
    )
    # Populate oracle_vars dictionary with the variance estimates for each lambda
    for f_lambda in f_lambda_in:
        oracle_vars[f_lambda][i] = oracle_regvar_dict[f_lambda].squeeze()
right_variance_estimates = {}
for f_lambda in f_lambda_in:
    right_variance_estimates[f_lambda] = compute_estimated_variance_right_only(test_gradients, hessian_inv_right_sketch[f_lambda], right_sketch_matrix)

left_sketch_vec = np.concatenate([
    np.linspace(1, min(10, num_params), 10),
    np.linspace(max(11, min(10, num_params) + 1), min(50, num_params), 11) if num_params > 10 else [],
    np.linspace(max(51, min(50, num_params) + 1), min(num_params//2, num_params), 10) if num_params > 50 else []
])
left_sketch_vec = np.unique(left_sketch_vec.astype(int))  # Remove duplicates and convert to integers
both_variance_estimates = {sketch_size: {f_lambda: {} for f_lambda in f_lambda_in} for sketch_size in left_sketch_vec}
for left_sketch_size in left_sketch_vec:
    # Generate left-side sketch for the current size
    Q_mat_dict, left_sketch_matrix = sketch_regvar_left(
        left_sketch_size=int(left_sketch_size), # Ensure it's an integer
        hessian_inv_right_sketch=hessian_inv_right_sketch,
        num_params=flat_params.shape[0],
        device=device
    )
    # Fill in the combined (left + right) variance estimates
    for f_lambda in f_lambda_in:
        both_variance_estimates[left_sketch_size][f_lambda] = compute_estimated_variance(
            test_gradients, left_sketch_matrix, Q_mat_dict[f_lambda], right_sketch_matrix
        )

approximation_errors_right_only = {}
approximation_errors_both = {}
summary_stats_right_only = {}
summary_stats_both = {}
for f_lambda in f_lambda_in:
    approximation_errors_right_only[f_lambda] = {
        'absolute': right_variance_estimates[f_lambda] - oracle_vars[f_lambda],
        'relative': (right_variance_estimates[f_lambda] - oracle_vars[f_lambda]) / oracle_vars[f_lambda],
    }
    approximation_errors_both[f_lambda] = {
        'absolute': {sketch_size: both_variance_estimates[sketch_size][f_lambda] - oracle_vars[f_lambda] for sketch_size in left_sketch_vec},
        'relative': {sketch_size: (both_variance_estimates[sketch_size][f_lambda] - oracle_vars[f_lambda]) / oracle_vars[f_lambda] for sketch_size in left_sketch_vec}
    }
    summary_stats_right_only[f_lambda] = {
        'absolute': {
            'mean': approximation_errors_right_only[f_lambda]['absolute'].mean().item(),
            'std': approximation_errors_right_only[f_lambda]['absolute'].std().item(),
        },
        'relative': {
            'mean': approximation_errors_right_only[f_lambda]['relative'].mean().item(),
            'std': approximation_errors_right_only[f_lambda]['relative'].std().item(),
        }
    }
    summary_stats_both[f_lambda] = {
        'absolute': {
            'mean': {sketch_size: approximation_errors_both[f_lambda]['absolute'][sketch_size].mean().item() for sketch_size in left_sketch_vec},
            'std': {sketch_size: approximation_errors_both[f_lambda]['absolute'][sketch_size].std().item() for sketch_size in left_sketch_vec},
        },
        'relative': {
            'mean': {sketch_size: approximation_errors_both[f_lambda]['relative'][sketch_size].mean().item() for sketch_size in left_sketch_vec},
            'std': {sketch_size: approximation_errors_both[f_lambda]['relative'][sketch_size].std().item() for sketch_size in left_sketch_vec},
        }
    }

results_dict = {}
results_dict[model_configs[0]['name']] = {
    'num_params': num_params,
    'left_sketch_vec': left_sketch_vec,
    'right_sketch_value': right_sketch_matrix.shape[0],
    'f_lambda_in': f_lambda_in,
    'both_variance_estimates': both_variance_estimates,
    'right_variance_estimates': right_variance_estimates,
    'oracle_vars': oracle_vars,
    'approximation_errors_right_only': approximation_errors_right_only,
    'approximation_errors_both': approximation_errors_both,
    'oracle_vars_stats': {
        'mean': {f_lambda: oracle_vars[f_lambda].mean().item() for f_lambda in f_lambda_in},
        'std': {f_lambda: oracle_vars[f_lambda].std().item() for f_lambda in f_lambda_in},
        'min': {f_lambda: oracle_vars[f_lambda].min().item() for f_lambda in f_lambda_in},
        'max': {f_lambda: oracle_vars[f_lambda].max().item() for f_lambda in f_lambda_in}
    },
    'summary_stats_right_only': summary_stats_right_only,
    'summary_stats_both': summary_stats_both,
    'training_info': {
        'final_train_loss': train_losses if train_losses else None,
        'final_test_accuracy': test_accuracies if test_accuracies else None,
        'num_epochs': num_pretrain_epochs,
    }
}
print(f"Results for {model_configs[0]['name']}:")
print(f"  Parameters: {num_params}")
for f_lambda in f_lambda_in:
    print(f"  Right-only rel errors for f_lambda {f_lambda}: {summary_stats_right_only[f_lambda]['relative']['mean']}")
    print(f"  Both rel errors for f_lambda {f_lambda}: {summary_stats_both[f_lambda]['relative']['mean']}")


if save_results:
    if not os.path.exists(os.path.join(base_path, "results")):
        os.makedirs(os.path.join(base_path, "results"))
    results_file = f"{base_path}/results/{model_configs[0]['name']}_positive_class_{positive_class}_pretrain_epochs_{num_pretrain_epochs}_finetune_epochs_{num_finetune_epochs}_lr_{lr}_right_sketch_size_{right_sketch_size}_max_samples_{max_samples}_f_lambda_in_{f_lambda_in}.pkl"
    print(f"\nSaving results to {results_file}...")
    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f)
    print("Results saved!")
