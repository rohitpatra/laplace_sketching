
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
from finetune_script import sketch_regvar_right, sketch_regvar_left, variance_estimate_regvar

input_dim = 2
output_dim = 1
f_lambda_in = [0.2]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("=== Model Finetuning Script ===")
# 1. Get a pretrained model (example: train a simple model first)
print("1. Getting pretrained model...")
approximator = LaplaceSketchApproximator(input_dim=input_dim, output_dim=output_dim)
train_loader, val_loader, data_splits = approximator.generate_data(num_samples=50000, batch_size=256)
pretrained_model = FittedDeepNet(input_dim=input_dim, output_dim=output_dim)
# pretrained_model = FittedLinear(input_dim=input_dim, output_dim=output_dim)
pretrained_model, _, _ = approximator.train_model(
    pretrained_model, train_loader, val_loader, data_splits[3], num_epochs=50
)
num_params = sum(p.numel() for p in pretrained_model.parameters())
print(f"Pretrained model has {num_params} parameters")
# create data for evaluation and finetuning
finetune_loader, _, _ = approximator.generate_data(num_samples=2000, batch_size=128)
print(f"Pretrained model has {sum(p.numel() for p in pretrained_model.parameters())} parameters")
# Print mean L2 norm of all weights and biases BEFORE finetuning
for name, param in pretrained_model.named_parameters():
    print(f"{name}: {param.data.norm(p=2, dim=0).mean():.6f} (BEFORE finetuning)")

hessian_inv_right_sketch, model_by_lambda, right_sketch_matrix = sketch_regvar_right(
    pretrained_model, 
    finetune_loader,
    right_sketch_size=30,
    f_lambda_vec=f_lambda_in,
    num_finetune_epochs=25,
    device=device,
    verbose=False
)
flat_params = torch.nn.utils.parameters_to_vector(pretrained_model.parameters()).detach().clone()
test_gradients, _ = generate_test_gradients_functorch(pretrained_model, input_dim, device)
right_variance_estimates = {}
both_variance_estimates = {}
Q_mat_dict, left_sketch_matrix = sketch_regvar_left(left_sketch_size=30, hessian_inv_right_sketch=hessian_inv_right_sketch, num_params=num_params, device=device)


for f_lambda in f_lambda_in:
    right_variance_estimates[f_lambda] = compute_estimated_variance_right_only(test_gradients, hessian_inv_right_sketch[f_lambda], right_sketch_matrix)
    both_variance_estimates[f_lambda] = compute_estimated_variance(test_gradients, left_sketch_matrix, Q_mat_dict[f_lambda], right_sketch_matrix)


# summary of ratios of the two methods
ratios = {}
for f_lambda in f_lambda_in:
    ratios[f_lambda] = right_variance_estimates[f_lambda] / both_variance_estimates[f_lambda]
    print(f"Mean Ratio for lambda={f_lambda}: {ratios[f_lambda].mean():.2f}")
    print(f"Std Ratio for lambda={f_lambda}: {ratios[f_lambda].std():.2f}")





print(f"Hessian inverse right sketch shape: {hessian_inv_right_sketch[0.005].shape}")
print(f"Right sketch matrix shape: {right_sketch_matrix.shape}")
print(f"Model by lambda shape: {model_by_lambda[0.005][0].shape}")

# Compute Hessian classically
hessian_inv, flat_params, _, _ = approximator.compute_hessian(pretrained_model, val_loader)
print(f"Hessian stable rank: {compute_stable_rank(hessian_inv):.2f}")


f_lambda_in = [0.005, 0.01, 0.02, 0.05, 0.1, .2]
ratios_u_based: Dict[float, List[float]] = {f_lambda: [] for f_lambda in f_lambda_in}
ratios_data_based: Dict[float, List[float]] = {f_lambda: [] for f_lambda in f_lambda_in}

num_rep = 1000
for i in range(num_rep):
    if i % 20 == 0:
        print(f"At iteration {i} out of {num_rep}")
    input_data = np.random.uniform(low=-2, high=2, size=(1, input_dim))
    input_data = torch.from_numpy(input_data).type(torch.float32).to(device)
    # Create variance estimates from hessian_computation
    _, regvar_est_u_based = variance_estimate_regvar(
        input_model=pretrained_model,
        finetune_loader=finetune_loader,
        u_eval=None,  # Set to None to compute gradients directly
        input_data=input_data,
        f_lambda_vec=f_lambda_in,
        num_finetune_epochs=num_finetune_epochs,
        device=device,
        verbose=False,
        method="u_based"
    )
    _, regvar_est_data_based = variance_estimate_regvar(
        input_model=pretrained_model,
        finetune_loader=finetune_loader,
        u_eval=None,  # Set to None to compute gradients directly
        input_data=input_data,
        f_lambda_vec=f_lambda_in,
        num_finetune_epochs=num_finetune_epochs,
        device=device,
        verbose=False,
        method="data_based"
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
        lambda fp: grad_func_local(fp, input_data),
        flat_params
    ).squeeze(1)
    oracle_var = torch.einsum('bi,ij,bj->b', new_gradient, hessian_inv, new_gradient)
    # Compare oracle_var with regvar_est
    for f_lambda, regvar in regvar_est_u_based.items():
        ratios_u_based[f_lambda].append(
            regvar.detach().cpu().item() / oracle_var.detach().cpu().item()
        )
    for f_lambda, regvar in regvar_est_data_based.items():
        ratios_data_based[f_lambda].append(
            regvar.detach().cpu().item() / oracle_var.detach().cpu().item()
        )
# Histogram of the ratios loop over f_lambda_in in a grid and save to pdf
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Top plot: u_based method
for f_lambda in f_lambda_in:
    ax1.hist(ratios_u_based[f_lambda], bins=20, alpha=0.7, label=f'lambda={f_lambda}')

ax1.set_xlabel('Ratio')
ax1.set_ylabel('Frequency')
ax1.set_title('U-based Method: Distribution of Ratios for Different Lambda Values')
ax1.legend()

# Bottom plot: data_based method
for f_lambda in f_lambda_in:
    ax2.hist(ratios_data_based[f_lambda], bins=20, alpha=0.7, label=f'lambda={f_lambda}')

ax2.set_xlabel('Ratio')
ax2.set_ylabel('Frequency')
ax2.set_title('Data-based Method: Distribution of Ratios for Different Lambda Values')
ax2.legend()

plt.tight_layout()
file_name = f"ratio_hist_params{num_params}_rep{num_rep}_synthetic_large_data_and_u_based.pdf"
plt.savefig(file_name)
plt.show()
