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


def mnist_regvar_sketching(
        positive_class={0, 2, 4, 6, 8},
        num_pretrain_epochs=100,
        num_finetune_epochs=10,
        lr=0.001,
        right_sketch_size=50,
        max_samples=1000,
        f_lambda_in=[0.05, .2],
        save_results=True,
        verbose=False
):
    """
    Train a FlexibleMLP model on MNIST and evaluate right-only sketching performance.
    """
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
            'name': '2_layer_32_32_mnist',
            'hidden_dims': [32, 32],
            'dropout_rate': 0.0,
            'batch_norm': False
        }
    ]
    # Create binary MNIST dataloaders
    print("Setting up binary MNIST classification...")
    train_loader, test_loader = create_binary_mnist_dataloaders(
        batch_size=128, 
        positive_class=positive_class
    )
    model = FlexibleMLP(input_dim=784, 
                        output_dim=1, 
                        hidden_dims=model_configs[0]['hidden_dims'], 
                        dropout_rate=model_configs[0]['dropout_rate'], 
                        batch_norm=model_configs[0]['batch_norm'])
    print(f"model size is {sum(p.numel() for p in model.parameters())}")
    
    # Create model checkpoint filename based on configuration
    model_filename = f"{base_path}/trained_models/{model_configs[0]['name']}_positive_class_{positive_class}_epochs_{num_pretrain_epochs}_lr_{lr}.pth"
    
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
        
        print(f"Model loaded successfully! Training was completed with {len(train_losses)} epochs.")
    else:
        print("Training new model...")
        pretrained_model, train_losses, test_accuracies, test_precisions, test_recalls, test_f1_scores = train_binary_model(model, train_loader, test_loader, num_epochs=num_pretrain_epochs, lr=lr)
        
        # Save the trained model and metrics
        print(f"Saving model to {model_filename}...")
        checkpoint = {
            'model_state_dict': pretrained_model.state_dict(),
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'test_precisions': test_precisions,
            'test_recalls': test_recalls,
            'test_f1_scores': test_f1_scores,
            'model_config': model_configs[0],
            'training_params': {
                'num_epochs': num_pretrain_epochs,
                'lr': lr,
                'positive_class': positive_class
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
    print(f"   Model parameters: {pretrained_model.get_num_parameters():,}")
    # Plot training results

    if not os.path.exists(os.path.join(base_path, "training_results")):
        os.makedirs(os.path.join(base_path, "training_results"))
    
    # Only plot if we have training metrics (i.e., model was trained, not loaded)
    if train_losses and test_accuracies:
        plot_training_results(
            train_losses, 
            test_accuracies, 
            filename=f"{base_path}/training_results/{model_configs[0]['name']}_positive_class_{positive_class}_epochs_{num_pretrain_epochs}_lr_{lr}_training_results.pdf"
        )
    else:
        print("Skipping training plots - model was loaded from checkpoint.")

    for name, param in pretrained_model.named_parameters():    
        print(f"{name}: {param.data.norm(p=2, dim=0).mean():.6f} (BEFORE finetuning)")
    hessian_inv_right_sketch, _, right_sketch_matrix = sketch_regvar_right(
        pretrained_model, 
        train_loader,
        right_sketch_size=right_sketch_size,
        f_lambda_vec=f_lambda_in,
        num_finetune_epochs=num_finetune_epochs,
        device=device,
        verbose=verbose
    )

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

    return results_dict


def main():
    mnist_regvar_sketching(
        positive_class={0, 2, 4, 6, 8},
        num_pretrain_epochs=100,
        num_finetune_epochs=20,
        lr=0.001,
        right_sketch_size=400,
        max_samples=200,
        save_results=True,
        verbose=False,
        f_lambda_in=[0.05, .2]
    )


if __name__ == "__main__":
    main()
