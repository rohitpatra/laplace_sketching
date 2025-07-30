"""
Utility functions for Laplace Sketch Approximation

This module contains additional specialized functions for advanced sketching methods
and analysis tools that complement the main implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from functorch import make_functional_with_buffers, vmap, jacrev



class FlexibleMLP(nn.Module):
    """Flexible Multi-Layer Perceptron with configurable depth for MNIST and other datasets."""
    
    def __init__(self, input_dim=784, output_dim=10, hidden_dims=None, activation='relu', 
                 dropout_rate=0.0, batch_norm=False, use_bias=True):
        """
        Initialize flexible MLP.
        
        Args:
            input_dim (int): Input dimension (784 for MNIST)
            output_dim (int): Output dimension (10 for MNIST classification)
            hidden_dims (list): List of hidden layer dimensions. If None, uses [512, 256, 128]
            activation (str): Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
            dropout_rate (float): Dropout rate (0.0 means no dropout)
            batch_norm (bool): Whether to use batch normalization
            use_bias (bool): Whether to use bias in linear layers
        """
        super(FlexibleMLP, self).__init__()
        
        # Default hidden dimensions if not specified
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Choose activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build the network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList() if dropout_rate > 0 else None
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim, bias=use_bias))
            
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim, bias=use_bias)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # He initialization for ReLU, Xavier for others
                if self.activation == F.relu or self.activation == F.leaky_relu:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(layer.weight)
                
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Output layer initialization
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Flatten input if needed (for MNIST images)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply batch normalization if enabled
            if self.batch_norm:
                x = self.batch_norms[i](x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout if enabled
            if self.dropout_rate > 0:
                x = self.dropouts[i](x)
        
        # Output layer (no activation for classification)
        x = self.output_layer(x)
        
        return x
    
    def get_num_parameters(self):
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_layer_info(self):
        """Get information about the network architecture."""
        info = []
        info.append(f"Input dimension: {self.input_dim}")
        
        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            params = prev_dim * hidden_dim + (hidden_dim if self.layers[i].bias is not None else 0)
            info.append(f"Hidden layer {i+1}: {prev_dim} -> {hidden_dim} ({params:,} parameters)")
            prev_dim = hidden_dim
        
        output_params = prev_dim * self.output_dim + (self.output_dim if self.output_layer.bias is not None else 0)
        info.append(f"Output layer: {prev_dim} -> {self.output_dim} ({output_params:,} parameters)")
        info.append(f"Total parameters: {self.get_num_parameters():,}")
        
        if self.batch_norm:
            info.append("Batch normalization: Enabled")
        if self.dropout_rate > 0:
            info.append(f"Dropout rate: {self.dropout_rate}")
        
        return '\n'.join(info)


def plot_results(input_k, results_dict, save_path=None, ylim_abs=None, ylim_rel=None, xlim=None, plot_bias=False):
    """Plot approximation error results."""
    cb_blue = "#0072B2"
    cb_orange = "#D55E00"
    cb_green = "#009E73"
    cb_black = "#000000"
    cb_purple = "#984EA3"
    colors = [cb_blue, cb_orange, cb_green, cb_black, cb_purple]
    colors_dict = {}
    for key in results_dict:
        colors_dict[key] = colors.pop(0)

    # Convert tensors to CPU for plotting
    def to_cpu(tensor_or_list):
        if isinstance(tensor_or_list, torch.Tensor):
            return tensor_or_list.detach().cpu().numpy()
        elif isinstance(tensor_or_list, list):
            return [t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in tensor_or_list]
        else:
            return tensor_or_list

    # Check if bias data is available and requested
    has_bias_data = all('bias' in results_dict[key] for key in results_dict if isinstance(results_dict[key], dict))
    
    if plot_bias and has_bias_data:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for key in results_dict:
        if isinstance(results_dict[key], dict):
            rel_data = to_cpu(results_dict[key]['rel'])
            abs_data = to_cpu(results_dict[key]['abs'])
            
            ax1.plot(input_k, rel_data, linestyle='-', 
                    color=colors_dict[key], label=key)
            ax2.plot(input_k, abs_data, linestyle='--', 
                    color=colors_dict[key], label=key)
            
            # Plot bias data if available and requested
            if plot_bias and has_bias_data and 'bias' in results_dict[key]:
                bias_data = to_cpu(results_dict[key]['bias'])
                ax3.plot(input_k, bias_data, linestyle=':', 
                        color=colors_dict[key], label=key)

    ax1.set_xlabel('Sketch size factor (k)')
    ax1.set_ylabel('Relative error')
    ax1.set_title('Relative Error vs. Sketch Size')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Sketch size factor (k)')
    ax2.set_ylabel('Absolute error')
    ax2.set_title('Absolute Error vs. Sketch Size')
    ax2.legend()
    ax2.grid(True)
    
    if plot_bias and has_bias_data:
        ax3.set_xlabel('Sketch size factor (k)')
        ax3.set_ylabel('Bias')
        ax3.set_title('Bias vs. Sketch Size')
        ax3.legend()
        ax3.grid(True)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Reference line at y=0
    
    if ylim_abs is not None:
        ax2.set_ylim(0, ylim_abs)
    if ylim_rel is not None:
        ax1.set_ylim(0, ylim_rel)
    if xlim is not None:
        ax1.set_xlim(0, xlim)
        ax2.set_xlim(0, xlim)
        if plot_bias and has_bias_data:
            ax3.set_xlim(0, xlim)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def compute_reg_var_estimates(new_gradients_vec, hessian_matrix_offset_inv, num_oracle_calls):
    """
    Computes reg_var_estimate on chunks of the new_gradients_vec.
    
    Args:
        new_gradients_vec (torch.Tensor): Input tensor of shape [N, d].
        hessian_matrix_offset_inv (torch.Tensor): Inverse Hessian matrix [d, d].
        num_oracle_calls (int): Number of chunks to divide the gradients into.
        
    Returns:
        torch.Tensor: Joined vector of reg_var_estimates.
    """
    num_samples = new_gradients_vec.shape[0]
    
    # Ensure num_oracle_calls doesn't exceed num_samples and chunk_size is at least 1
    num_oracle_calls = min(num_oracle_calls, num_samples)
    chunk_size = max(1, num_samples // num_oracle_calls)
    
    reg_var_estimates_list = []
    
    for i in range(0, num_samples, chunk_size):
        chunk = new_gradients_vec[i:i+chunk_size]
        chunk_avg = torch.mean(chunk, dim=0)
        H_chunk_avg = torch.matmul(hessian_matrix_offset_inv, chunk_avg)
        reg_var_chunk = torch.einsum('bi,i->b', chunk, H_chunk_avg)
        reg_var_estimates_list.append(reg_var_chunk)
    
    joined_vector = torch.cat(reg_var_estimates_list, dim=0)
    return joined_vector


def compute_reg_var_errors(k_vec, x, oracle, H_mat, num_params):
    """
    Compute RegVar approximation errors for different chunk sizes.
    
    Args:
        k_vec (array): Array of sketch size factors
        x (torch.Tensor): Gradients for new data points
        oracle (torch.Tensor): Oracle variances
        H_mat (torch.Tensor): Inverse Hessian matrix
        num_params (int): Number of parameters
        
    Returns:
        tuple: Matrices and vectors of errors
    """
    estimated_variance_mat = torch.zeros(len(k_vec), x.shape[0])
    rel_error_mat = torch.zeros(len(k_vec), x.shape[0])
    abs_error_mat = torch.zeros(len(k_vec), x.shape[0])
    rel_error_vec = [0.0 for _ in range(len(k_vec))]
    abs_error_vec = [0.0 for _ in range(len(k_vec))]

    for k in range(len(k_vec)):
        num_oracle_calls = max(1, int(num_params * k_vec[k]))  # Ensure at least 1 oracle call
        estimated_variances = compute_reg_var_estimates(x, H_mat, num_oracle_calls=num_oracle_calls)
        estimated_variance_mat[k, :] = estimated_variances
        rel_error_mat[k, :] = torch.abs(oracle - estimated_variances) / (oracle + 1)
        abs_error_mat[k, :] = torch.abs(oracle - estimated_variances)
        
        rel_error_vec[k] = torch.mean(rel_error_mat[k, :])
        abs_error_vec[k] = torch.mean(abs_error_mat[k, :])
        
        chunk_size = x.shape[0] // num_oracle_calls
        print(f"ind={k}, chunk_size={chunk_size}, rel_error:{rel_error_vec[k]:.4f}, "
              f"abs_error:{abs_error_vec[k]:.4f}")

    return estimated_variance_mat, rel_error_mat, abs_error_mat, rel_error_vec, abs_error_vec


def compute_approximation_errors_low_rank(k_vec, x, oracle, H_low, decomp_coef, num_params, 
                                        basis="g", method="two_sided", approximator=None):
    """
    Compute approximation errors using low-rank decomposition of the Hessian.
    
    Args:
        k_vec (array): Array of sketch size factors
        x (torch.Tensor): Gradients for new data points
        oracle (torch.Tensor): Oracle variances
        H_low (torch.Tensor): Low-rank component of inverse Hessian
        decomp_coef (float): Decomposition coefficient (lambda)
        num_params (int): Number of parameters
        basis (str): Sketching basis type
        method (str): Sketching method
        approximator: LaplaceSketchApproximator instance (no longer needed)
        
    Returns:
        tuple: Matrices and vectors of errors
    """
    estimated_variance_mat = torch.zeros(len(k_vec), x.shape[0])
    rel_error_mat = torch.zeros(len(k_vec), x.shape[0])
    abs_error_mat = torch.zeros(len(k_vec), x.shape[0])
    rel_error_vec = [0.0 for _ in range(len(k_vec))]
    abs_error_vec = [0.0 for _ in range(len(k_vec))]

    for k in range(len(k_vec)):
        sketch_size = int(num_params * k_vec[k])
        
        if method == "two_sided":
            test_Q_mat, test_left_sketch, test_right_sketch = compute_Q_mat(
                sketch_size, H_low, num_params, basis=basis, gradients=x)
        elif method == "right_only":
            H_sketch, test_right_sketch = right_sketch_only(
                sketch_size, H_low, num_params, basis=basis, gradients=x)

        estimated_variances = []
        for i in range(x.shape[0]):
            if method == "two_sided":
                estimated_variance = compute_estimated_variance(
                    x[i, :], test_left_sketch, test_Q_mat, test_right_sketch
                ) + decomp_coef * torch.dot(x[i, :], x[i, :])
            elif method == "right_only":
                estimated_variance = compute_estimated_variance_right_only(
                    x[i, :], H_sketch, test_right_sketch
                ) + decomp_coef * torch.dot(x[i, :], x[i, :])
            estimated_variances.append(estimated_variance)
        
        estimated_variances = torch.stack(estimated_variances)
        estimated_variance_mat[k, :] = estimated_variances
        rel_error_mat[k, :] = torch.abs(oracle - estimated_variances) / (oracle + 1)
        abs_error_mat[k, :] = torch.abs(oracle - estimated_variances)

        rel_error_vec[k] = torch.mean(rel_error_mat[k, :])
        abs_error_vec[k] = torch.mean(abs_error_mat[k, :])

        print(f"k={sketch_size:4d}, rel_error:{rel_error_vec[k]:.4f}, "
              f"abs_error:{abs_error_vec[k]:.4f}")

    return estimated_variance_mat, rel_error_mat, abs_error_mat, rel_error_vec, abs_error_vec


def analyze_gradient_correlations(gradients, num_trials=1000):
    """
    Analyze correlations between normalized gradients.
    
    Args:
        gradients (torch.Tensor): Gradient matrix [N, d]
        num_trials (int): Number of random pairs to sample
        
    Returns:
        tuple: (dot_products, mean_correlation, gradient_norms)
    """
    dot_products = []
    norms = gradients.norm(dim=1, keepdim=True)
    norms[norms == 0] = 1.0
    gradients_normalized = gradients / norms

    for _ in range(num_trials):
        indices = torch.randperm(gradients_normalized.shape[0])[:2]
        row1 = gradients_normalized[indices[0]]
        row2 = gradients_normalized[indices[1]]
        dp = torch.dot(row1, row2).item()
        dot_products.append(dp)

    return dot_products, np.mean(dot_products), norms.view(-1)


def plot_gradient_analysis(gradients, save_path=None):
    """
    Create comprehensive plots for gradient analysis.
    
    Args:
        gradients (torch.Tensor): Gradient matrix
        save_path (str): Optional path to save plot
    """
    dot_products, mean_corr, norms = analyze_gradient_correlations(gradients)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dot product histogram
    ax1.hist(dot_products, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(mean_corr, color='red', linestyle='--', 
                label=f'Mean: {mean_corr:.3f}')
    ax1.set_xlabel('Dot Product')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Gradient Correlations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Norm histogram
    ax2.hist(norms.cpu().numpy(), bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Gradient Norm')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Gradient Norms')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_stable_rank_analysis(hessian_matrices, labels=None, save_path=None):
    """
    Plot singular value distributions for stable rank analysis.
    
    Args:
        hessian_matrices (list): List of matrices to analyze
        labels (list): Labels for each matrix
        save_path (str): Optional path to save plot
    """
    if labels is None:
        labels = [f'Matrix {i+1}' for i in range(len(hessian_matrices))]
    
    n_plots = len(hessian_matrices)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 3.3))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, (matrix, label) in enumerate(zip(hessian_matrices, labels)):
        svdvals = torch.linalg.svdvals(matrix)
        axes[i].hist(svdvals.cpu().numpy(), bins=50, log=True)
        axes[i].set_title(label)
        axes[i].set_xlabel('Singular Values')
        if i == 0:
            axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def compute_stable_rank_vs_offset(hessian_inv, offset_range):
    """
    Compute stable rank as a function of diagonal offset.
    
    Args:
        hessian_inv (torch.Tensor): Inverse Hessian matrix
        offset_range (array): Range of offset values to test
        
    Returns:
        tuple: (offset_values, stable_ranks)
    """
    stable_ranks = []
    
    for offset in offset_range:
        modified_matrix = hessian_inv - offset * torch.eye(hessian_inv.shape[0])
        singular_vals = torch.linalg.svdvals(modified_matrix)
        frobenius_sq = torch.sum(singular_vals ** 2)
        spectral_sq = singular_vals[0] ** 2
        stable_rank = frobenius_sq / spectral_sq
        stable_ranks.append(stable_rank.item())
    
    return offset_range, stable_ranks


def plot_comprehensive_error_comparison(input_k, all_results, save_path=None):
    """
    Create a comprehensive comparison plot of all sketching methods.
    
    Args:
        input_k (array): Sketch size factors
        all_results (dict): Dictionary containing results from all methods
        save_path (str): Optional path to save plot
    """
    colors = {
        'two_sided': "#0072B2",
        'right_only': "#D55E00", 
        'low_rank': "#009E73",
        'low_rank_right': "#CC79A7",
        'regvar': "#F0E442"
    }
    
    linestyles = {
        'two_sided': '-',
        'right_only': '--',
        'low_rank': '-.',
        'low_rank_right': ':',
        'regvar': '-'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Relative error plot
    for method, results in all_results.items():
        if 'rel' in results:
            ax1.plot(input_k, results['rel'], 
                    linestyle=linestyles.get(method, '-'),
                    color=colors.get(method, 'black'),
                    label=method.replace('_', ' ').title(),
                    linewidth=2)
    
    ax1.set_xlabel('Sketch size factor (k)', fontsize=12)
    ax1.set_ylabel('Relative error', fontsize=12)
    ax1.set_title('Relative Error Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Absolute error plot  
    for method, results in all_results.items():
        if 'abs' in results:
            ax2.plot(input_k, results['abs'],
                    linestyle=linestyles.get(method, '-'),
                    color=colors.get(method, 'black'),
                    label=method.replace('_', ' ').title(),
                    linewidth=2)
    
    ax2.set_xlabel('Sketch size factor (k)', fontsize=12)
    ax2.set_ylabel('Absolute error', fontsize=12)
    ax2.set_title('Absolute Error Comparison', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def create_visualization_summary(approximator, model, hessian_inv, flat_params, save_dir="./results/"):
    """
    Create a comprehensive set of visualizations for the analysis.
    
    Args:
        approximator: LaplaceSketchApproximator instance (for input_dim and device)
        model: Trained model
        hessian_inv: Inverse Hessian matrix
        flat_params: Flattened parameters
        save_dir: Directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate test gradients
    test_gradients, _ = generate_test_gradients(model, flat_params, approximator.input_dim, approximator.device, num_samples=1000)
    oracle_vars = torch.einsum('bi,ij,bj->b', test_gradients, hessian_inv, test_gradients)
    
    # 1. Gradient correlation analysis
    print("Creating gradient correlation analysis...")
    plot_gradient_analysis(test_gradients, save_path=f"{save_dir}/gradient_analysis.pdf")
    
    # 2. Stable rank analysis  
    print("Creating stable rank analysis...")
    low_rank_hessian = hessian_inv - 10.0 * torch.eye(hessian_inv.shape[0])
    plot_stable_rank_analysis(
        [hessian_inv, low_rank_hessian],
        labels=['Original Hessian⁻¹', 'Low Rank Hessian⁻¹'],
        save_path=f"{save_dir}/stable_rank_analysis.pdf"
    )
    
    # 3. Comprehensive error comparison
    print("Running comprehensive error analysis...")
    input_k = np.concatenate([[0.01, 0.02, 0.03], np.linspace(0.04, 0.3, 8)])
    
    # Run all methods (simplified for demo)
    _, _, _, rel_two, abs_two, bias_two = compute_approximation_errors(
        k_vec=input_k, x=test_gradients, oracle=oracle_vars,
        H_inv=hessian_inv, num_params=flat_params.shape[0],
        basis="g", method="two_sided"
    )
    
    _, _, _, rel_right, abs_right, bias_right = compute_approximation_errors(
        k_vec=input_k, x=test_gradients, oracle=oracle_vars,
        H_inv=hessian_inv, num_params=flat_params.shape[0],
        basis="g", method="right_only"
    )
    
    all_results = {
        'two_sided': {'rel': rel_two, 'abs': abs_two, 'bias': bias_two},
        'right_only': {'rel': rel_right, 'abs': abs_right, 'bias': bias_right}
    }
    
    plot_comprehensive_error_comparison(
        input_k, all_results, 
        save_path=f"{save_dir}/comprehensive_comparison.pdf"
    )
    
    print(f"All visualizations saved to {save_dir}")


# Visualization utility functions
def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_3d_surface(X, Y, Z, title="3D Surface", save_path=None):
    """Create a 3D surface plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y') 
    ax.set_zlabel('z')
    ax.set_title(title)
    ax.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax, shrink=0.5)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def compute_stable_rank(A):
    """Compute the stable rank of a matrix."""
    singular_vals = torch.linalg.svdvals(A)
    frobenius_sq = torch.sum(singular_vals ** 2)
    spectral_sq = singular_vals[0] ** 2
    stable_rank = frobenius_sq / spectral_sq
    return stable_rank


def create_cyclic_basis_matrix(k, n):
    """Create a cyclic basis matrix for sketching."""
    matrix = torch.zeros((k, n))
    indices = torch.arange(k) % n
    matrix[torch.arange(k), indices] = 1
    column_norms = torch.norm(matrix, dim=0)
    matrix = matrix / torch.clamp(column_norms, min=1.0)
    return matrix


def compute_estimated_variance(x, S1, Q, S2):
    """Compute estimated variance using two-sided sketching.
    
    Args:
        x: Input gradients - can be:
           - [num_params] for single gradient vector
           - [n, num_params] for batch of n gradient vectors
        S1: Left sketch matrix [sketch_size, num_params]
        Q: Q matrix [sketch_size, sketch_size]
        S2: Right sketch matrix [sketch_size, num_params]
    
    Returns:
        - scalar if x is [num_params]
        - [n] tensor if x is [n, num_params]
    """
    # Ensure all tensors are on the same device
    device = Q.device
    x = x.to(device)
    S1 = S1.to(device)
    S2 = S2.to(device)
    
    # Handle both single vector and matrix cases
    if x.dim() == 1:
        # Single vector case: x is [num_params]
        S2x = torch.matmul(S2, x)  # [sketch_size]
        QS2x = torch.matmul(Q, S2x)  # [sketch_size]
        S1T_QS2x = torch.matmul(S1.transpose(0, 1), QS2x)  # [num_params]
        result = torch.dot(x, S1T_QS2x)  # scalar
        return result
    elif x.dim() == 2:
        # Matrix case: x is [n, num_params]
        S2x = torch.matmul(S2, x.T)  # [sketch_size, n]
        QS2x = torch.matmul(Q, S2x)  # [sketch_size, n]
        S1T_QS2x = torch.matmul(S1.transpose(0, 1), QS2x)  # [num_params, n]
        result = torch.einsum('ni,in->n', x, S1T_QS2x)  # [n]
        return result
    else:
        raise ValueError(f"x must be 1D or 2D, got {x.dim()}D")


def compute_estimated_variance_subspace(x, H_inv_k):
    """Compute estimated variance using subspace sketching x'H_inv_k x.
    
    Args:
        x: Input gradients - can be:
           - [num_params] for single gradient vector
           - [n, num_params] for batch of n gradient vectors
        H_inv_k: Subspace approximation of inverse Hessian [num_params, num_params]
    
    Returns:
        - scalar if x is [num_params]
        - [n] tensor if x is [n, num_params]
    """
    # Ensure all tensors are on the same device
    device = H_inv_k.device
    x = x.to(device)
    H_inv_k = H_inv_k.to(device)
    
    # Handle both single vector and matrix cases
    if x.dim() == 1:
        # Single vector case: x is [num_params]
        H_inv_k_x = torch.matmul(H_inv_k, x)  # [num_params]
        result = torch.dot(x, H_inv_k_x)  # scalar
        return result
    elif x.dim() == 2:
        # Matrix case: x is [n, num_params]
        H_inv_k_x = torch.matmul(H_inv_k, x.T)  # [num_params, n]
        result = torch.einsum('ni,in->n', x, H_inv_k_x)  # [n]
        return result
    else:
        raise ValueError(f"x must be 1D or 2D, got {x.dim()}D")

def compute_estimated_variance_right_only(x, H_sketch, S_right):
    """Compute estimated variance using right-only sketching.
    
    Args:
        x: Input gradients - can be:
           - [num_params] for single gradient vector
           - [n, num_params] for batch of n gradient vectors
        H_sketch: Sketched Hessian Inverse [num_params, sketch_size]
        S_right: Right sketch matrix [sketch_size, num_params]
    
    Returns:
        - scalar if x is [num_params]
        - [n] tensor if x is [n, num_params]
    """
    # Ensure all tensors are on the same device
    device = H_sketch.device
    x = x.to(device)
    S_right = S_right.to(device)
    
    # Handle both single vector and matrix cases
    if x.dim() == 1:
        # Single vector case: x is [num_params]
        S_right_x = torch.matmul(S_right, x)  # [sketch_size]
        H_sketch_S_right_x = torch.matmul(H_sketch, S_right_x)  # [num_params]
        result = torch.dot(x, H_sketch_S_right_x)  # scalar
        return result
    elif x.dim() == 2:
        # Matrix case: x is [n, num_params]
        S_right_x = torch.matmul(S_right, x.T)  # [sketch_size, n]
        H_sketch_S_right_x = torch.matmul(H_sketch, S_right_x)  # [num_params, n]
        result = torch.einsum('ni,in->n', x, H_sketch_S_right_x)  # [n]
        return result
    else:
        raise ValueError(f"x must be 1D or 2D, got {x.dim()}D")

def compute_estimated_variance_right_bagging(x, H_sketch, S_right):
    """Compute estimated variance using right-only sketching with bagging
    
    Args:
        x: Input gradients
        H_sketch: List of Sketch matrices
        S_right: List of Right sketch matrices
        bag_factor: Bagging factor for the right-only sketching
    """
    assert len(H_sketch) == len(S_right), "H_sketch and S_right must have the same length"
    # Ensure all tensors are on the same device
    device = H_sketch.device
    x = x.to(device)
    S_right = [S_right.to(device) for S_right in S_right]
    H_sketch = [H_sketch.to(device) for H_sketch in H_sketch]
    
    if x.dim() > 1:
        x = x.squeeze()
    
    S_right_x = [torch.matmul(S_right_i, x) for S_right_i in S_right]
    H_sketch_S_right_x = [torch.matmul(H_sketch_i, S_right_x_i) for H_sketch_i, S_right_x_i in zip(H_sketch, S_right_x)]
    result = torch.mean([torch.dot(x, H_sketch_S_right_x_i) for H_sketch_S_right_x_i in H_sketch_S_right_x])
    return result

def compute_Q_mat(sketch_size, H_in, num_params, verbose=False, basis="g", gradients=None):
    """Compute the Q matrix for two-sided sketching."""
    # Get the device of H_in to ensure all tensors are on the same device
    device = H_in.device
    
    if basis == "c":
        sketch_matrix_left = create_cyclic_basis_matrix(sketch_size, num_params).to(device)
        sketch_matrix_right = create_cyclic_basis_matrix(sketch_size, num_params).to(device)
    elif basis == "g":
        sketch_matrix_left = torch.randn(sketch_size, num_params, device=device) / np.sqrt(sketch_size)
        sketch_matrix_right = torch.randn(sketch_size, num_params, device=device) / np.sqrt(sketch_size)
    elif basis == "radmacher":
        sketch_matrix_left = (torch.randint(0, 2, (sketch_size, num_params), device=device).float() * 2 - 1) / np.sqrt(sketch_size)    
        sketch_matrix_right = (torch.randint(0, 2, (sketch_size, num_params), device=device).float() * 2 - 1) / np.sqrt(sketch_size)    
    elif basis == "pca":
        if gradients is None:
            raise ValueError("Gradients must be provided for PCA sketching")
        pca = PCA(n_components=sketch_size, svd_solver='randomized')
        pca.fit(gradients.cpu().numpy())
        sketch_matrix_left = torch.from_numpy(pca.components_).float().to(device)
        sketch_matrix_right = torch.from_numpy(pca.components_).float().to(device)
        sketch_matrix_left = sketch_matrix_left / torch.norm(sketch_matrix_left, dim=1, keepdim=True)
        sketch_matrix_right = sketch_matrix_right / torch.norm(sketch_matrix_right, dim=1, keepdim=True)
    
    def compute_quadratic_form(vec):
        vec = vec.float().to(device)
        result = torch.dot(vec, torch.matmul(H_in, vec))
        return result
    
    Q_mat = torch.zeros(sketch_size, sketch_size, device=device)
    
    for i in range(sketch_size):
        for j in range(sketch_size):
            Q_mat[i, j] = compute_quadratic_form(sketch_matrix_left[i,:] + sketch_matrix_right[j,:])
            Q_mat[i, j] -= compute_quadratic_form(sketch_matrix_left[i,:])
            Q_mat[i, j] -= compute_quadratic_form(sketch_matrix_right[j,:])
            Q_mat[i, j] = Q_mat[i, j] / 2
    
    return Q_mat, sketch_matrix_left, sketch_matrix_right


def right_sketch_only(sketch_size, H_in, num_params, verbose=False, basis="g", gradients=None):
    """Compute H_sketch matrix using right-only sketching."""
    # Get the device of H_in to ensure all tensors are on the same device
    device = H_in.device
    
    if basis == "c":
        sketch_matrix_right = create_cyclic_basis_matrix(sketch_size, num_params).to(device)
    elif basis == "g":
        sketch_matrix_right = torch.randn(sketch_size, num_params, device=device) / np.sqrt(sketch_size)
    elif basis == "radmacher":
        sketch_matrix_right = (torch.randint(0, 2, (sketch_size, num_params), device=device).float() * 2 - 1) / np.sqrt(sketch_size)
    elif basis == "pca":
        if gradients is None:
            raise ValueError("Gradients must be provided for PCA sketching")
        pca = PCA(n_components=sketch_size, svd_solver='randomized')
        pca.fit(gradients.cpu().numpy())
        sketch_matrix_right = torch.from_numpy(pca.components_).float().to(device)
        sketch_matrix_right = sketch_matrix_right / torch.norm(sketch_matrix_right, dim=1, keepdim=True)
    
    H_sketch = H_in @ sketch_matrix_right.T
    if verbose:
        print(f"H_in shape: {H_in.shape}")
        print(f"H_sketch shape: {H_sketch.shape}")
        print(f"sketch_matrix_right shape: {sketch_matrix_right.shape}")
    return H_sketch, sketch_matrix_right

def right_sketch_only_with_bagging(sketch_size, H_in, num_params, verbose=False, basis="g", gradients=None, bag_factor: int = 1):
    """Compute H_sketch matrix using right-only sketching with bagging"""
    assert bag_factor > 0, "Bagging factor must be positive"

    H_sketch = []   
    S_right = []
    size_local = max(sketch_size // bag_factor, 1)
    for i in range(bag_factor):
        # Ensure at least one sketch per bag
        H_sketch_i, S_right_i = right_sketch_only(size_local, H_in, num_params, verbose=verbose, basis=basis, gradients=gradients)
        H_sketch.append(H_sketch_i)
        S_right.append(S_right_i)
    assert len(H_sketch) == bag_factor, "H_sketch must have the same length as the bagging factor"
    return H_sketch, S_right
 

def top_subspace_approximation_hessian(subspace_size, hessian, diagonal_offset=0.1):
    """
    Return (H_k + δI)^{-1}, where H_k is the rank-k eigen-approximation of `hessian`.
    `hessian` must be symmetric (as Hessians are).
    """
    # Eigen-decomposition (symmetric ⇒ eigh)
    eigvals, eigvecs = torch.linalg.eigh(hessian)          # ascending order
    top_vals  = eigvals[-subspace_size:]                   # largest k
    top_vecs  = eigvecs[:, -subspace_size:]                # corresponding eigen-vectors

    # Rank-k reconstruction
    Hk = top_vecs @ torch.diag(top_vals) @ top_vecs.T
    print(f"Hk :{Hk}")

    # Add diagonal offset for numerical stability and invert
    I  = torch.eye(hessian.size(0), dtype=hessian.dtype, device=hessian.device)
    return torch.linalg.inv(Hk + diagonal_offset * I)

def top_subspace_approximation_hessian_woodbury(subspace_size, hessian, diagonal_offset=0.1):
    """Input is the hessian matrix and the offset. Returns (δI + UΛUᵀ)⁻¹ using Woodbury.  Returns H_inv of shape (n,n)."""
    # Eigen-decompose once (symmetric Hessian)
    eigvals, eigvecs = torch.linalg.eigh(hessian)     # ascending
    U = eigvecs[:, -subspace_size:]                   # n × k
    Λ = eigvals[-subspace_size:]                      # (k,)

    δ_inv = 1.0 / diagonal_offset                     # scalar
    Λ_inv = torch.diag(1.0 / Λ)                       # k × k

    # B = (Λ⁻¹ + δ⁻¹ I_k)⁻¹  = (Λ⁻¹ + δ⁻¹ I)⁻¹   (k × k)
    B = torch.linalg.inv(Λ_inv + δ_inv * torch.eye(subspace_size,
                                                   device=hessian.device,
                                                   dtype=hessian.dtype))

    # Woodbury: δ⁻¹ I - δ⁻¹ U B Uᵀ δ⁻¹
    UH = U @ B                                        # n × k
    H_inv = δ_inv * torch.eye(hessian.size(0),
                              dtype=hessian.dtype,
                              device=hessian.device) \
            - δ_inv**2 * UH @ U.T
    return H_inv

def top_subspace_approximation_of_inverse_hessian(k: int, inv_hessian: torch.Tensor):
    """
    Rank-k approximation of the *inverse* Hessian, (H⁻¹)_k  ≈  V_k Λ_k V_kᵀ.

    Args
    ----
    k : int
        Number of leading eigen-components to keep.
    inv_hessian : (n,n) torch.Tensor
        Symmetric positive-definite inverse Hessian.

    Returns
    -------
    (n,n) torch.Tensor
        Low-rank approximation of the inverse Hessian.
    """
    # Eigen-decomposition is cheaper and numerically exact for symmetric matrices
    eigvals, eigvecs = torch.linalg.eigh(inv_hessian)       # ascending order
    top_vals = eigvals[-k:]                                 # k largest eigenvalues
    top_vecs = eigvecs[:, -k:]                              # corresponding eigenvectors

    return top_vecs @ torch.diag(top_vals) @ top_vecs.T


def generate_test_gradients(model, flat_params, input_dim, device, num_samples=5000):
    """Generate gradients for new test data points."""
    print(f"Generating gradients for {num_samples} test points...")
    
    new_data = np.random.uniform(low=-2, high=2, size=(int(num_samples), input_dim))
    new_inputs = torch.from_numpy(new_data).type(torch.float32).to(device)
    
    def grad_func(flat_params, new_inputs):
        param_dict = {}
        pointer = 0
        for name, param in model.named_parameters():
            num_param = param.numel()
            param_dict[name] = flat_params[pointer:pointer + num_param].view_as(param)
            pointer += num_param
        
        outputs = torch.func.functional_call(model, param_dict, new_inputs)
        return outputs
    
    new_gradients = torch.autograd.functional.jacobian(
        lambda fp: grad_func(fp, new_inputs),
        flat_params
    ).squeeze(1)
    
    return new_gradients, new_inputs


@torch.no_grad()                     # avoids accidental second‑order graphs
def generate_test_gradients_functorch(
    model,
    input_dim,
    device,
    num_samples: int = 5_000,
    batch_size: int = 1_024,
):
    """
    Generate per‑example gradients ∂f/∂θ for randomly‑drawn test points.

    Returns
    -------
    gradients : torch.Tensor [num_samples, n_params]
        Row *i* is the gradient for input *i*.
    inputs    : torch.Tensor [num_samples, input_dim]
        The test points themselves (useful for downstream plots).
    """
    # --------------------------------------------------------------------- #
    # 1. Sample inputs
    # --------------------------------------------------------------------- #
    inputs = torch.empty(num_samples, input_dim, device=device).uniform_(-2, 2)

    # --------------------------------------------------------------------- #
    # 2. Turn the module into a purely functional version
    #    (params, buffers) held externally so functorch can differentiate
    #    w.r.t. parameters efficiently.
    # --------------------------------------------------------------------- #
    fmodel, params, buffers = make_functional_with_buffers(model.eval())

    # total number of (leaf) trainable parameters
    n_params = sum(p.numel() for p in params)

    # single‑point function:  θ, x  ↦  f_θ(x)
    def f(current_params, x):
        return fmodel(current_params, buffers, x.unsqueeze(0)).squeeze(0)

    # --------------------------------------------------------------------- #
    # 3. Compute Jacobians in mini‑batches
    # --------------------------------------------------------------------- #
    grad_chunks = []
    for x_chunk in inputs.split(batch_size):
        # jacrev → reverse AD over parameters
        # vmap   → do it for every example in the chunk in one shot
        # Result is a *tuple* of tensors, one per parameter tensor.
        per_ex_jac = vmap(jacrev(f), (None, 0))(params, x_chunk)

        # Flatten each per‑parameter Jacobian and concatenate along columns
        flat = torch.cat(
            [g.reshape(g.shape[0], -1) for g in per_ex_jac], dim=1
        )                                   # shape = [chunk, n_params]
        grad_chunks.append(flat)

    gradients = torch.cat(grad_chunks, dim=0)        # [num_samples, n_params]
    return gradients.contiguous(), inputs


def generate_real_data_gradients(model, flat_params, data_loader, device, max_samples=None):
    """
    Generate gradients for real data points in a batched manner.
    
    Args:
        model: Trained model
        flat_params: Flattened model parameters
        data_loader: DataLoader containing real data
        device: Device to run computations on
        max_samples: Maximum number of samples to process (None for all)
        
    Returns:
        torch.Tensor: Gradients matrix [N, d] where N is number of samples, d is parameter dimension
    """
    print(f"Generating gradients for real data...")
    
    def grad_func(flat_params, inputs):
        """Compute model outputs given flattened parameters and inputs."""
        param_dict = {}
        pointer = 0
        for name, param in model.named_parameters():
            num_param = param.numel()
            param_dict[name] = flat_params[pointer:pointer + num_param].view_as(param)
            pointer += num_param
        
        outputs = torch.func.functional_call(model, param_dict, inputs)
        return outputs
    
    gradient_batches = []
    input_batches = []
    total_samples = 0
    
    model.eval()
    
    for batch_idx, batch in enumerate(data_loader):
        # Handle different batch formats (data only vs data+targets)
        if isinstance(batch, (tuple, list)):
            batch_inputs = batch[0].to(device)
        else:
            batch_inputs = batch.to(device)
        
        # Flatten inputs if needed (e.g., for image data)
        if batch_inputs.dim() > 2:
            batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)
        
        # Compute gradients for this batch
        batch_gradients = torch.autograd.functional.jacobian(
            lambda fp: grad_func(fp, batch_inputs),
            flat_params
        ).squeeze(1)  # Remove singleton dimension for binary classification (output dim)
        
        gradient_batches.append(batch_gradients)
        total_samples += batch_inputs.size(0)
        input_batches.append(batch_inputs)
        # Check if we've reached the maximum samples limit
        if max_samples is not None and total_samples >= max_samples:
            # Trim the last batch if it exceeds max_samples
            excess = total_samples - max_samples
            if excess > 0:
                gradient_batches[-1] = gradient_batches[-1][:-excess]
                total_samples = max_samples
            break
    
    # Merge all gradient batches
    all_gradients = torch.cat(gradient_batches, dim=0)
    all_inputs = torch.cat(input_batches, dim=0)
    # return the inputs as well    
    print(f"Generated gradients for {total_samples} real data points")
    print(f"Gradient matrix shape: {all_gradients.shape}")
    print(f"Input matrix shape: {all_inputs.shape}")
    
    return all_gradients, all_inputs


def load_and_print_results(results_file):
    """
    Load and print results from a saved pickle file.
    
    Args:
        results_file (str): Path to the saved pickle results file
    """
    try:
        with open(results_file, 'rb') as f:
            results_dict = pickle.load(f)
        
        print(f"Loaded results from: {results_file}")
        print("=" * 60)
        
        # Iterate through each model configuration
        for model_name, results in results_dict.items():
            print(f"\nModel: {model_name}")
            print("-" * 40)
            
            # Basic model info
            print(f"Number of parameters: {results['num_params']:,}")
            print(f"Right sketch size: {results['right_sketch_value']}")
            print(f"Left sketch sizes: {results['left_sketch_vec']}")
            print(f"Lambda values: {results['f_lambda_in']}")
            
            # Training info
            if 'training_info' in results:
                training_info = results['training_info']
                print(f"Training epochs: {training_info['num_epochs']}")
                if training_info['final_train_loss'] is not None:
                    print(f"Final train loss: {training_info['final_train_loss']:.4f}")
                if training_info['final_test_loss'] is not None:
                    print(f"Final test accuracy: {training_info['final_test_loss']:.4f}")
            
            # Oracle variance statistics
            if 'oracle_vars_stats' in results:
                print("\nOracle Variance Statistics:")
                oracle_stats = results['oracle_vars_stats']
                for f_lambda in results['f_lambda_in']:
                    print(f"  λ={f_lambda}: mean={oracle_stats['mean'][f_lambda]:.6f}, "
                          f"std={oracle_stats['std'][f_lambda]:.6f}, "
                          f"min={oracle_stats['min'][f_lambda]:.6f}, "
                          f"max={oracle_stats['max'][f_lambda]:.6f}")
            
            # Right-only approximation errors
            if 'summary_stats_right_only' in results:
                print("\nRight-only Sketching Errors:")
                right_stats = results['summary_stats_right_only']
                for f_lambda in results['f_lambda_in']:
                    rel_mean = right_stats[f_lambda]['relative']['mean']
                    rel_std = right_stats[f_lambda]['relative']['std']
                    abs_mean = right_stats[f_lambda]['absolute']['mean']
                    abs_std = right_stats[f_lambda]['absolute']['std']
                    print(f"  λ={f_lambda}: rel_error={rel_mean:.4f}±{rel_std:.4f}, "
                          f"abs_error={abs_mean:.6f}±{abs_std:.6f}")
            
            # Both-sided approximation errors (summary for different sketch sizes)
            if 'summary_stats_both' in results:
                print("\nBoth-sided Sketching Errors (best and worst):")
                both_stats = results['summary_stats_both']
                for f_lambda in results['f_lambda_in']:
                    rel_means = both_stats[f_lambda]['relative']['mean']
                    abs_means = both_stats[f_lambda]['absolute']['mean']
                    
                    # Find best and worst sketch sizes
                    sketch_sizes = list(rel_means.keys())
                    best_sketch = min(sketch_sizes, key=lambda s: rel_means[s])
                    worst_sketch = max(sketch_sizes, key=lambda s: rel_means[s])
                    
                    print(f"  λ={f_lambda}:")
                    print(f"    Best (sketch_size={best_sketch}): rel_error={rel_means[best_sketch]:.4f}, "
                          f"abs_error={abs_means[best_sketch]:.6f}")
                    print(f"    Worst (sketch_size={worst_sketch}): rel_error={rel_means[worst_sketch]:.4f}, "
                          f"abs_error={abs_means[worst_sketch]:.6f}")
            
            print("=" * 60)
    
    except FileNotFoundError:
        print(f"Error: File {results_file} not found.")
    except Exception as e:
        print(f"Error loading results: {e}")

def compute_approximation_errors(k_vec, x, oracle, H_inv, num_params, basis="g", method="two_sided", H_mat=None, offset=0.1, bag_factor=1.0):
    """Compute approximation errors for different sketch sizes.
    
    Args:
        k_vec: List of sketch size factors, these values are relative to the number of parameters
        x: Input gradients [n, num_params]
        oracle: Oracle variances [n]
        H_inv: Inverse of Hessian matrix
        num_params: Number of parameters
        basis: Basis for sketching
        method: Method for approximation
        H_mat: Hessian matrix 
        offset: Offset for the Hessian matrix
        bag_factor: Bagging factor for the right-only sketching
    """
    # Get the device of H_inv to ensure all tensors are on the same device
    device = H_inv.device
    
    # Ensure input tensors are on the correct device
    x = x.to(device)
    oracle = oracle.to(device)
    
    estimated_variance_mat = torch.zeros(len(k_vec), x.shape[0], device=device)
    rel_error_mat = torch.zeros(len(k_vec), x.shape[0], device=device)
    abs_error_mat = torch.zeros(len(k_vec), x.shape[0], device=device)
    rel_error_vec = [0.0 for _ in range(len(k_vec))]
    abs_error_vec = [0.0 for _ in range(len(k_vec))]
    bias_vec = [0.0 for _ in range(len(k_vec))]
    memory_footprint_vec = [0 for _ in range(len(k_vec))]
    memory_footprint = None
    
    for k in range(len(k_vec)):
        sketch_size = int(num_params * k_vec[k])
        
        if method == "two_sided":
            test_Q_mat, test_left_sketch, test_right_sketch = compute_Q_mat(
                sketch_size, H_inv, num_params, basis=basis, gradients=x
            )
            memory_footprint = sketch_size**2
            # Vectorized computation for all gradients at once
            estimated_variances = compute_estimated_variance(
                x, test_left_sketch, test_Q_mat, test_right_sketch
            )
            
        elif method == "right_only":
            H_sketch, test_right_sketch = right_sketch_only(
                sketch_size, H_inv, num_params, basis=basis, gradients=x
            )
            memory_footprint = sketch_size * num_params
            # Vectorized computation for all gradients at once
            estimated_variances = compute_estimated_variance_right_only(
                x, H_sketch, test_right_sketch
            )
            
        elif method == "right_bagging":
            H_sketch, test_right_sketch = right_sketch_only_with_bagging(
                sketch_size, H_inv, num_params, basis=basis, gradients=x, bag_factor=bag_factor
            )
            # Note: This still needs individual computation due to bagging complexity
            estimated_variances = []
            for i in range(x.shape[0]):
                estimated_variance = compute_estimated_variance_right_bagging(
                    x[i, :], H_sketch, test_right_sketch
                )
                estimated_variances.append(estimated_variance)
            estimated_variances = torch.stack(estimated_variances)
            
        elif method == "subspace": # compute the top r subspace of the inverse
            H_inv_k = top_subspace_approximation_of_inverse_hessian(sketch_size, H_inv)
            # Vectorized computation for all gradients at once
            estimated_variances = compute_estimated_variance_subspace(x, H_inv_k)
            
        elif method == "inverse_of_subspace": # first compute the subspace and do the inverse of the subspace
            H_k_inv = top_subspace_approximation_hessian_woodbury(sketch_size, H_mat, offset)
            # Vectorized computation for all gradients at once
            estimated_variances = compute_estimated_variance_subspace(x, H_k_inv)
        
        estimated_variance_mat[k, :] = estimated_variances
        rel_error_mat[k, :] = torch.abs(oracle - estimated_variances) / (oracle + 1)
        abs_error_mat[k, :] = torch.abs(oracle - estimated_variances)
        
        rel_error_vec[k] = torch.mean(rel_error_mat[k, :])
        abs_error_vec[k] = torch.mean(abs_error_mat[k, :])
        bias_vec[k] = torch.mean(oracle - estimated_variances)
        memory_footprint_vec[k] = memory_footprint if memory_footprint is not None else 0
        
        print(f"method={method}, k={sketch_size:4d}, memory_footprint= {memory_footprint:4d}, rel_error: {rel_error_vec[k]:.4f}, abs_error: {abs_error_vec[k]:.4f}, bias: {bias_vec[k]:.4f}")
    
    return estimated_variance_mat, rel_error_mat, abs_error_mat, rel_error_vec, abs_error_vec, bias_vec