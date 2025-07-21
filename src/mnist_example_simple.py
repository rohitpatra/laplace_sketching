#!/usr/bin/env python3
"""
MNIST Classification with FlexibleMLP - Multi-class and Binary Classification
"""

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
from utils import FlexibleMLP

def download_mnist():
    """Download MNIST dataset manually with multiple fallback URLs."""
    
    # Multiple mirror URLs for robustness
    urls = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/mnist/",
    ]
    
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz", 
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    os.makedirs("/shared/public/sharing/laplace-sketching/data", exist_ok=True)
    
    for file in files:
        filepath = os.path.join("/shared/public/sharing/laplace-sketching/data", file)
        if os.path.exists(filepath):
            print(f"✅ {file} already exists")
            continue
            
        downloaded = False
        for base_url in urls:
            try:
                print(f"Downloading {file} from {base_url}...")
                urllib.request.urlretrieve(base_url + file, filepath)
                print(f"✅ Successfully downloaded {file}")
                downloaded = True
                break
            except Exception as e:
                print(f"❌ Failed to download from {base_url}: {e}")
                continue
        
        if not downloaded:
            print(f"❌ Could not download {file} from any source!")
            print("Please check your internet connection or try again later.")
            return False
    
    return True

def load_mnist_images(filename):
    """Load MNIST images from gz file."""
    with gzip.open(filename, 'rb') as f:
        # Skip header (16 bytes)
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        # Reshape to (num_images, 784)
        return data.reshape(-1, 784).astype(np.float32) / 255.0

def load_mnist_labels(filename):
    """Load MNIST labels from gz file."""
    with gzip.open(filename, 'rb') as f:
        # Skip header (8 bytes)
        return np.frombuffer(f.read(), np.uint8, offset=8)

def convert_to_binary_labels(labels, positive_class=7):
    """Convert multi-class labels to binary labels.
    
    Args:
        labels: Original labels (0-9)
        positive_class: Single digit (int) or collection of digits (list/set/tuple) 
                       that should be labeled as positive (1)
    
    Returns:
        binary_labels: 1 for positive class(es), 0 for negative class(es)
    """
    # Handle both single digit and collection of digits
    if isinstance(positive_class, (int, float)):
        # Single digit case
        binary_labels = (labels == positive_class).astype(np.float32)
        positive_digits = [positive_class]
    else:
        # Multiple digits case (list, set, tuple)
        positive_digits = list(positive_class)
        binary_labels = np.isin(labels, positive_digits).astype(np.float32)
    
    return binary_labels, positive_digits

def create_mnist_dataloaders_simple(batch_size=128, normalize=True):
    """Create MNIST dataloaders for multi-class classification."""
    base_path = "/shared/public/sharing/laplace-sketching/mnist_data"
    # Check if MNIST files exist, download if needed
    files_to_check = [
        os.path.join(base_path, "train-images-idx3-ubyte.gz"),
        os.path.join(base_path, "train-labels-idx1-ubyte.gz"),
        os.path.join(base_path, "t10k-images-idx3-ubyte.gz"),
        os.path.join(base_path, "t10k-labels-idx1-ubyte.gz")
    ]
    
    files_exist = all(os.path.exists(file) for file in files_to_check)
    
    if not files_exist:
        print("MNIST files not found. Downloading...")
        success = download_mnist()
        if not success:
            raise RuntimeError("Failed to download MNIST dataset. Please check your internet connection.")
    else:
        print("✅ MNIST files already exist, skipping download.")
    
    # Load MNIST data
    print("Loading MNIST data...")
    train_images = load_mnist_images(os.path.join(base_path, "train-images-idx3-ubyte.gz"))
    train_labels = load_mnist_labels(os.path.join(base_path, "train-labels-idx1-ubyte.gz"))
    test_images = load_mnist_images(os.path.join(base_path, "t10k-images-idx3-ubyte.gz"))
    test_labels = load_mnist_labels(os.path.join(base_path, "t10k-labels-idx1-ubyte.gz"))
    
    # Normalize if requested (MNIST standard normalization)
    if normalize:
        mean, std = 0.1307, 0.3081
        train_images = (train_images - mean) / std
        test_images = (test_images - mean) / std
    
    # Convert to PyTorch tensors
    train_images = torch.FloatTensor(train_images)
    train_labels = torch.LongTensor(train_labels)
    test_images = torch.FloatTensor(test_images)
    test_labels = torch.LongTensor(test_labels)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)  
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader

def create_binary_mnist_dataloaders(batch_size=128, normalize=True, positive_class=7):
    """Create MNIST dataloaders for binary classification.
    
    Args:
        batch_size: Batch size for dataloaders
        normalize: Whether to normalize the data
        positive_class: Single digit (int) or collection of digits (list/set/tuple)
                       that should be labeled as positive class
    """
    base_path = "/shared/public/sharing/laplace-sketching/mnist_data"
    # Check if MNIST files exist, download if needed
    files_to_check = [
        os.path.join(base_path, "train-images-idx3-ubyte.gz"),
        os.path.join(base_path, "train-labels-idx1-ubyte.gz"),
        os.path.join(base_path, "t10k-images-idx3-ubyte.gz"),
        os.path.join(base_path, "t10k-labels-idx1-ubyte.gz")
    ]
    
    files_exist = all(os.path.exists(file) for file in files_to_check)
    
    if not files_exist:
        print("MNIST files not found. Downloading...")
        success = download_mnist()
        if not success:
            raise RuntimeError("Failed to download MNIST dataset. Please check your internet connection.")
    else:
        print("✅ MNIST files already exist, skipping download.")
    
    # Load MNIST data
    print("Loading MNIST data...")
    train_images = load_mnist_images(os.path.join(base_path, "train-images-idx3-ubyte.gz"))
    train_labels = load_mnist_labels(os.path.join(base_path, "train-labels-idx1-ubyte.gz"))
    test_images = load_mnist_images(os.path.join(base_path, "t10k-images-idx3-ubyte.gz"))
    test_labels = load_mnist_labels(os.path.join(base_path, "t10k-labels-idx1-ubyte.gz"))
    
    # Convert to binary labels
    train_labels_binary, positive_digits = convert_to_binary_labels(train_labels, positive_class)
    test_labels_binary, _ = convert_to_binary_labels(test_labels, positive_class)
    
    # Create description strings for display
    if len(positive_digits) == 1:
        positive_desc = f"digit {positive_digits[0]}"
        negative_desc = "other digits"
    else:
        positive_desc = f"digits {positive_digits}"
        negative_desc = f"digits {[d for d in range(10) if d not in positive_digits]}"
    
    print(f"Converting to binary classification: {positive_desc} vs {negative_desc}...")
    
    # Print class distribution
    train_positive = np.sum(train_labels_binary)
    train_negative = len(train_labels_binary) - train_positive
    test_positive = np.sum(test_labels_binary)
    test_negative = len(test_labels_binary) - test_positive
    
    print(f"📊 Training set:")
    print(f"   Positive ({positive_desc}): {train_positive:,} samples ({train_positive/len(train_labels_binary)*100:.1f}%)")
    print(f"   Negative ({negative_desc}): {train_negative:,} samples ({train_negative/len(train_labels_binary)*100:.1f}%)")
    
    print(f"📊 Test set:")
    print(f"   Positive ({positive_desc}): {test_positive:,} samples ({test_positive/len(test_labels_binary)*100:.1f}%)")
    print(f"   Negative ({negative_desc}): {test_negative:,} samples ({test_negative/len(test_labels_binary)*100:.1f}%)")
    
    # Normalize if requested (MNIST standard normalization)
    if normalize:
        mean, std = 0.1307, 0.3081
        train_images = (train_images - mean) / std
        test_images = (test_images - mean) / std
    
    # Convert to PyTorch tensors
    train_images = torch.FloatTensor(train_images)
    train_labels_binary = torch.FloatTensor(train_labels_binary)
    test_images = torch.FloatTensor(test_images)
    test_labels_binary = torch.FloatTensor(test_labels_binary)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_images, train_labels_binary)
    test_dataset = TensorDataset(test_images, test_labels_binary)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=10, lr=0.001, device='cpu'):
    """Train the FlexibleMLP model on MNIST (multi-class)."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    
    print(f"Training on device: {device}")
    print(f"Model architecture:\n{model.get_layer_info()}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}')
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Testing phase
        test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, '
              f'Test Accuracy = {test_accuracy:.2f}%')
    
    return train_losses, test_accuracies

def train_binary_model(model, train_loader, test_loader, num_epochs=10, lr=0.001, device='cpu', best_model_path='best_local_model.pt'):
    """Train the FlexibleMLP model for binary classification."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    
    # Track best model
    best_accuracy = 0.0
    
    print(f"Training binary classifier on device: {device}")
    print(f"Model architecture:\n{model.get_layer_info()}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            output = model(data).squeeze()  # Remove extra dimension for binary classification
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}')
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Testing phase
        metrics = evaluate_binary_model(model, test_loader, device)
        test_accuracies.append(metrics['accuracy'])
        test_precisions.append(metrics['precision'])
        test_recalls.append(metrics['recall'])
        test_f1_scores.append(metrics['f1'])
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, '
              f'Test Acc = {metrics["accuracy"]:.2f}%, '
              f'Precision = {metrics["precision"]:.2f}%, '
              f'Recall = {metrics["recall"]:.2f}%, '
              f'F1 = {metrics["f1"]:.2f}%')
        
        # Save best model based on accuracy
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            torch.save(model.state_dict(), best_model_path)
    
    # Load best model at the end (only if accuracy improved)
    if best_accuracy > 0.0 and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    return model, train_losses, test_accuracies, test_precisions, test_recalls, test_f1_scores


def compute_hessian_binary(model, test_loader, device='cpu', diagonal_offset=0.1):
    """Compute the Hessian matrix of the loss with respect to model parameters for binary classification."""
    print("Computing Hessian matrix...")
        
    # Create an uncompiled copy of the model for Hessian computation
    # torch.compile interferes with higher-order derivatives needed for Hessian
    if hasattr(model, '_orig_mod'):
        # Model was compiled, use the original uncompiled version
        hessian_model = model._orig_mod
        print("Using uncompiled model for Hessian computation")
    else:
        # Model was not compiled
        hessian_model = model
    # Collect all validation data
    eval_inputs = []
    eval_targets = []
    for batch_data, batch_target in test_loader:
        eval_inputs.append(batch_data.to(device))
        eval_targets.append(batch_target.to(device))
    eval_inputs = torch.cat(eval_inputs, dim=0)
    eval_targets = torch.cat(eval_targets, dim=0)
    hessian_model.eval()
    flat_params = torch.nn.utils.parameters_to_vector(hessian_model.parameters()).detach().clone()
    flat_params.requires_grad = True
    def loss_func(flat_params):
        param_dict = {}
        pointer = 0
        for name, param in hessian_model.named_parameters():
            num_param = param.numel()
            param_dict[name] = flat_params[pointer:pointer + num_param].view_as(param)
            pointer += num_param
        outputs = torch.func.functional_call(hessian_model, param_dict, eval_inputs)
        # Squeeze outputs to match target shape for binary classification
        outputs = outputs.squeeze()
        loss = nn.BCEWithLogitsLoss()(outputs, eval_targets)
        return loss
    hessian_matrix = torch.autograd.functional.hessian(loss_func, flat_params)
    # Add diagonal offset for numerical stability
    identity = torch.eye(hessian_matrix.size(0), dtype=hessian_matrix.dtype, device=hessian_matrix.device)
    hessian_matrix_offset = hessian_matrix + diagonal_offset * identity
    hessian_matrix_offset_inv = torch.inverse(hessian_matrix_offset)    
    return hessian_matrix_offset_inv, flat_params, hessian_matrix, hessian_matrix_offset



    
    
def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy on test set (multi-class)."""
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def evaluate_binary_model(model, test_loader, device='cpu'):
    """Evaluate binary classification model with detailed metrics."""
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data).squeeze()
            
            # Convert logits to probabilities and then to predictions
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    true_positives = np.sum((all_predictions == 1) & (all_targets == 1))
    false_positives = np.sum((all_predictions == 1) & (all_targets == 0))
    false_negatives = np.sum((all_predictions == 0) & (all_targets == 1))
    true_negatives = np.sum((all_predictions == 0) & (all_targets == 0))
    
    accuracy = (true_positives + true_negatives) / len(all_targets) * 100
    precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }

def plot_training_results(train_losses, test_accuracies, filename=None):
    """Plot training loss and test accuracy (multi-class)."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training loss
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot test accuracy
        ax2.plot(test_accuracies)
        ax2.set_title('Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")
        print("Skipping plots...")

def plot_binary_training_results(train_losses, test_accuracies, test_precisions, test_recalls, test_f1_scores):
    """Plot training results for binary classification."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training loss
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot test accuracy
        ax2.plot(test_accuracies)
        ax2.set_title('Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        # Plot precision and recall
        ax3.plot(test_precisions, label='Precision')
        ax3.plot(test_recalls, label='Recall')
        ax3.set_title('Precision and Recall')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Percentage (%)')
        ax3.legend()
        ax3.grid(True)
        
        # Plot F1 score
        ax4.plot(test_f1_scores)
        ax4.set_title('F1 Score')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score (%)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")
        print("Skipping plots...")

def run_multiclass_classification():
    """Run multi-class MNIST classification."""
    print("🔢 Running Multi-class MNIST Classification (0-9)")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create MNIST dataloaders
    print("Loading MNIST dataset...")
    train_loader, test_loader = create_mnist_dataloaders_simple(batch_size=128)
    
    # Define model configurations to test
    model_configs = [
        {
            'name': 'Small MLP (2 layers)',
            'hidden_dims': [32, 32],
            'dropout_rate': 0.0,
            'batch_norm': False
        },
        {
            'name': 'Medium MLP (3 layers) with Dropout',
            'hidden_dims': [512, 256, 128],
            'dropout_rate': 0.2,
            'batch_norm': False
        },
        {
            'name': 'Large MLP (4 layers) with BatchNorm',
            'hidden_dims': [800, 400, 200, 100],
            'dropout_rate': 0.1,
            'batch_norm': True
        }
    ]
    
    results = {}
    
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"Training: {config['name']}")
        print(f"{'='*60}")
        
        # Create model
        model = FlexibleMLP(
            input_dim=784,  # 28x28 MNIST images
            output_dim=10,  # 10 digit classes
            hidden_dims=config['hidden_dims'],
            activation='relu',
            dropout_rate=config['dropout_rate'],
            batch_norm=config['batch_norm']
        )
        
        # Train model
        train_losses, test_accuracies = train_model(
            model, train_loader, test_loader, 
            num_epochs=10, lr=0.001, device=device
        )
        
        results[config['name']] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1],
            'num_params': model.get_num_parameters()
        }
        
        print(f"Final test accuracy: {test_accuracies[-1]:.2f}%")
        print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("MULTI-CLASS CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  - Final Accuracy: {result['final_accuracy']:.2f}%")
        print(f"  - Parameters: {result['num_params']:,}")
        print()
    
    # Plot results for the first model
    if results:
        first_config = list(results.keys())[0]
        plot_training_results(
            results[first_config]['train_losses'], 
            results[first_config]['test_accuracies']
        )

def run_binary_classification():
    """Run binary MNIST classification with different positive class options."""
    print("🔢 Running Binary MNIST Classification")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Different positive class configurations
    print("\nChoose positive class configuration:")
    print("1. Single digit 7 vs all others")
    print("2. High digits {7,8,9} vs low digits {0,1,2,3,4,5,6}")
    print("3. Even digits {0,2,4,6,8} vs odd digits {1,3,5,7,9}")
    print("4. Middle digits {3,4,5,6} vs edge digits {0,1,2,7,8,9}")
    print("5. Custom (enter your own)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                positive_class = 7
                task_name = "Digit 7 vs All Others"
                break
            elif choice == '2':
                positive_class = {7, 8, 9}
                task_name = "High Digits vs Low Digits"
                break
            elif choice == '3':
                positive_class = {0, 2, 4, 6, 8}
                task_name = "Even Digits vs Odd Digits"
                break
            elif choice == '4':
                positive_class = {3, 4, 5, 6}
                task_name = "Middle Digits vs Edge Digits"
                break
            elif choice == '5':
                try:
                    digits_str = input("Enter positive digits (e.g., '1,3,5' or just '7'): ").strip()
                    if ',' in digits_str:
                        positive_class = {int(d.strip()) for d in digits_str.split(',')}
                    else:
                        positive_class = int(digits_str)
                    
                    # Validate digits
                    if isinstance(positive_class, set):
                        if not all(0 <= d <= 9 for d in positive_class):
                            raise ValueError("All digits must be between 0 and 9")
                        if len(positive_class) == 0:
                            raise ValueError("Must specify at least one positive digit")
                        if len(positive_class) == 10:
                            raise ValueError("Cannot use all digits as positive class")
                        task_name = f"Custom: {sorted(positive_class)} vs Others"
                    else:
                        if not (0 <= positive_class <= 9):
                            raise ValueError("Digit must be between 0 and 9")
                        task_name = f"Custom: Digit {positive_class} vs Others"
                    break
                except ValueError as e:
                    print(f"❌ Invalid input: {e}")
                    continue
            else:
                print("❌ Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\n👋 Cancelled!")
            return
    
    print(f"\n🎯 Selected task: {task_name}")
    
    # Create binary MNIST dataloaders
    print("Setting up binary MNIST classification...")
    train_loader, test_loader = create_binary_mnist_dataloaders(
        batch_size=128, 
        positive_class=positive_class
    )
    
    # Define model configurations to test
    model_configs = [
        {
            'name': 'Small Binary MLP (2 layers)',
            'hidden_dims': [32, 32],
            'dropout_rate': 0.0,
            'batch_norm': False
        },
        {
            'name': 'Medium Binary MLP (3 layers) with Dropout',
            'hidden_dims': [512, 256, 128],
            'dropout_rate': 0.2,
            'batch_norm': False
        },
        {
            'name': 'Large MLP (4 layers) with BatchNorm',
            'hidden_dims': [800, 400, 200, 100],
            'dropout_rate': 0.1,
            'batch_norm': True
        }
    ]
    
    results = {}
    
    for config in model_configs:
        print(f"\n{'='*70}")
        print(f"Training: {config['name']}")
        print(f"{'='*70}")
        
        # Create model for binary classification (output_dim=1)
        model = FlexibleMLP(
            input_dim=784,  # 28x28 MNIST images
            output_dim=1,   # Binary classification (single output)
            hidden_dims=config['hidden_dims'],
            activation='relu',
            dropout_rate=config['dropout_rate'],
            batch_norm=config['batch_norm']
        )
        
        # Train model
        model, train_losses, test_accuracies, test_precisions, test_recalls, test_f1_scores = train_binary_model(
            model, train_loader, test_loader, 
            num_epochs=10, lr=0.001, device=device
        )
        
        # Final evaluation
        final_metrics = evaluate_binary_model(model, test_loader, device)
        
        results[config['name']] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'final_metrics': final_metrics,
            'num_params': model.get_num_parameters(),
            'precisions': test_precisions,
            'recalls': test_recalls,
            'f1_scores': test_f1_scores
        }
        
        print(f"\n📊 Final Results:")
        print(f"   Accuracy: {final_metrics['accuracy']:.2f}%")
        print(f"   Precision: {final_metrics['precision']:.2f}%")
        print(f"   Recall: {final_metrics['recall']:.2f}%")
        print(f"   F1 Score: {final_metrics['f1']:.2f}%")
        print(f"   Model parameters: {model.get_num_parameters():,}")
        
        # Confusion matrix info
        print(f"\n🔍 Confusion Matrix:")
        print(f"   True Positives: {final_metrics['true_positives']}")
        print(f"   False Positives: {final_metrics['false_positives']}")
        print(f"   False Negatives: {final_metrics['false_negatives']}")
        print(f"   True Negatives: {final_metrics['true_negatives']}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("BINARY CLASSIFICATION SUMMARY")
    print(f"Task: {task_name}")
    print(f"{'='*70}")
    for name, result in results.items():
        metrics = result['final_metrics']
        print(f"{name}:")
        print(f"  - Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  - Precision: {metrics['precision']:.2f}%")
        print(f"  - Recall: {metrics['recall']:.2f}%")
        print(f"  - F1 Score: {metrics['f1']:.2f}%")
        print(f"  - Parameters: {result['num_params']:,}")
        print()
    
    # Plot results for the first model
    if results:
        first_config = list(results.keys())[0]
        first_result = results[first_config]
        plot_binary_training_results(
            first_result['train_losses'], 
            first_result['test_accuracies'],
            first_result['precisions'],
            first_result['recalls'],
            first_result['f1_scores']
        )

def main():
    """Main function with menu for classification type."""
    print("🎯 MNIST Classification with FlexibleMLP")
    print("=" * 50)
    print("Choose classification type:")
    print("1. Multi-class Classification (digits 0-9)")
    print("2. Binary Classification (flexible positive class selection)")
    print("3. Both")
    
    while True:
        try:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            
            if choice == '1':
                run_multiclass_classification()
                break
            elif choice == '2':
                run_binary_classification()
                break
            elif choice == '3':
                print("\n" + "="*80)
                print("🔄 Running Both Classification Types")
                print("="*80)
                run_multiclass_classification()
                print("\n" + "="*80)
                run_binary_classification()
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 