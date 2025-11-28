"""
Utility functions for training
Checkpointing, early stopping, plotting, etc.
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        is_best: Whether this is the best model so far
        filename: Path to save checkpoint
    """
    torch.save(state, filename)
    print(f"💾 Checkpoint saved: {filename}")
    
    if is_best:
        best_path = os.path.join(os.path.dirname(filename), 'best_model.pth')
        torch.save(state, best_path)
        print(f"⭐ Best model saved: {best_path}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        epoch: Epoch number from checkpoint
        vocab: Vocabulary from checkpoint (if available)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint. get('epoch', 0)
    vocab = checkpoint.get('vocab', None)
    
    print(f"✅ Checkpoint loaded from epoch {epoch}")
    return epoch, vocab


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    def __init__(self, patience=5, verbose=True, delta=0.0):
        """
        Args:
            patience: How many epochs to wait after last improvement
            verbose: Whether to print messages
            delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        
    def __call__(self, val_loss, model):
        """
        Check if should stop training
        
        Args:
            val_loss: Current validation loss
            model: Model (unused, kept for compatibility)
        """
        score = -val_loss
        
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"   ⏳ EarlyStopping counter: {self.counter}/{self. patience}")
            
            if self.counter >= self. patience:
                self.early_stop = True
        else:
            # Improvement! 
            self.best_score = score
            self. val_loss_min = val_loss
            self.counter = 0


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    
    plt. xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss Over Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for best validation loss
    best_val_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    plt.annotate(f'Best Val Loss: {best_val_loss:.4f}',
                xy=(float(best_val_epoch), float(best_val_loss)),
                xytext=(float(best_val_epoch + 1), float(best_val_loss + 0.2)),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training curves saved: {save_path}")


def count_parameters(model):
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        total: Total number of parameters
        trainable: Number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds):
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"