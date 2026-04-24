import torch
import numpy as np
from typing import Optional

import os
import json
from pathlib import Path
from datetime import datetime
import torch.nn.functional as F


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_config(args, save_dir):
    """
    Save configuration to JSON file
    
    Args:
        args: argparse arguments
        save_dir: Path to save directory
    """
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"💾 Config saved to {config_path}")


def save_checkpoint(
    epoch, 
    model, 
    optimizer, 
    scheduler, 
    metrics, 
    save_dir, 
    is_best=False,
    max_keep=5
):
    """
    Save model checkpoint
    
    Args:
        epoch: Current epoch
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state (optional)
        metrics: Current metrics dict
        save_dir: Directory to save checkpoint
        is_best: Whether this is the best model
        max_keep: Maximum number of checkpoints to keep
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics
    }
    
    # Save regular checkpoint
    checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = save_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"   💎 Best model saved! (Val loss: {metrics['loss']:.4f})")
    
    # Keep only last N checkpoints
    checkpoints = sorted(save_dir.glob('checkpoint_epoch_*.pt'))
    if len(checkpoints) > max_keep:
        for old_ckpt in checkpoints[:-max_keep]:
            old_ckpt.unlink()


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
    
    Returns:
        epoch, metrics
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"✅ Checkpoint loaded from epoch {epoch}")
    return epoch, metrics


def create_run_directory(base_dir, run_name=None):
    """
    Create directory for current run
    
    Args:
        base_dir: Base directory for all runs
        run_name: Optional custom run name
    
    Returns:
        Path to run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if run_name:
        dir_name = f"{run_name}_{timestamp}"
    else:
        dir_name = f"run_{timestamp}"
    
    run_dir = Path(base_dir) / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir
