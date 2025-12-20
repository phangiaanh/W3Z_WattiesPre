"""
Training logger for iteration-level metrics and convergence statistics.
"""
import os
import csv
import json
import time
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class TrainingLogger:
    """Logger for training and validation metrics at iteration level."""
    
    def __init__(
        self,
        log_dir: str,
        log_format: str = 'csv',
        log_gradient_norm: bool = True,
        log_learning_rate: bool = True,
        log_iteration_every: int = 1
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save log files
            log_format: 'csv' or 'json'
            log_gradient_norm: Whether to log gradient norms
            log_learning_rate: Whether to log learning rate
            log_iteration_every: Log every N iterations
        """
        self.log_dir = log_dir
        self.log_format = log_format
        self.log_gradient_norm = log_gradient_norm
        self.log_learning_rate = log_learning_rate
        self.log_iteration_every = log_iteration_every
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files
        self.train_log_file = os.path.join(log_dir, 'training_iterations.csv')
        self.val_log_file = os.path.join(log_dir, 'validation_iterations.csv')
        self.convergence_log_file = os.path.join(log_dir, 'convergence_stats.csv')
        
        # Initialize CSV writers
        self.train_writer = None
        self.val_writer = None
        self.convergence_writer = None
        
        # Track data for convergence statistics
        self.epoch_stats = {
            'lr_values': [],
            'grad_norm_values': [],
            'train_losses': [],
            'val_losses': [],
            'val_metrics': defaultdict(list)
        }
        
        # Global iteration counter
        self.global_iteration = 0
        
        # Initialize CSV files with headers
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Training iterations CSV
        train_headers = ['epoch', 'iteration', 'batch_idx', 'loss_total']
        train_headers.extend(['loss_classification', 'loss_keypoints_2d', 'loss_keypoints_3d'])
        train_headers.extend(['loss_pose', 'loss_shape', 'loss_camera'])
        if self.log_learning_rate:
            train_headers.append('lr')
        if self.log_gradient_norm:
            train_headers.append('grad_norm')
        train_headers.append('time')
        
        with open(self.train_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(train_headers)
        
        # Validation iterations CSV
        val_headers = ['epoch', 'iteration', 'batch_idx', 'loss_total']
        val_headers.extend(['loss_classification', 'loss_keypoints_2d', 'loss_keypoints_3d'])
        val_headers.extend(['loss_pose', 'loss_shape', 'loss_camera'])
        val_headers.extend(['P@0.1', 'P@0.15', 'P@H', 'PAJ', 'PAV'])
        val_headers.append('time')
        
        with open(self.val_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(val_headers)
        
        # Convergence statistics CSV
        convergence_headers = ['epoch', 'avg_lr', 'avg_grad_norm', 'avg_train_loss']
        convergence_headers.extend(['avg_val_loss', 'best_metric_value', 'metric_trend'])
        convergence_headers.extend(['P@0.1', 'P@0.15', 'P@H', 'PAJ', 'PAV'])
        
        with open(self.convergence_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(convergence_headers)
    
    def log_training_iteration(
        self,
        epoch: int,
        batch_idx: int,
        losses: Dict[str, float],
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        iteration_time: Optional[float] = None
    ):
        """
        Log training iteration metrics.
        
        Args:
            epoch: Current epoch
            batch_idx: Batch index
            losses: Dictionary of loss values
            lr: Current learning rate
            grad_norm: Gradient norm
            iteration_time: Time taken for this iteration
        """
        if self.global_iteration % self.log_iteration_every != 0:
            self.global_iteration += 1
            return
        
        row = [epoch, self.global_iteration, batch_idx, losses.get('total', 0.0)]
        row.append(losses.get('classification', 0.0))
        row.append(losses.get('keypoints_2d', 0.0))
        row.append(losses.get('keypoints_3d', 0.0))
        row.append(losses.get('pose', 0.0))
        row.append(losses.get('shape', 0.0))
        row.append(losses.get('camera', 0.0))
        
        if self.log_learning_rate:
            row.append(lr if lr is not None else 0.0)
        if self.log_gradient_norm:
            row.append(grad_norm if grad_norm is not None else 0.0)
        
        row.append(iteration_time if iteration_time is not None else 0.0)
        
        # Write to CSV
        with open(self.train_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Track for convergence statistics
        if lr is not None:
            self.epoch_stats['lr_values'].append(lr)
        if grad_norm is not None:
            self.epoch_stats['grad_norm_values'].append(grad_norm)
        self.epoch_stats['train_losses'].append(losses.get('total', 0.0))
        
        self.global_iteration += 1
    
    def log_validation_iteration(
        self,
        epoch: int,
        batch_idx: int,
        losses: Dict[str, float],
        metrics: Dict[str, float],
        iteration_time: Optional[float] = None
    ):
        """
        Log validation iteration metrics.
        
        Args:
            epoch: Current epoch
            batch_idx: Batch index
            losses: Dictionary of loss values
            metrics: Dictionary of validation metrics
            iteration_time: Time taken for this iteration
        """
        if self.global_iteration % self.log_iteration_every != 0:
            return
        
        row = [epoch, self.global_iteration, batch_idx, losses.get('total', 0.0)]
        row.append(losses.get('classification', 0.0))
        row.append(losses.get('keypoints_2d', 0.0))
        row.append(losses.get('keypoints_3d', 0.0))
        row.append(losses.get('pose', 0.0))
        row.append(losses.get('shape', 0.0))
        row.append(losses.get('camera', 0.0))
        
        # Add validation metrics
        row.append(metrics.get('P@0.1', 0.0))
        row.append(metrics.get('P@0.15', 0.0))
        row.append(metrics.get('P@H', 0.0))
        row.append(metrics.get('PAJ', 0.0))
        row.append(metrics.get('PAV', 0.0))
        
        row.append(iteration_time if iteration_time is not None else 0.0)
        
        # Write to CSV
        with open(self.val_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Track for convergence statistics
        self.epoch_stats['val_losses'].append(losses.get('total', 0.0))
        for metric_name, metric_value in metrics.items():
            self.epoch_stats['val_metrics'][metric_name].append(metric_value)
    
    def log_convergence_stats(
        self,
        epoch: int,
        best_metric_value: float,
        metric_trend: str = 'stable'
    ):
        """
        Log convergence statistics at end of epoch.
        
        Args:
            epoch: Current epoch
            best_metric_value: Best metric value so far
            metric_trend: Trend description ('improving', 'stable', 'degrading')
        """
        # Compute averages
        avg_lr = np.mean(self.epoch_stats['lr_values']) if self.epoch_stats['lr_values'] else 0.0
        avg_grad_norm = np.mean(self.epoch_stats['grad_norm_values']) if self.epoch_stats['grad_norm_values'] else 0.0
        avg_train_loss = np.mean(self.epoch_stats['train_losses']) if self.epoch_stats['train_losses'] else 0.0
        avg_val_loss = np.mean(self.epoch_stats['val_losses']) if self.epoch_stats['val_losses'] else 0.0
        
        # Compute average metrics
        avg_metrics = {}
        for metric_name in ['P@0.1', 'P@0.15', 'P@H', 'PAJ', 'PAV']:
            if metric_name in self.epoch_stats['val_metrics']:
                avg_metrics[metric_name] = np.mean(self.epoch_stats['val_metrics'][metric_name])
            else:
                avg_metrics[metric_name] = 0.0
        
        # Write to CSV
        row = [epoch, avg_lr, avg_grad_norm, avg_train_loss, avg_val_loss, best_metric_value, metric_trend]
        row.extend([
            avg_metrics.get('P@0.1', 0.0),
            avg_metrics.get('P@0.15', 0.0),
            avg_metrics.get('P@H', 0.0),
            avg_metrics.get('PAJ', 0.0),
            avg_metrics.get('PAV', 0.0)
        ])
        
        with open(self.convergence_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Reset epoch stats for next epoch
        self.epoch_stats = {
            'lr_values': [],
            'grad_norm_values': [],
            'train_losses': [],
            'val_losses': [],
            'val_metrics': defaultdict(list)
        }
    
    def get_global_iteration(self) -> int:
        """Get current global iteration number."""
        return self.global_iteration
    
    def set_global_iteration(self, iteration: int):
        """Set global iteration number (for resume)."""
        self.global_iteration = iteration

