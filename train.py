"""
Training script for Category-Routed Multi-Regressor Model.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import numpy as np
import time

from models.model import CategoryRoutedModel
from models.losses import CombinedLoss
from utils.metrics import Evaluator
from utils.logger import TrainingLogger
from dataset import Animal3DDataset


def setup_device(cfg):
    """Setup device."""
    if cfg.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def create_model(cfg, device):
    """Create model from config."""
    # Build config dict for model
    model_cfg = {
        'image_size': cfg.model.image_size,
        'num_categories': cfg.model.num_categories,
        'backbone': OmegaConf.to_container(cfg.model.backbone, resolve=True),
        'cls_header': OmegaConf.to_container(cfg.model.get('cls_header', {}), resolve=True),
        'classifier': OmegaConf.to_container(cfg.model.classifier, resolve=True),
        'regressors': OmegaConf.to_container(cfg.model.regressors, resolve=True),
        'router': OmegaConf.to_container(cfg.model.get('router', {}), resolve=True),
        'smal': OmegaConf.to_container(cfg.model.get('smal', {}), resolve=True),
        'huggingface': OmegaConf.to_container(cfg.get('huggingface', {}), resolve=True),
    }
    
    model = CategoryRoutedModel(model_cfg)
    model = model.to(device)
    
    return model


def create_dataloader(cfg, split='train'):
    """Create dataloader."""
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    
    # Resolve token with priority: command-line > huggingface.token > data.dataset.token > env var
    token = None
    if 'huggingface' in cfg and cfg.huggingface.get('token') is not None:
        token = cfg.huggingface.token
    elif dataset_cfg.get('token') is not None:
        token = dataset_cfg['token']
    else:
        token = os.getenv('HUGGINGFACE_TOKEN', None)
    
    # Override token in dataset config if resolved
    if token is not None:
        dataset_cfg['token'] = token
    
    # Also pass cache_dir from huggingface config if available
    hf_cache_dir = cfg.get('huggingface', {}).get('cache_dir', None)
    if hf_cache_dir is not None and dataset_cfg.get('cache_dir') is None:
        dataset_cfg['cache_dir'] = hf_cache_dir
    
    dataset = Animal3DDataset(split=split, **dataset_cfg)
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size if split == 'train' else cfg.training.batch_size,
        shuffle=(split == 'train'),
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=(split == 'train')
    )
    
    return dataloader


def create_optimizer(model, cfg):
    """Create optimizer."""
    opt_cfg = OmegaConf.to_container(cfg.training.optimizer, resolve=True)
    opt_type = opt_cfg.pop('type', 'AdamW')
    
    if opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **opt_cfg)
    elif opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **opt_cfg)
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **opt_cfg)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    return optimizer


def create_scheduler(optimizer, cfg, num_epochs):
    """Create learning rate scheduler."""
    sched_cfg = OmegaConf.to_container(cfg.training.scheduler, resolve=True)
    sched_type = sched_cfg.pop('type', 'cosine')
    
    if sched_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, **sched_cfg
        )
    elif sched_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sched_cfg)
    elif sched_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_cfg)
    else:
        scheduler = None
    
    return scheduler


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_metric_value: float,
    train_loss: float,
    val_loss: float = None,
    val_metrics: dict = None,
    cfg: DictConfig = None,
    is_best: bool = False,
    is_latest: bool = False
):
    """
    Save checkpoint with full state for recovery.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch (will save as epoch + 1 for resume)
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save (can be None)
        best_metric_value: Best metric value seen so far
        train_loss: Last training loss
        val_loss: Last validation loss (optional)
        val_metrics: Last validation metrics (optional)
        cfg: Config to save (optional)
        is_best: Whether this is the best model
        is_latest: Whether this is the latest checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch + 1,  # Next epoch to start from
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_acc': best_metric_value,  # Best metric value (can be accuracy or loss)
        'best_metric_value': best_metric_value,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_metrics': val_metrics if val_metrics else {},
    }
    
    # Save scheduler state if available
    if scheduler is not None:
        try:
            checkpoint['scheduler'] = scheduler.state_dict()
        except AttributeError:
            # Some schedulers don't have state_dict
            checkpoint['scheduler'] = None
    else:
        checkpoint['scheduler'] = None
    
    # Save config if provided
    if cfg is not None:
        checkpoint['config'] = OmegaConf.to_container(cfg, resolve=True)
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save latest checkpoint
    if is_latest:
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: torch.device = None
):
    """
    Load checkpoint and restore model, optimizer, scheduler states.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
    
    Returns:
        Dictionary with loaded checkpoint info:
        - start_epoch: Epoch to start from
        - best_metric_value: Best metric value
        - train_loss: Last training loss
        - val_loss: Last validation loss
        - val_metrics: Last validation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Validate checkpoint keys
        required_keys = ['epoch', 'state_dict']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
        
        # Load model state
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model state from epoch {checkpoint['epoch']}")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded optimizer state")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("Loaded scheduler state")
            except (AttributeError, ValueError) as e:
                print(f"Warning: Could not load scheduler state: {e}")
        
        return {
            'start_epoch': checkpoint['epoch'],
            'best_metric_value': checkpoint.get('best_metric_value', checkpoint.get('best_acc', 0.0)),
            'train_loss': checkpoint.get('train_loss', 0.0),
            'val_loss': checkpoint.get('val_loss', None),
            'val_metrics': checkpoint.get('val_metrics', {}),
        }
    
    except Exception as e:
        print(f"Warning: Failed to load checkpoint from {checkpoint_path}: {e}")
        print("Starting training from scratch...")
        return {
            'start_epoch': 0,
            'best_metric_value': float('inf') if 'loss' in str(e).lower() else 0.0,
            'train_loss': 0.0,
            'val_loss': None,
            'val_metrics': {},
        }


# Parse --token argument from command line before Hydra processes it
# This must be done at module level before @hydra.main decorator is applied
_CLI_TOKEN = None
if '--token' in sys.argv:
    token_idx = sys.argv.index('--token')
    if token_idx + 1 < len(sys.argv):
        _CLI_TOKEN = sys.argv[token_idx + 1]
        # Remove --token and its value from sys.argv so Hydra doesn't see it
        sys.argv.pop(token_idx)
        sys.argv.pop(token_idx)  # Remove the value (now at token_idx after first pop)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # Use token parsed from command line if provided
    cli_token = _CLI_TOKEN
    
    # Override config with command-line token if provided
    if cli_token is not None:
        OmegaConf.set_struct(cfg, False)  # Allow modifications
        if 'huggingface' not in cfg:
            cfg['huggingface'] = {}
        cfg.huggingface.token = cli_token
        if 'data' not in cfg or 'dataset' not in cfg.data:
            if 'data' not in cfg:
                cfg['data'] = {}
            if 'dataset' not in cfg.data:
                cfg.data['dataset'] = {}
        cfg.data.dataset.token = cli_token
        OmegaConf.set_struct(cfg, True)  # Re-enable struct mode
    
    print("=" * 60)
    print("Category-Routed Multi-Regressor Model Training")
    print("=" * 60)
    # Don't print full config to avoid exposing token - print masked version
    cfg_safe = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if 'huggingface' in cfg_safe and 'token' in cfg_safe.huggingface and cfg_safe.huggingface.token:
        cfg_safe.huggingface.token = '***'
    if 'data' in cfg_safe and 'dataset' in cfg_safe.data and 'token' in cfg_safe.data.dataset and cfg_safe.data.dataset.token:
        cfg_safe.data.dataset.token = '***'
    print(f"Config:\n{OmegaConf.to_yaml(cfg_safe)}")
    
    # Set random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    
    # Setup device
    device = setup_device(cfg)
    
    # Create model
    print("\nCreating model...")
    model = create_model(cfg, device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(cfg, split='train')
    val_loader = create_dataloader(cfg, split='val') if hasattr(cfg.data.dataset, 'split') else None
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg, cfg.training.num_epochs)
    
    # Create loss function
    loss_weights = OmegaConf.to_container(cfg.training.get('loss_weights', {}), resolve=True)
    joint_rep_type = cfg.model.regressors.get('joint_rep', '6d')
    num_joints = cfg.model.regressors.get('num_joints', 34)
    criterion = CombinedLoss(loss_weights, joint_rep_type=joint_rep_type, num_joints=num_joints)
    
    # Create evaluator
    eval_cfg = OmegaConf.to_container(cfg.evaluation.metrics, resolve=True)
    evaluator = Evaluator(**eval_cfg)
    
    # Initialize logger
    logging_cfg = cfg.get('logging', {})
    logger = TrainingLogger(
        log_dir=cfg.paths.log_dir,
        log_format=logging_cfg.get('log_format', 'csv'),
        log_gradient_norm=logging_cfg.get('log_gradient_norm', True),
        log_learning_rate=logging_cfg.get('log_learning_rate', True),
        log_iteration_every=logging_cfg.get('log_iteration_every', 1)
    )
    
    # Checkpoint configuration
    checkpoint_cfg = cfg.get('checkpoint', {})
    checkpoint_dir = cfg.paths.checkpoint_dir
    save_checkpoint_every = checkpoint_cfg.get('save_checkpoint_every', 10)
    save_best_model = checkpoint_cfg.get('save_best_model', True)
    resume_from = checkpoint_cfg.get('resume_from', None)
    best_metric = checkpoint_cfg.get('best_metric', 'val_loss')
    
    # Initialize best metric tracking
    if best_metric == 'val_loss':
        best_metric_value = float('inf')
        is_better = lambda new, best: new < best
    else:  # val_acc or other metrics where higher is better
        best_metric_value = 0.0
        is_better = lambda new, best: new > best
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        resume_info = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        start_epoch = resume_info['start_epoch']
        best_metric_value = resume_info['best_metric_value']
        # Estimate global iteration from start_epoch (approximate)
        estimated_iteration = start_epoch * len(train_loader)
        logger.set_global_iteration(estimated_iteration)
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best metric value so far: {best_metric_value:.4f}")
    
    # Training loop
    print("\nStarting training...")
    
    for epoch in range(start_epoch, cfg.training.num_epochs):
        # Training phase
        model.train()
        train_losses = {}
        train_loss_total = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            iteration_start_time = time.time()
            
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            output = model(batch, training=True)
            
            # Compute loss
            losses = criterion(output, batch)
            loss = losses['total']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Compute gradient norm if logging is enabled
            grad_norm = None
            if logging_cfg.get('log_gradient_norm', True):
                # Compute gradient norm before clipping
                grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
                if grads:
                    grad_norm = torch.norm(torch.cat(grads)).item()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr'] if logging_cfg.get('log_learning_rate', True) else None
            
            # Compute iteration time
            iteration_time = time.time() - iteration_start_time
            
            # Log training iteration
            loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            logger.log_training_iteration(
                epoch=epoch,
                batch_idx=batch_idx,
                losses=loss_dict,
                lr=current_lr,
                grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                iteration_time=iteration_time
            )
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in train_losses:
                    train_losses[k] = 0.0
                train_losses[k] += v.item()
            train_loss_total += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Average losses
        num_batches = len(train_loader)
        for k in train_losses:
            train_losses[k] /= num_batches
        train_loss_total /= num_batches
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_losses = {}
            val_loss_total = 0.0
            val_metrics = {}
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                    iteration_start_time = time.time()
                    
                    # Move batch to device
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                    
                    # Forward pass
                    output = model(batch, training=False)
                    
                    # Compute loss
                    losses = criterion(output, batch)
                    loss = losses['total']
                    
                    # Accumulate losses
                    for k, v in losses.items():
                        if k not in val_losses:
                            val_losses[k] = 0.0
                        val_losses[k] += v.item()
                    val_loss_total += loss.item()
                    
                    # Compute metrics
                    metrics = evaluator.evaluate(output, batch)
                    for k, v in metrics.items():
                        if k not in val_metrics:
                            val_metrics[k] = []
                        val_metrics[k].append(v)
                    
                    # Log validation iteration
                    loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
                    iteration_time = time.time() - iteration_start_time
                    logger.log_validation_iteration(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        losses=loss_dict,
                        metrics=metrics,
                        iteration_time=iteration_time
                    )
            
            # Average losses and metrics
            num_batches = len(val_loader)
            for k in val_losses:
                val_losses[k] /= num_batches
            val_loss_total /= num_batches
            
            for k in val_metrics:
                val_metrics[k] = np.mean(val_metrics[k])
            
            # Get metric value for best model tracking
            if best_metric == 'val_loss':
                current_metric_value = val_loss_total
            elif best_metric == 'val_acc' and 'P@0.1' in val_metrics:
                # Use P@0.1 as accuracy proxy if available
                current_metric_value = val_metrics['P@0.1']
            elif best_metric in val_metrics:
                current_metric_value = val_metrics[best_metric]
            else:
                # Fallback to val_loss
                current_metric_value = val_loss_total
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{cfg.training.num_epochs}")
            print(f"Train Loss: {train_loss_total:.4f}")
            print(f"Val Loss: {val_loss_total:.4f}")
            if val_metrics:
                print("Val Metrics:")
                for k, v in val_metrics.items():
                    print(f"  {k}: {v:.4f}")
            
            # Check if this is the best model
            is_best = is_better(current_metric_value, best_metric_value)
            if is_best:
                best_metric_value = current_metric_value
                print(f"New best {best_metric}: {best_metric_value:.4f}")
            
            # Determine metric trend
            metric_trend = 'improving' if is_best else 'stable'
            
            # Log convergence statistics
            logger.log_convergence_stats(
                epoch=epoch,
                best_metric_value=best_metric_value,
                metric_trend=metric_trend
            )
            
            # Save checkpoints
            # Save regular checkpoint every N epochs
            save_regular = (epoch + 1) % save_checkpoint_every == 0
            # Save latest checkpoint always
            is_latest = True
            
            if save_regular or is_best or is_latest:
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric_value=best_metric_value,
                    train_loss=train_loss_total,
                    val_loss=val_loss_total,
                    val_metrics=val_metrics,
                    cfg=cfg,
                    is_best=is_best and save_best_model,
                    is_latest=is_latest
                )
        else:
            # No validation, just print training loss
            print(f"\nEpoch {epoch+1}/{cfg.training.num_epochs}")
            print(f"Train Loss: {train_loss_total:.4f}")
            
            # Log convergence statistics (without validation metrics)
            logger.log_convergence_stats(
                epoch=epoch,
                best_metric_value=best_metric_value,
                metric_trend='stable'
            )
            
            # Save checkpoints even without validation
            save_regular = (epoch + 1) % save_checkpoint_every == 0
            is_latest = True
            
            if save_regular or is_latest:
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric_value=best_metric_value,
                    train_loss=train_loss_total,
                    val_loss=None,
                    val_metrics={},
                    cfg=cfg,
                    is_best=False,
                    is_latest=is_latest
                )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

