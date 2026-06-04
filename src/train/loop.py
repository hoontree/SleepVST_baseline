"""
Training and evaluation loops for SleepVST models.

This module provides:
- Pretrain: Train on SHHS + MESA datasets
- Finetune: Fine-tune pretrained model on KVSS dataset
- Test: Evaluate on test sets with sliding window inference
"""

import os
import copy
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.eval.metrics import MetricsTracker, AverageMeter
from src.models.SleepVST import SleepVST, SleepVST_BW
from src.data.datasets.SHHS import SHHS
from src.data.datasets.MESA import MESA
from src.data.datasets.KVSS import KVSS
from src.data.datasets.KVSS import KVSSDataModule

# ==================== Model Registry ====================
MODEL_REGISTRY = {
    "SleepVST": SleepVST,
    "SleepVST_BW": SleepVST_BW
}


# ==================== Training Functions ====================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    use_bw_only: bool = False
) -> Tuple[float, float]:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        use_bw_only: If True, use only BW input (for SleepVST_BW)
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    losses = AverageMeter()
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        labels = batch['label'].cuda().long()  # (B, 240)
        
        optimizer.zero_grad()
        
        if use_bw_only:
            x_bw = batch['x_bw'].cuda().float()  # (B, 240, 150)
            logits = model(x_bw)  # (B, 240, num_classes)
        else:
            x_hw = batch['x_hw'].cuda().float()  # (B, 240, 300)
            x_bw = batch['x_bw'].cuda().float()  # (B, 240, 150)
            logits = model(x_hw, x_bw)  # (B, 240, num_classes)

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # NaN/Inf check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Batch {batch_idx}] Loss is NaN or Inf! Skipping batch.")
            print(f"  logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"  labels range: [{labels.min().item()}, {labels.max().item()}]")
            continue
        
        try:
            loss.backward()
        except RuntimeError as e:
            print(f"[Batch {batch_idx}] RuntimeError during backward: {e}")
            print(f"  Loss value: {loss.item()}")
            raise

        optimizer.step()
        
        preds = logits.argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        losses.update(loss.item(), labels.numel())
        
    return losses.avg, correct / total if total > 0 else 0.0


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    use_bw_only: bool = False
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        criterion: Loss function
        use_bw_only: If True, use only BW input
        
    Returns:
        Tuple of (loss, accuracy, f1, kappa, confusion_matrix, normalized_cm)
    """
    model.eval()
    
    metrics = MetricsTracker()
    losses = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = batch['label'].cuda().long()
            
            if use_bw_only:
                x_bw = batch['x_bw'].cuda().float()
                logits = model(x_bw)
            else:
                x_hw = batch['x_hw'].cuda().float()
                x_bw = batch['x_bw'].cuda().float()
                logits = model(x_hw, x_bw)
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            losses.update(loss.item(), labels.numel())
            preds = logits.argmax(-1)
            metrics.update(preds, labels)

    val_loss = losses.avg
    acc, f1, kappa, cm, cm_norm, cr = metrics.compute()

    return val_loss, acc, f1, kappa, cm, cm_norm


def sliding_window_inference(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    window_size: int = 240,
    step_size: int = 60,
    use_bw_only: bool = False
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, str]:
    """Sliding window inference with majority voting.
    
    Args:
        model: Model to use for inference
        dataloader: Test data loader (batch_size=1)
        criterion: Loss function
        window_size: Window size for sliding window
        step_size: Step size for sliding window
        use_bw_only: If True, use only BW input
        
    Returns:
        Tuple of (loss, accuracy, f1, kappa, cm, normalized_cm, classification_report)
    """
    model.eval()
    metrics = MetricsTracker()
    losses = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Sliding Inference", leave=False)):
            labels = batch['label'].squeeze(0).cpu().numpy()  # (T,)
            
            if use_bw_only:
                x_bw = batch['x_bw'].squeeze(0).cuda()  # (T, 150)
                T = x_bw.shape[0]
                
                # Handle short sequences
                if T < window_size:
                    if T < 7:  # Too small for convolution
                        all_preds.append(np.zeros_like(labels))
                        all_labels.append(labels)
                        continue
                    
                    bw_chunk = x_bw.unsqueeze(0)
                    logits = model(bw_chunk)
                    preds = logits.squeeze(0).argmax(-1).cpu().numpy()
                    
                    batch_loss = criterion(logits.view(-1, logits.size(-1)), 
                                          torch.tensor(labels).cuda().view(-1).long())
                    losses.update(batch_loss.item(), len(labels))
                    
                    all_preds.append(preds)
                    all_labels.append(labels)
                    continue
                
                # Sliding window
                batch_preds = [[] for _ in range(T)]
                
                for start in range(0, T - window_size + 1, step_size):
                    end = start + window_size
                    bw_chunk = x_bw[start:end].unsqueeze(0)
                    logits = model(bw_chunk)
                    preds = logits.squeeze(0).argmax(-1).cpu().numpy()
                    
                    for i in range(window_size):
                        batch_preds[start + i].append(preds[i])
            else:
                x_hw = batch['x_hw'].squeeze(0).cuda()  # (T, 300)
                x_bw = batch['x_bw'].squeeze(0).cuda()  # (T, 150)
                T = x_hw.shape[0]
                
                # Handle short sequences
                if T < window_size:
                    if T < 7:
                        all_preds.append(np.zeros_like(labels))
                        all_labels.append(labels)
                        continue
                    
                    hw_chunk = x_hw.unsqueeze(0)
                    bw_chunk = x_bw.unsqueeze(0)
                    logits = model(hw_chunk, bw_chunk)
                    preds = logits.squeeze(0).argmax(-1).cpu().numpy()
                    
                    batch_loss = criterion(logits.view(-1, logits.size(-1)), 
                                          torch.tensor(labels).cuda().view(-1).long())
                    losses.update(batch_loss.item(), len(labels))
                    
                    all_preds.append(preds)
                    all_labels.append(labels)
                    continue
                
                # Sliding window
                batch_preds = [[] for _ in range(T)]
                
                for start in range(0, T - window_size + 1, step_size):
                    end = start + window_size
                    hw_chunk = x_hw[start:end].unsqueeze(0)
                    bw_chunk = x_bw[start:end].unsqueeze(0)
                    logits = model(hw_chunk, bw_chunk)
                    preds = logits.squeeze(0).argmax(-1).cpu().numpy()
                    
                    for i in range(window_size):
                        batch_preds[start + i].append(preds[i])
            
            # Majority voting
            final_preds = np.zeros(T, dtype=int)
            for i in range(T):
                if len(batch_preds[i]) == 0:
                    non_empty_preds = [p for sublist in batch_preds if sublist for p in sublist]
                    if non_empty_preds:
                        final_preds[i] = np.argmax(np.bincount(non_empty_preds))
                    else:
                        final_preds[i] = 0
                else:
                    counter = Counter(batch_preds[i])
                    final_preds[i] = counter.most_common(1)[0][0]
            
            all_preds.append(final_preds)
            all_labels.append(labels)
    
    # Combine all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    metrics.update(torch.tensor(all_preds).cuda(), torch.tensor(all_labels).cuda().long())
    
    val_loss = losses.avg
    acc, f1, kappa, cm, cm_norm, cr = metrics.compute()
    
    return val_loss, acc, f1, kappa, cm, cm_norm, cr


# ==================== Main Training Loops ====================
def pretrain(cfg, logger, wandb_run=None):
    """Pretrain model on SHHS + MESA datasets.
    
    Args:
        cfg: Configuration object
        logger: Logger instance
        wandb_run: Weights & Biases run (optional)
    """
    logger.info("="*60)
    logger.info("PRETRAINING MODE: SHHS + MESA")
    logger.info("="*60)
    
    # Setup datasets
    logger.info("Loading SHHS and MESA datasets...")
    
    shhs_train = SHHS(cfg.data.shhs, split='train')
    mesa_train = MESA(cfg.data.mesa, split='train')
    shhs_val = SHHS(cfg.data.shhs, split='val')
    mesa_val = MESA(cfg.data.mesa, split='val')
    shhs_test = SHHS(cfg.data.shhs, split='test')
    mesa_test = MESA(cfg.data.mesa, split='test')
    
    train_dataset = ConcatDataset([shhs_train, mesa_train])
    val_dataset = ConcatDataset([shhs_val, mesa_val])
    
    logger.info(f"Training samples: {len(train_dataset)} (SHHS: {len(shhs_train)}, MESA: {len(mesa_train)})")
    logger.info(f"Validation samples: {len(val_dataset)} (SHHS: {len(shhs_val)}, MESA: {len(mesa_val)})")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory
    )
    shhs_test_loader = DataLoader(shhs_test, batch_size=1, shuffle=False, 
                                   num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory)
    mesa_test_loader = DataLoader(mesa_test, batch_size=1, shuffle=False,
                                   num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory)
    
    # Initialize model
    model_name = cfg.model.name
    use_bw_only = 'BW' in model_name
    logger.info(f"Initializing model: {model_name}")
    
    model = MODEL_REGISTRY[model_name](cfg.model).cuda()
    
    # Setup training
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.get('weight_decay', 0.01))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Load checkpoint if exists and not training from scratch
    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{cfg.train.log_name}.pth"
    
    start_epoch = 0
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, train_accs = [], []
    val_losses, val_accs, val_f1s = [], [], []
    
    if not cfg.train.get('from_scratch', True) and checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        val_losses = checkpoint.get('val_losses', [])
        val_accs = checkpoint.get('val_accs', [])
        val_f1s = checkpoint.get('val_f1s', [])
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info(f"Starting training for {cfg.train.epochs} epochs")
    patience = 0
    max_patience = cfg.train.get('early_stopping_patience', 10)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, cfg.train.epochs):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{cfg.train.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, use_bw_only)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_f1, val_kappa, val_cm, val_cm_norm = evaluate(
            model, val_loader, criterion, use_bw_only
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")
        
        # Log to wandb
        if wandb_run:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1": val_f1,
                "val/kappa": val_kappa,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'val_f1s': val_f1s,
            }, checkpoint_path)
            logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience += 1
            logger.info(f"  Patience: {patience}/{max_patience}")
            
            if patience >= max_patience:
                logger.info("Early stopping triggered!")
                break
    
    total_training_time = time.time() - training_start_time
    logger.info(f"\nPretraining completed in {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model for testing")
    
    # Test on SHHS and MESA
    logger.info("\n" + "="*60)
    logger.info("TESTING ON SHHS")
    logger.info("="*60)
    loss, acc, f1, kappa, cm, cm_norm, cr = sliding_window_inference(
        model, shhs_test_loader, criterion, use_bw_only=use_bw_only
    )
    logger.info(f"SHHS Test - Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Normalized CM:\n{cm_norm}")
    logger.info(f"Classification Report:\n{cr}")
    
    if wandb_run:
        wandb_run.log({"test/shhs_acc": acc, "test/shhs_f1": f1, "test/shhs_kappa": kappa})
    
    logger.info("\n" + "="*60)
    logger.info("TESTING ON MESA")
    logger.info("="*60)
    loss, acc, f1, kappa, cm, cm_norm, cr = sliding_window_inference(
        model, mesa_test_loader, criterion, use_bw_only=use_bw_only
    )
    logger.info(f"MESA Test - Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Normalized CM:\n{cm_norm}")
    logger.info(f"Classification Report:\n{cr}")
    
    if wandb_run:
        wandb_run.log({"test/mesa_acc": acc, "test/mesa_f1": f1, "test/mesa_kappa": kappa})
    
    logger.info("\nPretraining process completed!")
    return model


def finetune(cfg, logger, wandb_run=None):
    """Fine-tune pretrained model on KVSS dataset.
    
    Args:
        cfg: Configuration object
        logger: Logger instance
        wandb_run: Weights & Biases run (optional)
    """
    logger.info("="*60)
    logger.info("FINETUNING MODE: KVSS")
    logger.info("="*60)
    
    # Load pretrained model
    model_name = cfg.model.name
    use_bw_only = 'BW' in model_name
    logger.info(f"Initializing model: {model_name}")
    
    model = MODEL_REGISTRY[model_name](cfg.model).cuda()
    pretrained_checkpoint = Path(cfg.train.pretrained_checkpoint)
    if not pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")
    
    logger.info(f"Loading pretrained weights from {pretrained_checkpoint}")
    checkpoint = torch.load(pretrained_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("✓ Pretrained weights loaded successfully")
    
    # Setup KVSS dataset
    logger.info("Loading KVSS datasets...")
    
    cfg_train = cfg.copy()
    cfg_train.data.split = 'train'
    kvss_train_module = KVSSDataModule(cfg_train)

    cfg_val = cfg.copy()
    cfg_val.data.split = 'valid'
    kvss_val_module = KVSSDataModule(cfg_val)

    cfg_test = cfg.copy()
    cfg_test.data.split = 'test'
    kvss_test_module = KVSSDataModule(cfg_test)
    
    train_loader = kvss_train_module.get_dataloader()
    val_loader = kvss_val_module.get_dataloader()
    test_loader = kvss_test_module.get_dataloader()
    
    # Setup training (with lower learning rate for finetuning)
    finetune_lr = cfg.train.get('finetune_lr', cfg.train.lr / 10)
    logger.info(f"Using learning rate: {finetune_lr} (base lr: {cfg.train.lr})")
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=cfg.train.get('weight_decay', 0.01))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Checkpoint setup
    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{cfg.train.log_name}_finetuned.pth"
    
    start_epoch = 0
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, train_accs = [], []
    val_losses, val_accs, val_f1s = [], [], []
    
    # Training loop
    logger.info(f"Starting fine-tuning for {cfg.train.epochs} epochs")
    patience = 0
    max_patience = cfg.train.get('early_stopping_patience', 10)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, cfg.train.epochs):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{cfg.train.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, use_bw_only)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_f1, val_kappa, val_cm, val_cm_norm = evaluate(
            model, val_loader, criterion, use_bw_only
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")
        
        # Log to wandb
        if wandb_run:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1": val_f1,
                "val/kappa": val_kappa,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'val_f1s': val_f1s,
            }, checkpoint_path)
            logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience += 1
            logger.info(f"  Patience: {patience}/{max_patience}")
            
            if patience >= max_patience:
                logger.info("Early stopping triggered!")
                break
    
    total_training_time = time.time() - training_start_time
    logger.info(f"\nFine-tuning completed in {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model for testing")
    
    # Test on KVSS
    logger.info("\n" + "="*60)
    logger.info("TESTING ON KVSS")
    logger.info("="*60)
    loss, acc, f1, kappa, cm, cm_norm, cr = sliding_window_inference(
        model, test_loader, criterion, use_bw_only=use_bw_only
    )
    logger.info(f"KVSS Test - Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Normalized CM:\n{cm_norm}")
    logger.info(f"Classification Report:\n{cr}")
    
    if wandb_run:
        wandb_run.log({"test/kvss_acc": acc, "test/kvss_f1": f1, "test/kvss_kappa": kappa})
    
    logger.info("\nFine-tuning process completed!")
    return model


def test(cfg, logger):
    """Test model on specified datasets.
    
    Args:
        cfg: Configuration object
        logger: Logger instance
    """
    logger.info("="*60)
    logger.info("TESTING MODE")
    logger.info("="*60)
    
    # Load model
    model_name = cfg.model.name
    use_bw_only = 'BW' in model_name
    logger.info(f"Initializing model: {model_name}")
    
    model = MODEL_REGISTRY[model_name](cfg.model).cuda()
    
    checkpoint_path = Path(cfg.test.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("✓ Checkpoint loaded successfully")
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Test on specified datasets
    test_datasets = cfg.test.get('datasets', ['kvss'])
    
    for dataset_name in test_datasets:
        logger.info("\n" + "="*60)
        logger.info(f"TESTING ON {dataset_name.upper()}")
        logger.info("="*60)
        
        if dataset_name == 'kvss':
            test_data = KVSS(cfg.data.kvss, split='test')
        elif dataset_name == 'shhs':
            test_data = SHHS(cfg.data.shhs, split='test')
        elif dataset_name == 'mesa':
            test_data = MESA(cfg.data.mesa, split='test')
        else:
            logger.warning(f"Unknown dataset: {dataset_name}, skipping...")
            continue
        
        test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.system.num_workers,
            pin_memory=cfg.system.pin_memory
        )
        
        loss, acc, f1, kappa, cm, cm_norm, cr = sliding_window_inference(
            model, test_loader, criterion, use_bw_only=use_bw_only
        )
        
        logger.info(f"{dataset_name.upper()} Test - Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Normalized CM:\n{cm_norm}")
        logger.info(f"Classification Report:\n{cr}")
    
    logger.info("\nTesting completed!")
    
    