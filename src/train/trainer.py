"""
Trainer class for SleepVST models.

Encapsulates the shared training loop used by both pretraining and finetuning:
forward pass (HW+BW or BW-only), one-epoch training, validation, checkpointing,
learning-rate scheduling, best-model tracking, and early stopping.
"""

import copy
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import get_logger
from src.eval.metrics import MetricsTracker, AverageMeter

logger = get_logger(__name__)


class Trainer:
    """Drives the epoch loop for a SleepVST model.

    Handles training, validation, checkpoint save/resume, LR scheduling, and
    early stopping. The model input layout (dual HW+BW vs BW-only) is selected
    by ``use_bw_only``.

    Args:
        model: Model to train (already on the target device).
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: ``ReduceLROnPlateau``-style scheduler stepped on val loss.
        checkpoint_path: Where the best checkpoint is written.
        epochs: Maximum number of epochs.
        use_bw_only: If True, feed only the BW input to the model.
        max_patience: Early-stopping patience (epochs without val-loss improvement).
        wandb_run: Optional Weights & Biases run for metric logging.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        checkpoint_path,
        epochs: int,
        use_bw_only: bool = False,
        max_patience: int = 10,
        wandb_run=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_path = Path(checkpoint_path)
        self.epochs = epochs
        self.use_bw_only = use_bw_only
        self.max_patience = max_patience
        self.wandb_run = wandb_run

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.history = {
            "train_losses": [],
            "train_accs": [],
            "val_losses": [],
            "val_accs": [],
            "val_f1s": [],
        }

    # ==================== Forward pass ====================
    def _forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the model on a batch, returning (logits, labels)."""
        labels = batch["label"].cuda().long()
        x_bw = batch["x_bw"].cuda().float()
        if self.use_bw_only:
            logits = self.model(x_bw)
        else:
            x_hw = batch["x_hw"].cuda().float()
            logits = self.model(x_hw, x_bw)
        return logits, labels

    # ==================== One epoch ====================
    def train_one_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train the model for one epoch, returning (avg_loss, accuracy)."""
        self.model.train()

        losses = AverageMeter()
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            self.optimizer.zero_grad()

            logits, labels = self._forward(batch)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # NaN/Inf check
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(
                    f"Batch {batch_idx}: loss is NaN/Inf, skipping "
                    f"(logits [{logits.min().item():.4f}, {logits.max().item():.4f}], "
                    f"labels [{labels.min().item()}, {labels.max().item()}])"
                )
                continue

            try:
                loss.backward()
            except RuntimeError as e:
                logger.error(f"Batch {batch_idx}: backward failed (loss={loss.item()}): {e}")
                raise

            self.optimizer.step()

            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            losses.update(loss.item(), labels.numel())

        return losses.avg, correct / total if total > 0 else 0.0

    def evaluate(
        self, dataloader: DataLoader
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """Evaluate on a validation loader.

        Returns (loss, accuracy, f1, kappa, confusion_matrix, normalized_cm).
        """
        self.model.eval()

        metrics = MetricsTracker()
        losses = AverageMeter()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                logits, labels = self._forward(batch)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                losses.update(loss.item(), labels.numel())
                preds = logits.argmax(-1)
                metrics.update(preds, labels)

        acc, f1, kappa, cm, cm_norm, cr = metrics.compute()
        return losses.avg, acc, f1, kappa, cm, cm_norm

    # ==================== Checkpointing ====================
    def _save_checkpoint(self, epoch: int):
        """Persist the current best checkpoint with training history."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "train_losses": self.history["train_losses"],
                "train_accs": self.history["train_accs"],
                "val_losses": self.history["val_losses"],
                "val_accs": self.history["val_accs"],
                "val_f1s": self.history["val_f1s"],
            },
            self.checkpoint_path,
        )

    def resume(self):
        """Resume optimizer/scheduler/model state and history from checkpoint."""
        if not self.checkpoint_path.exists():
            logger.info(f"No checkpoint at {self.checkpoint_path}, starting fresh")
            return

        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        for key in self.history:
            self.history[key] = checkpoint.get(key, [])
        logger.info(f"Resumed from epoch {self.start_epoch}")

    def load_pretrained(self, pretrained_checkpoint):
        """Load model weights only, from a pretrained checkpoint (for finetuning)."""
        pretrained_checkpoint = Path(pretrained_checkpoint)
        if not pretrained_checkpoint.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")
        checkpoint = torch.load(pretrained_checkpoint)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded pretrained weights from {pretrained_checkpoint}")

    # ==================== Logging ====================
    def _log_epoch(self, epoch, epoch_time, train_loss, train_acc,
                   val_loss, val_acc, val_f1, val_kappa):
        """Log one epoch's metrics to console and wandb in a compact form."""
        lr = self.optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{self.epochs} ({epoch_time:.1f}s) | "
            f"train: loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val: loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} kappa {val_kappa:.4f}"
        )
        if self.wandb_run:
            self.wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1": val_f1,
                "val/kappa": val_kappa,
                "lr": lr,
            })

    # ==================== Fit ====================
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, label: str = "Training") -> nn.Module:
        """Run the full epoch loop with validation, checkpointing and early stopping.

        After training, the best (lowest val-loss) weights are loaded back into
        the model before it is returned.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            label: Human-readable label for log messages (e.g. "Pretraining").

        Returns:
            The model with the best weights restored.
        """
        logger.info(f"{label} for {self.epochs} epochs (resume from {self.start_epoch})")
        patience = 0
        training_start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start_time = time.time()

            train_loss, train_acc = self.train_one_epoch(train_loader)
            self.history["train_losses"].append(train_loss)
            self.history["train_accs"].append(train_acc)

            val_loss, val_acc, val_f1, val_kappa, _, _ = self.evaluate(val_loader)
            self.history["val_losses"].append(val_loss)
            self.history["val_accs"].append(val_acc)
            self.history["val_f1s"].append(val_f1)

            epoch_time = time.time() - epoch_start_time
            self._log_epoch(epoch + 1, epoch_time, train_loss, train_acc,
                            val_loss, val_acc, val_f1, val_kappa)

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience = 0
                self._save_checkpoint(epoch)
                logger.info(f"  ↳ best model saved (val loss {val_loss:.4f})")
            else:
                patience += 1
                if patience >= self.max_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(patience {patience}/{self.max_patience})"
                    )
                    break

        total_time = time.time() - training_start_time
        logger.info(f"{label} done in {total_time/60:.1f} min (best val loss {self.best_val_loss:.4f})")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.model
