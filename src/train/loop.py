"""
Training and evaluation setup for SleepVST models.

This module wires datasets, models and the :class:`~src.train.trainer.Trainer`
together. The ``prepare_*`` functions build everything needed for a command and
return a :class:`TrainSetup`; the caller (``cli_train``) then drives
``trainer.fit`` and the test evaluation directly:
- prepare_pretrain: Train on SHHS + MESA datasets
- prepare_finetune: Fine-tune pretrained model on KVSS dataset
- test: Evaluate on test sets with sliding window inference

The reusable epoch loop (training, validation, checkpointing, early stopping)
lives in :class:`~src.train.trainer.Trainer`.
"""

import copy
from collections import Counter, namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.logger import get_logger
from src.eval.metrics import MetricsTracker, AverageMeter
from src.train.trainer import Trainer
from src.data.datasets.SHHS import SHHS
from src.data.datasets.MESA import MESA
from src.data.datasets.KVSS import KVSS

logger = get_logger(__name__)

# Dataset classes are constructed directly (instead of via Hydra's instantiate),
# keyed by the ``name`` field in their config/data/*.yaml.
_DATASET_CLASSES = {"shhs": SHHS, "mesa": MESA, "kvss": KVSS}


# Bundle of everything ``cli_train`` needs to drive a training command:
# a configured Trainer, the train/val loaders, and a list of (name, loader)
# test sets to evaluate after fitting.
TrainSetup = namedtuple("TrainSetup", ["trainer", "train_loader", "val_loader", "test_loaders"])

# Standalone dataset configs live in config/data/<name>.yaml. The composed
# cfg.data only ever holds the single dataset selected by Hydra's `data=` group,
# so pretrain/test load the other datasets' configs directly from disk.
_CONFIG_DATA_DIR = Path(__file__).resolve().parents[2] / "config" / "data"


def _load_dataset_cfg(name):
    """Load a standalone dataset config from ``config/data/<name>.yaml``."""
    cfg_path = _CONFIG_DATA_DIR / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def _mode_name(cfg):
    mode = cfg.get("mode", None)
    if isinstance(mode, str):
        return mode
    if mode is not None and "name" in mode:
        return mode.name
    return None


def _data_cfg_for_split(cfg, data_cfg, split):
    data_cfg = copy.deepcopy(data_cfg)
    data_cfg.split = split
    if "model" in data_cfg:
        data_cfg.model = cfg.model.name
    if "mode" in data_cfg:
        mode_name = _mode_name(cfg)
        if mode_name is not None:
            data_cfg.mode = mode_name
    return data_cfg


def _build_dataset(cfg, data_cfg, split):
    """Construct a dataset class directly from its config fields.

    The class is chosen by the config's ``name`` field; remaining yaml fields are
    passed as keyword arguments (each dataset constructor absorbs unused ones via
    ``**kwargs``). This avoids Hydra's ``instantiate`` indirection.
    """
    data_cfg = _data_cfg_for_split(cfg, data_cfg, split)
    name = data_cfg.name
    if name not in _DATASET_CLASSES:
        raise ValueError(f"Unknown dataset name: {name}")

    kwargs = OmegaConf.to_container(data_cfg, resolve=True)
    kwargs.pop("_target_", None)
    return _DATASET_CLASSES[name](**kwargs)


def _make_dataloader(cfg, dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
    )


def _build_trainer(cfg, model, checkpoint_path, use_bw_only, wandb_run, lr):
    """Create a Trainer with the standard SleepVST optimizer/scheduler setup."""
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=cfg.train.get("weight_decay", 0.01)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    return Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        epochs=cfg.train.epochs,
        use_bw_only=use_bw_only,
        max_patience=cfg.train.get("early_stopping_patience", 10),
        wandb_run=wandb_run,
    )


# ==================== Logging Helpers ====================
def _log_test(wandb_run, name, acc, f1, kappa, cm, cm_norm, cr):
    """Log test results (summary on console, confusion matrices at debug level)."""
    logger.info(f"[{name}] test | acc {acc:.4f} f1 {f1:.4f} kappa {kappa:.4f}")
    logger.debug(f"[{name}] confusion matrix:\n{cm}")
    logger.debug(f"[{name}] confusion matrix (normalized):\n{cm_norm}")
    logger.debug(f"[{name}] classification report:\n{cr}")
    if wandb_run:
        key = name.lower()
        wandb_run.log({
            f"test/{key}/accuracy": acc,
            f"test/{key}/f1": f1,
            f"test/{key}/kappa": kappa,
        })


# ==================== Inference ====================
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


# ==================== Command Setup ====================
def prepare_pretrain(cfg, wandb_run=None) -> TrainSetup:
    """Build the Trainer, loaders and test sets for pretraining (SHHS + MESA).

    The caller is expected to run ``setup.trainer.fit(...)`` followed by
    ``run_test_sets(setup.trainer, setup.test_loaders, wandb_run)``.

    Args:
        cfg: Configuration object
        wandb_run: Weights & Biases run (optional)

    Returns:
        A :class:`TrainSetup` with the trainer and data loaders.
    """
    logger.info("Pretraining: SHHS + MESA")

    # Setup datasets (configs loaded directly from config/data/*.yaml)
    shhs_cfg = _load_dataset_cfg('shhs')
    mesa_cfg = _load_dataset_cfg('mesa')
    shhs_train = _build_dataset(cfg, shhs_cfg, 'train')
    mesa_train = _build_dataset(cfg, mesa_cfg, 'train')
    shhs_val = _build_dataset(cfg, shhs_cfg, 'valid')
    mesa_val = _build_dataset(cfg, mesa_cfg, 'valid')
    shhs_test = _build_dataset(cfg, shhs_cfg, 'test')
    mesa_test = _build_dataset(cfg, mesa_cfg, 'test')

    train_dataset = ConcatDataset([shhs_train, mesa_train])
    val_dataset = ConcatDataset([shhs_val, mesa_val])

    logger.info(
        f"Datasets | train {len(train_dataset)} (SHHS {len(shhs_train)} + MESA {len(mesa_train)}) | "
        f"val {len(val_dataset)} (SHHS {len(shhs_val)} + MESA {len(mesa_val)})"
    )

    # Create dataloaders
    train_loader = _make_dataloader(cfg, train_dataset, cfg.train.batch_size, shuffle=True)
    val_loader = _make_dataloader(cfg, val_dataset, cfg.train.batch_size, shuffle=False)
    test_loaders = [
        ("SHHS", _make_dataloader(cfg, shhs_test, batch_size=1, shuffle=False)),
        ("MESA", _make_dataloader(cfg, mesa_test, batch_size=1, shuffle=False)),
    ]

    # Initialize model
    model_name = cfg.model.name
    use_bw_only = 'BW' in model_name
    logger.info(f"Model: {model_name}")

    model = instantiate(cfg.model).cuda()

    # Setup trainer
    checkpoint_path = Path(cfg.train.checkpoint_dir) / f"{cfg.train.log_name}.pth"
    trainer = _build_trainer(cfg, model, checkpoint_path, use_bw_only, wandb_run, cfg.train.lr)

    # Resume from checkpoint if not training from scratch
    if not cfg.train.get('from_scratch', True):
        trainer.resume()

    return TrainSetup(trainer, train_loader, val_loader, test_loaders)


def prepare_finetune(cfg, wandb_run=None) -> TrainSetup:
    """Build the Trainer, loaders and test set for finetuning on KVSS.

    The pretrained weights are loaded into the model before returning, so the
    caller only needs to run ``setup.trainer.fit(...)`` then the test evaluation.

    Args:
        cfg: Configuration object
        wandb_run: Weights & Biases run (optional)

    Returns:
        A :class:`TrainSetup` with the trainer and data loaders.
    """
    logger.info("Finetuning: KVSS")

    model_name = cfg.model.name
    use_bw_only = 'BW' in model_name
    logger.info(f"Model: {model_name}")

    model = instantiate(cfg.model).cuda()

    # Setup KVSS dataset (from the composed cfg.data, honoring CLI overrides)
    train_dataset = _build_dataset(cfg, cfg.data, 'train')
    val_dataset = _build_dataset(cfg, cfg.data, 'valid')
    test_dataset = _build_dataset(cfg, cfg.data, 'test')

    train_loader = _make_dataloader(cfg, train_dataset, cfg.train.batch_size, shuffle=True)
    val_loader = _make_dataloader(cfg, val_dataset, cfg.train.batch_size, shuffle=False)
    test_loaders = [("KVSS", _make_dataloader(cfg, test_dataset, batch_size=1, shuffle=False))]

    # Setup trainer (with lower learning rate for finetuning)
    finetune_lr = cfg.train.get('finetune_lr', cfg.train.lr / 10)
    logger.info(f"Finetune lr: {finetune_lr} (base lr: {cfg.train.lr})")

    checkpoint_path = Path(cfg.train.checkpoint_dir) / f"{cfg.train.log_name}_finetuned.pth"
    trainer = _build_trainer(cfg, model, checkpoint_path, use_bw_only, wandb_run, finetune_lr)

    # Load pretrained weights before finetuning
    trainer.load_pretrained(cfg.train.pretrained_checkpoint)

    return TrainSetup(trainer, train_loader, val_loader, test_loaders)


def run_test_sets(trainer: Trainer, test_loaders, wandb_run=None):
    """Run sliding-window inference over each ``(name, loader)`` test set and log results."""
    for name, loader in test_loaders:
        loss, acc, f1, kappa, cm, cm_norm, cr = sliding_window_inference(
            trainer.model, loader, trainer.criterion, use_bw_only=trainer.use_bw_only
        )
        _log_test(wandb_run, name, acc, f1, kappa, cm, cm_norm, cr)


def test(cfg, wandb_run=None):
    """Test model on specified datasets.

    Args:
        cfg: Configuration object
        wandb_run: Weights & Biases run (optional)
    """
    logger.info("Testing")

    # Load model
    model_name = cfg.model.name
    use_bw_only = 'BW' in model_name
    logger.info(f"Model: {model_name}")

    model = instantiate(cfg.model).cuda()

    checkpoint_path = Path(cfg.test.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    criterion = nn.CrossEntropyLoss().cuda()

    # Test on specified datasets
    test_datasets = cfg.test.get('datasets', ['kvss'])

    for dataset_name in test_datasets:
        if dataset_name in ('kvss', 'shhs', 'mesa'):
            test_data = _build_dataset(cfg, _load_dataset_cfg(dataset_name), 'test')
        else:
            logger.warning(f"Unknown dataset: {dataset_name}, skipping")
            continue

        test_loader = _make_dataloader(cfg, test_data, batch_size=1, shuffle=False)

        loss, acc, f1, kappa, cm, cm_norm, cr = sliding_window_inference(
            model, test_loader, criterion, use_bw_only=use_bw_only
        )
        _log_test(wandb_run, dataset_name.upper(), acc, f1, kappa, cm, cm_norm, cr)
