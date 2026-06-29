"""Regenerate full-oracle KVSS test predictions from a saved Random Forest.

This reuses the already-trained SleepVST -> RandomForest transfer head
(``checkpoint/randomforest/SleepVST_rf_model.pkl``) as-is; it does **not**
retrain anything. It exists because the loader's usual label/fps source
(``video_dir/<id>_annotation.json``) is currently unmounted. Instead, per-epoch
ground-truth labels for the 70 KVSS test recordings are read from an existing
experiment-result CSV (``results/predictions/full_oracle_predictions_*.csv``),
and the motion patch size is derived from the signal length rather than the
video frame rate.

"Full oracle" features = SleepVST features from the real ECG (``*_hw.npy``) and
the real respiratory belt (``*_bw.npy``), concatenated with video-derived motion
features, exactly as the RF was trained.

Run:
    python -m src.eval.full_oracle_predict \
        +rf_checkpoint=checkpoint/randomforest/SleepVST_rf_model.pkl \
        +label_csv=results/predictions/full_oracle_predictions_2025-10-29.csv
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.RFClassifier import SleepVSTVideoRF
from src.train.transfer import (
    calculate_per_recording_metrics,
    extract_features_from_loader,
    plot_confusion_matrix,
)
from src.utils.logger import get_logger, setup_logging
from src.utils.utils import setup_device

logger = get_logger(__name__)

CLASS_NAMES = ["Wake", "N1+N2", "N3", "REM"]


class _ListDataset(Dataset):
    """Minimal in-memory dataset over pre-built sample dicts."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _read_ids(txt_path):
    """Read record IDs from a split definition file (stripping ``.h5``)."""
    with open(txt_path) as f:
        return [line.strip().replace(".h5", "") for line in f if line.strip()]


# A_test.csv stores a 5-stage PSG ``Stage`` column; the model is 4-class.
# N1 and N2 are merged into class 1 (matching BaseDataset.parse_csv / .parse_json).
_STAGE_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}


def _labels_by_subject(label_csv):
    """Map ``subject_id -> {epoch_idx: label}`` from a label CSV.

    Two schemas are auto-detected:
      - ``A_test.csv``: ``Case_num`` / ``Start_Epoch`` (1-indexed) / ``Stage``
        (5-stage, mapped to 4 classes). Has gaps where epochs were not scored.
      - prediction CSV (``results/predictions/*.csv``): ``subject_id`` /
        ``epoch_idx`` (0-indexed) / ``ground_truth_label`` (already 4-class).
    Returning ``{epoch_idx: label}`` lets callers place labels at their true
    signal index regardless of gaps.
    """
    df = pd.read_csv(label_csv)
    out = {}
    if "Stage" in df.columns:  # A_test.csv
        df = df.dropna(subset=["Stage"])
        for sid, grp in df.groupby("Case_num"):
            epoch = grp["Start_Epoch"].astype(int) - 1
            label = grp["Stage"].astype(int).map(_STAGE_MAP)
            out[str(sid)] = dict(zip(epoch.tolist(), label.tolist()))
    elif "ground_truth_label" in df.columns:  # prediction CSV
        for sid, grp in df.groupby("subject_id"):
            epoch = grp["epoch_idx"].astype(int)
            label = grp["ground_truth_label"].astype(int)
            out[str(sid)] = dict(zip(epoch.tolist(), label.tolist()))
    else:
        raise ValueError(f"Unrecognized label schema in {label_csv}: {list(df.columns)[:8]}")
    return out


def _epoch_motion(motion, motion_keys, num_epochs):
    """Reduce each motion channel to exactly ``num_epochs`` per-epoch means.

    Uses ``np.array_split`` so every epoch gets one value regardless of how the
    frame count divides — unlike fixed-size patchify, which drops the trailing
    remainder and silently loses the last epoch(s). Then expands each epoch with
    its -3 / current / +3 temporal context, matching
    :meth:`KVSS._create_temporal_features` (column order = past|current|future
    over sorted keys), which is the order the RF was trained on.
    """
    per_epoch = np.stack([
        np.array([seg.mean() for seg in np.array_split(np.asarray(motion[key], dtype=np.float32), num_epochs)])
        for key in motion_keys
    ], axis=1)  # (num_epochs, F)

    N = num_epochs
    padded = np.pad(per_epoch, ((3, 3), (0, 0)), mode="edge")
    past = padded[0:N]
    current = padded[3:N + 3]
    future = padded[6:N + 6]
    return np.concatenate([past, current, future], axis=1).astype(np.float32)  # (N, 3F)


def _build_samples(cfg, ids, labels_by_subject):
    """Build full-oracle transfer samples (real ECG + real belt + motion).

    Mirrors :meth:`KVSS.load_motion_samples`, but (a) sources labels from
    ``A_test.csv`` and (b) aligns motion 1:1 with the recording's epochs so no
    epoch is dropped. Every signal epoch is predicted; epochs with no expert
    label get ``-1`` (excluded from metrics, still written out).
    """
    signal_dir = Path(cfg.data.signal_dir)
    motion_dir = Path(cfg.data.motion_dir)

    samples = []
    skipped = []
    for rid in ids:
        hw_file = signal_dir / f"{rid}_hw.npy"
        bw_file = signal_dir / f"{rid}_bw.npy"
        motion_file = motion_dir / f"{rid}_motion_features.npy"
        if not (hw_file.exists() and bw_file.exists() and motion_file.exists()):
            skipped.append((rid, "missing-file"))
            continue

        x_hw = np.load(hw_file).astype(np.float32)
        x_bw = np.load(bw_file).astype(np.float32)
        # Number of epochs is defined by the signal length (one row per epoch).
        N = min(len(x_hw), len(x_bw))

        motion = np.load(motion_file, allow_pickle=True).item()
        motion_keys = sorted(motion.keys())
        expanded = _epoch_motion(motion, motion_keys, N)

        # Place A_test labels at their true epoch index; -1 where unscored.
        label_map = labels_by_subject.get(rid, {})
        labels = np.full(N, -1, dtype=np.int64)
        for e, lab in label_map.items():
            if 0 <= e < N:
                labels[e] = lab

        samples.append({
            "x_hw": x_hw[:N],
            "x_bw": x_bw[:N],
            "motion": expanded[:N],
            "label": labels[:N],
            "subject_id": rid,
            "start_idx": 0,
        })

    if skipped:
        logger.warning("Skipped %d recordings: %s", len(skipped), skipped)
    n_label = sum(int((s["label"] >= 0).sum()) for s in samples)
    n_total = sum(len(s["label"]) for s in samples)
    logger.info("Built %d samples | %d epochs total, %d labeled", len(samples), n_total, n_label)
    return samples, None


@hydra.main(version_base=None, config_path="../../config", config_name="defaults")
def main(cfg: DictConfig):
    setup_logging(cfg.log.dir, cfg.log.name)
    setup_device(cfg.system.device, cfg.system.get("gpu_ids", "0"))

    rf_checkpoint = cfg.get("rf_checkpoint", "checkpoint/randomforest/SleepVST_rf_model.pkl")
    label_csv = cfg.get("label_csv", "data/kvss/A_test.csv")
    test_ids_file = cfg.get("test_ids", str(Path(cfg.data.root) / "A-test_set.txt"))
    out_dir = Path(cfg.get("out_dir", "results/predictions"))
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = pd.Timestamp.today().strftime("%Y-%m-%d")
    pred_csv = out_dir / cfg.get("out_name", f"full_oracle_predictions_regenerated_{date_tag}.csv")
    metrics_csv = out_dir / cfg.get(
        "metrics_name", f"full_oracle_subject_metrics_regenerated_{date_tag}.csv"
    )

    # Frozen SleepVST feature extractor (classifier head -> Identity).
    model = instantiate(cfg.model).cuda()
    checkpoint_path = cfg.model.finetuned_checkpoint if cfg.model.use_finetuned else cfg.model.checkpoint
    state = torch.load(checkpoint_path, map_location=cfg.system.device)
    model.load_state_dict(state["model_state_dict"])
    model.classifier = nn.Identity().cuda()
    model.eval()
    logger.info("Loaded %s feature extractor from %s", cfg.model.name, checkpoint_path)

    rf_model = SleepVSTVideoRF.load(rf_checkpoint)
    logger.info("Loaded RF (%d features) from %s", rf_model.model_.named_steps["clf"].n_features_in_, rf_checkpoint)

    # Build test samples and extract features with the same windowing used in training.
    ids = _read_ids(test_ids_file)
    labels_by_subject = _labels_by_subject(label_csv)
    samples, _ = _build_samples(cfg, ids, labels_by_subject)
    loader = DataLoader(_ListDataset(samples), batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        test_features, test_labels, test_subject_ids = extract_features_from_loader(
            cfg.seq_len,
            cfg.step_size,
            cfg.data.motion_dim,
            model,
            loader,
            "Extracting full-oracle KVSS test features",
        )

    # Per-recording predictions, per-epoch CSV, and per-subject metrics.
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix as sk_cm,
        f1_score,
        precision_score,
        recall_score,
    )

    csv_rows = []
    subject_metrics = []
    masked_features = []  # per-recording, labeled epochs only (for pooled/per-recording metrics)
    masked_labels = []
    n_pred_total = 0
    for i, (features, labels, sid) in enumerate(zip(test_features, test_labels, test_subject_ids)):
        preds = rf_model.predict(features)          # predict every signal epoch
        n_pred_total += len(preds)

        # Write all predicted epochs; epochs with no expert label use -1 / "Unlabeled".
        for epoch_idx, (gt, pr) in enumerate(zip(labels, preds)):
            csv_rows.append({
                "subject_id": sid,
                "recording_idx": i,
                "epoch_idx": epoch_idx,
                "ground_truth_label": int(gt),
                "ground_truth_name": CLASS_NAMES[int(gt)] if gt >= 0 else "Unlabeled",
                "prediction_label": int(pr),
                "prediction_name": CLASS_NAMES[int(pr)],
            })

        # Metrics only over labeled epochs.
        mask = labels >= 0
        y_m, p_m = labels[mask], preds[mask]
        masked_features.append(features[mask])
        masked_labels.append(y_m)
        cm = sk_cm(y_m, p_m, labels=list(range(len(CLASS_NAMES))))
        per_class_acc = [
            cm[c, c] / cm[c, :].sum() if cm[c, :].sum() > 0 else np.nan
            for c in range(len(CLASS_NAMES))
        ]
        subject_metrics.append({
            "subject_id": sid,
            "recording_idx": i,
            "num_epochs": int(mask.sum()),
            "num_epochs_predicted": len(preds),
            "accuracy": accuracy_score(y_m, p_m),
            "kappa": cohen_kappa_score(y_m, p_m),
            "f1_macro": f1_score(y_m, p_m, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_m, p_m, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_m, p_m, average="macro", zero_division=0),
            "recall_macro": recall_score(y_m, p_m, average="macro", zero_division=0),
            "wake_accuracy": per_class_acc[0],
            "n1n2_accuracy": per_class_acc[1],
            "n3_accuracy": per_class_acc[2],
            "rem_accuracy": per_class_acc[3],
        })

    pd.DataFrame(csv_rows).to_csv(pred_csv, index=False)
    df_metrics = pd.DataFrame(subject_metrics)
    df_metrics.to_csv(metrics_csv, index=False)
    n_labeled = int(sum(len(y) for y in masked_labels))
    logger.info("Saved %d predicted epochs (%d labeled) to %s", n_pred_total, n_labeled, pred_csv)
    logger.info("Saved %d subject metrics to %s", len(subject_metrics), metrics_csv)

    # Overall (pooled) and per-recording metrics over labeled epochs.
    X_test = np.vstack(masked_features)
    y_test = np.hstack(masked_labels)
    overall = rf_model.evaluate(X_test, y_test)
    logger.info(
        "Overall (pooled, %d labeled epochs) | acc %.4f kappa %.4f f1_macro %.4f",
        len(y_test), overall["accuracy"], overall["kappa"], overall["f1"],
    )
    logger.info(
        "Per-subject (mean+/-std) | acc %.4f+/-%.4f | kappa %.4f+/-%.4f | f1_macro %.4f+/-%.4f",
        df_metrics["accuracy"].mean(), df_metrics["accuracy"].std(),
        df_metrics["kappa"].mean(), df_metrics["kappa"].std(),
        df_metrics["f1_macro"].mean(), df_metrics["f1_macro"].std(),
    )
    calculate_per_recording_metrics(masked_features, masked_labels, rf_model)

    report = rf_model.get_detailed_report(X_test, y_test)
    plot_confusion_matrix(
        report["basic_metrics"],
        report["classification_report"],
        class_names=CLASS_NAMES,
        title=f"{cfg.model.name} full-oracle (regenerated)",
        save_path=str(out_dir / f"full_oracle_confusion_matrix_{date_tag}.png"),
    )


if __name__ == "__main__":
    main()
