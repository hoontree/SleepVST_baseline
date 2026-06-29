import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from src.data.base_datamodule import BaseDataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from hydra.utils import instantiate

from logging import getLogger

logger = getLogger(__name__)


def transfer_to_video(cfg):
    """Transfer a pretrained SleepVST encoder to video via a Random Forest head.

    The SleepVST classifier head is replaced with ``nn.Identity`` to use the model
    as a frozen feature extractor. Per-epoch features (signal + motion) are
    extracted for the KVSS splits and a Random Forest is fit / evaluated on them.

    Behaviour depends on ``cfg.transfer.classifier.mode``:
        - ``fit``: train the RF and save it.
        - ``fit_test``: train, save, then evaluate on the test split.
        - otherwise (``test``): load a saved RF and evaluate on the test split.

    Args:
        cfg: Config with model, data, system, and mode settings.

    Returns:
        The fitted or loaded ``SleepVSTVideoRF`` model.
    """
    
    from src.models.RFClassifier import SleepVSTVideoRF, SleepVSTVideoRFConfig
    # Load the SleepVST model and checkpoint
    model = instantiate(cfg.model).cuda()
    checkpoint_path = cfg.model.finetuned_checkpoint if cfg.model.use_finetuned else cfg.model.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=cfg.system.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded {cfg.model.name} from {checkpoint_path}")

    # Replace classifier head with identity to use the model as a feature extractor
    model.classifier = nn.Identity().cuda()
    feature_extractor = model
    feature_extractor.eval()

    if cfg.transfer.classifier.mode in ('fit', 'fit_test'):
        cfg_train = cfg.copy()
        cfg_train.data.split = 'train'
        train_dataset: BaseDataset = instantiate(cfg.data, split='train')

        cfg_val = cfg.copy()
        cfg_val.data.split = 'valid'
        val_dataset: BaseDataset = instantiate(cfg.data, split='valid')

        # Build feature names (signal features + motion features)
        motion_feature_keys = train_dataset.get_motion_feature_keys()
        signal_features = [f"signal_feat_{i}" for i in range(cfg.d_model)]
        if motion_feature_keys is not None:
            motion_features = motion_feature_keys
            logger.info(f"Loaded {len(motion_feature_keys)} motion feature keys")
        else:
            motion_features = [f"motion_feat_{i}" for i in range(cfg.motion_dim)]
        feature_names = signal_features + motion_features
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.get("batch_size", 1),
            num_workers=cfg.data.get("num_workers", 8),
            shuffle=cfg.data.get("shuffle", True),
            pin_memory=cfg.data.get("pin_memory", True),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.get("batch_size", 1),
            num_workers=cfg.data.get("num_workers", 8),
            shuffle=cfg.data.get("shuffle", False),
            pin_memory=cfg.data.get("pin_memory", True),
        )
        
        rf_config = SleepVSTVideoRFConfig(
            n_estimators=300,
            search_params=False,
            verbose=1
        )
        rf_model = SleepVSTVideoRF(rf_config)
        
        with torch.no_grad():
            train_features, train_labels, train_subject_ids = extract_features_from_loader(
                cfg_train.seq_len,
                cfg_train.step_size,
                cfg_train.data.motion_dim,
                feature_extractor,
                train_loader,
                "Extracting features from KVSS train set",
            )

            val_features, val_labels, val_subject_ids = extract_features_from_loader(
                cfg_val.seq_len,
                cfg_val.step_size,
                cfg_val.data.motion_dim,
                feature_extractor,
                val_loader,
                "Extracting features from KVSS validation set",
            )
            
        X_train = np.vstack([features for features in train_features])
        X_val = np.vstack([features for features in val_features])

        y_train = np.hstack([labels for labels in train_labels])
        y_val = np.hstack([labels for labels in val_labels])

        logger.info(
            f"Train: {len(y_train):,} epochs from {len(train_features)} recordings {X_train.shape} | "
            f"Val: {len(y_val):,} epochs from {len(val_features)} recordings {X_val.shape}"
        )

        rf_model.fit(X_train, y_train, feature_names=feature_names)

        # Basic evaluation (overall metrics only)
        train_results = rf_model.evaluate(X_train, y_train)
        val_results = rf_model.evaluate(X_val, y_val)
        logger.info(
            f"Overall | train: acc {train_results['accuracy']:.4f} kappa {train_results['kappa']:.4f} "
            f"f1 {train_results['f1']:.4f} | "
            f"val: acc {val_results['accuracy']:.4f} kappa {val_results['kappa']:.4f} "
            f"f1 {val_results['f1']:.4f}"
        )

        # Per-recording metrics (paper equations 7, 8)
        logger.info("Train set:")
        calculate_per_recording_metrics(train_features, train_labels, rf_model)
        logger.info("Validation set:")
        calculate_per_recording_metrics(val_features, val_labels, rf_model)

        # Save the RF model with motion feature keys as metadata
        if cfg.model.use_finetuned:
            save_path = f"./checkpoint/randomforest/{cfg.model.name}_rf_model_used_finetuned_{cfg.data.data_source}.pkl"
        else:
            save_path = f"./checkpoint/randomforest/{cfg.model.name}_rf_model_{cfg.data.data_source}.pkl"
        rf_model.save(save_path, metadata={'motion_feature_keys': motion_feature_keys})
        logger.info(f"Saved RF model to {save_path}")

        if cfg.transfer.classifier.mode == 'fit_test':
            _run_test_evaluation(cfg, feature_extractor, rf_model)
    else:
        # Test-only mode: load a saved RF model and evaluate
        if cfg.model.use_finetuned:
            rf_model = SleepVSTVideoRF.load(f"./checkpoint/randomforest/{cfg.model.name}_rf_model_used_finetuned.pkl")
        else:
            rf_model = SleepVSTVideoRF.load(f"./checkpoint/randomforest/{cfg.model.name}_rf_model.pkl")

        _run_test_evaluation(cfg, feature_extractor, rf_model)

    return rf_model


def _make_dataloader(data_cfg, dataset):
    return DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 1),
        num_workers=data_cfg.get("num_workers", 8),
        shuffle=data_cfg.get("shuffle", True),
        pin_memory=data_cfg.get("pin_memory", True),
    )


def extract_features_from_loader(seq_len, step_size, motion_dim, feature_extractor: torch.nn.Module, data_loader, desc, use_bw_only=False):
    """Extract per-epoch features for every recording in a dataloader.

    Each sliding window is encoded once; for every time step the feature is taken
    from the window whose center is closest to that step, then concatenated with
    the step's motion vector. Supports both single-input (BW) and dual-input
    (HW + BW) models depending on ``cfg.model.name``.

    Args:
        seq_len: Length of each sliding window.
        step_size: Step size for sliding window.
        motion_dim: Dimensionality of the motion features.
        feature_extractor: Model with its classifier head replaced by ``nn.Identity``.
        data_loader: Dataloader yielding batches with x_hw/x_bw/motion/label/subject_id.
        desc: tqdm progress bar description.
        use_bw_only: Whether to use only the black-and-white input.

        feature_extractor: Model with its classifier head replaced by ``nn.Identity``.
        data_loader: Dataloader yielding batches with x_hw/x_bw/motion/label/subject_id.
        desc: tqdm progress bar description.
        use_bw_only: Whether to use only the black-and-white input.

    Returns:
        Tuple ``(features, labels, subject_ids)`` as lists with one entry per
        recording: features are (T, d_model + motion_dim), labels are (T,).
    """
    feature_extractor.eval()
    window_size = seq_len
    step = step_size
    center_index = (window_size - 1) // 2
    use_bw_only = use_bw_only or 'x_hw' not in data_loader.dataset[0]
    # Signal-feature width = encoder d_model (the classifier head is nn.Identity),
    # which is independent of the sliding-window length.
    signal_dim = int(getattr(feature_extractor, 'd_model'))

    all_final_features = []
    all_labels = []
    all_subject_ids = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            if not use_bw_only and ('x_hw' not in batch or 'x_bw' not in batch):
                raise KeyError("Batch must contain both 'x_hw' and 'x_bw' keys for dual-input models.")

            x_bw = batch['x_bw'].cuda().float()        # (B, N, 150)
            motion = batch['motion'].float().cpu().numpy()  # (B, N, motion_dim)
            labels = batch['label'].cpu().numpy()      # (B, N)
            subject_ids = batch['subject_id']          # (B,)
            B, N, _ = x_bw.shape
            T = min(N, batch['label'].shape[1])
            x_hw = batch['x_hw'].cuda().float() if not use_bw_only else None

            # Encode each sliding window once; force a final window to cover the tail
            starts = list(range(0, max(T - window_size + 1, 0), step))
            if T >= window_size and (not starts or starts[-1] != T - window_size):
                starts.append(T - window_size)

            window_features = {}
            for start in starts:
                sl = slice(start, start + window_size)
                feats = feature_extractor(x_bw[:, sl]) if use_bw_only \
                    else feature_extractor(x_hw[:, sl], x_bw[:, sl])
                window_features[start] = feats.cpu().numpy()  # (B, W, d_model)

            # For each step, take the feature from the window whose center is closest,
            # then concatenate with that step's motion vector
            batch_final_features = np.zeros((B, T, signal_dim + motion_dim), dtype=np.float32)
            batch_labels = np.zeros((B, T), dtype=np.int64)

            for t in range(T):
                best_start = _closest_window(window_features, t, window_size, center_index)
                if best_start is not None:
                    feature_vec = window_features[best_start][:, t - best_start]  # (B, d_model)
                    batch_final_features[:, t, :] = np.concatenate([feature_vec, motion[:, t, :]], axis=1)
                batch_labels[:, t] = labels[:, t]

            for i in range(B):
                all_final_features.append(batch_final_features[i])
                all_labels.append(batch_labels[i])
                all_subject_ids.append(subject_ids[i])

    return all_final_features, all_labels, all_subject_ids


def _closest_window(window_features, t, window_size, center_index):
    """Return the start index of the window covering step ``t`` whose center is
    closest to ``t``, or ``None`` if no window covers it.

    On ties the earliest window wins (dict keys are in increasing start order).
    """
    best_start = None
    min_distance = float('inf')
    for start in window_features:
        if start <= t < start + window_size:
            distance = abs((t - start) - center_index)
            if distance < min_distance:
                min_distance = distance
                best_start = start
    return best_start


def analyze_motion_features(X_train, cfg):
    """Diagnose the motion-feature block of a stacked feature matrix.

    Splits each row into visual features (first ``cfg.d_model`` dims) and motion
    features (remaining ``cfg.motion_dim`` dims), then reports degenerate
    features (zero/low variance, constant), the visual-vs-motion scale ratio, and
    NaN/Inf counts. Output is logged at debug level since it is purely diagnostic.

    Args:
        X_train: Feature matrix of shape (N, d_model + motion_dim).
        cfg: Config providing ``d_model`` and ``motion_dim``.

    Returns:
        Dict with zero-variance / valid / constant feature indices and the
        visual-to-motion standard-deviation ratio.
    """
    visual_data = X_train[:, :cfg.d_model]
    motion_data = X_train[:, cfg.d_model:]

    motion_vars = np.var(motion_data, axis=0)
    motion_mins = np.min(motion_data, axis=0)
    motion_maxs = np.max(motion_data, axis=0)

    zero_var_indices = np.where(motion_vars == 0)[0]
    very_low_var_indices = np.where(motion_vars < 1e-10)[0]
    constant_indices = np.where(motion_mins == motion_maxs)[0]
    valid_var_indices = np.where(motion_vars > 1e-6)[0]

    scale_ratio = np.std(visual_data) / np.std(motion_data) if np.std(motion_data) > 0 else float('inf')
    nan_count = int(np.sum(np.isnan(motion_data)))
    inf_count = int(np.sum(np.isinf(motion_data)))

    logger.debug(
        f"Motion features: {cfg.motion_dim} dims | "
        f"zero-var {len(zero_var_indices)}, low-var {len(very_low_var_indices)}, "
        f"constant {len(constant_indices)}, valid {len(valid_var_indices)}"
    )
    logger.debug(f"Visual/motion std ratio: {scale_ratio:.2f} | NaN {nan_count}, Inf {inf_count}")
    if scale_ratio > 100:
        logger.warning("Visual features dominate motion features in scale; consider normalization")

    return {
        'zero_var_indices': zero_var_indices,
        'valid_var_indices': valid_var_indices,
        'constant_indices': constant_indices,
        'scale_ratio': scale_ratio
    }


def calculate_per_recording_metrics(features_list, labels_list, rf_model):
    """Compute total and per-recording accuracy / Cohen's kappa.

    Implements the paper's Equations 7 (total, pooled over all epochs) and 8
    (mean over per-recording scores).

    Args:
        features_list: Per-recording feature arrays [(T1, D), (T2, D), ...].
        labels_list: Per-recording label arrays [(T1,), (T2,), ...].
        rf_model: Trained RF classifier exposing ``predict``.

    Returns:
        Dict with ``Acc_T``, ``Acc_mu``, ``Kappa_T``, ``Kappa_mu``, the per-recording
        accuracy/kappa lists, and ``num_recordings``.
    """
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    # Per-recording metrics
    per_recording_acc = []
    per_recording_kappa = []

    all_y_true = []
    all_y_pred = []

    for i, (features, labels) in enumerate(zip(features_list, labels_list)):
        # Predict for this recording
        y_pred = rf_model.predict(features)

        # Per-recording metrics (κ_i, Acc_i)
        acc_i = accuracy_score(labels, y_pred)
        kappa_i = cohen_kappa_score(labels, y_pred)

        per_recording_acc.append(acc_i)
        per_recording_kappa.append(kappa_i)

        # Collect for overall metrics
        all_y_true.extend(labels)
        all_y_pred.extend(y_pred)

    # Acc_μ = (1/N) * Σ Acc_i  (Equation 8 equivalent for accuracy)
    acc_mu = np.mean(per_recording_acc)

    # κ_μ = (1/N) * Σ κ_i  (Equation 8)
    kappa_mu = np.mean(per_recording_kappa)

    # Acc_T: Overall accuracy across all epochs (Equation 7 equivalent)
    acc_T = accuracy_score(all_y_true, all_y_pred)

    # κ_T: Overall kappa across all epochs (Equation 7)
    kappa_T = cohen_kappa_score(all_y_true, all_y_pred)

    results = {
        'Acc_T': acc_T,
        'Acc_mu': acc_mu,
        'Kappa_T': kappa_T,
        'Kappa_mu': kappa_mu,
        'per_recording_acc': per_recording_acc,
        'per_recording_kappa': per_recording_kappa,
        'num_recordings': len(features_list)
    }

    logger.info(
        f"Per-recording metrics ({len(features_list)} recordings, {len(all_y_true)} epochs) | "
        f"Acc_T {acc_T:.4f} Acc_mu {acc_mu:.4f} (std {np.std(per_recording_acc):.4f}) | "
        f"Kappa_T {kappa_T:.4f} Kappa_mu {kappa_mu:.4f} (std {np.std(per_recording_kappa):.4f})"
    )

    return results


def _run_test_evaluation(cfg_base, feature_extractor, rf_model):
    """Evaluate ``rf_model`` on the KVSS test split.

    Extracts test features, writes per-epoch predictions and per-subject
    metrics to CSV, and logs overall / per-recording / per-class results plus
    a saved confusion-matrix figure. Shared by the ``fit_test`` and ``test`` modes.
    """
    cfg_test = cfg_base.copy()
    cfg_test.data.split = 'test'
    test_dataset = instantiate(cfg_base.data, split='test')
    test_loader = _make_dataloader(cfg_test.data, test_dataset)

    with torch.no_grad():
        test_features, test_labels, test_subject_ids = extract_features_from_loader(
            cfg_test.seq_len, cfg_test.step_size, cfg_test.data.motion_dim, feature_extractor, test_loader, "Extracting features from KVSS test set"
        )

    X_test = np.vstack([features for features in test_features])
    y_test = np.hstack([labels for labels in test_labels])

    logger.info(
        f"Test set: {len(y_test):,} epochs from {len(test_features)} recordings, "
        f"feature shape {X_test.shape}"
    )

    # Sleep stage label mapping
    class_names = ['Wake', 'N1+N2', 'N3', 'REM']

    # Per-recording predictions and per-subject metrics
    csv_data = []
    subject_metrics = []

    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, confusion_matrix as sk_cm

    for i, (features, labels, subject_id) in enumerate(zip(test_features, test_labels, test_subject_ids)):
        predictions = rf_model.predict(features)

        subject_acc = accuracy_score(labels, predictions)
        subject_kappa = cohen_kappa_score(labels, predictions)
        subject_f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        subject_f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        subject_precision = precision_score(labels, predictions, average='macro', zero_division=0)
        subject_recall = recall_score(labels, predictions, average='macro', zero_division=0)

        subject_cm = sk_cm(labels, predictions)

        per_class_acc = []
        for class_idx in range(len(class_names)):
            if class_idx < subject_cm.shape[0] and np.sum(subject_cm[class_idx, :]) > 0:
                class_acc = subject_cm[class_idx, class_idx] / np.sum(subject_cm[class_idx, :])
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(np.nan)

        subject_metrics.append({
            'subject_id': subject_id,
            'recording_idx': i,
            'num_epochs': len(labels),
            'accuracy': subject_acc,
            'kappa': subject_kappa,
            'f1_macro': subject_f1_macro,
            'f1_weighted': subject_f1_weighted,
            'precision_macro': subject_precision,
            'recall_macro': subject_recall,
            'wake_accuracy': per_class_acc[0],
            'n1n2_accuracy': per_class_acc[1],
            'n3_accuracy': per_class_acc[2],
            'rem_accuracy': per_class_acc[3]
        })

        for epoch_idx in range(len(labels)):
            csv_data.append({
                'subject_id': subject_id,
                'recording_idx': i,
                'epoch_idx': epoch_idx,
                'ground_truth_label': int(labels[epoch_idx]),
                'ground_truth_name': class_names[int(labels[epoch_idx])],
                'prediction_label': int(predictions[epoch_idx]),
                'prediction_name': class_names[int(predictions[epoch_idx])]
            })

    # Save per-epoch predictions and per-subject metrics
    df_predictions = pd.DataFrame(csv_data)
    csv_path = f"{cfg_base.model.name}_test_predictions.csv"
    df_predictions.to_csv(csv_path, index=False)

    df_metrics = pd.DataFrame(subject_metrics)
    metrics_path = f"{cfg_base.model.name}_test_subject_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    logger.info(
        f"Saved {len(csv_data):,} predictions to {csv_path} and "
        f"{len(subject_metrics)} subject metrics to {metrics_path}"
    )

    # Per-subject metric summary (mean ± std across subjects)
    logger.info(
        f"Per-subject (mean±std) | "
        f"acc {df_metrics['accuracy'].mean():.4f}±{df_metrics['accuracy'].std():.4f} | "
        f"kappa {df_metrics['kappa'].mean():.4f}±{df_metrics['kappa'].std():.4f} | "
        f"f1_macro {df_metrics['f1_macro'].mean():.4f}±{df_metrics['f1_macro'].std():.4f} | "
        f"f1_wtd {df_metrics['f1_weighted'].mean():.4f}±{df_metrics['f1_weighted'].std():.4f}"
    )
    logger.info(
        f"Per-class acc (mean across subjects) | "
        f"Wake {df_metrics['wake_accuracy'].mean():.4f} | "
        f"N1+N2 {df_metrics['n1n2_accuracy'].mean():.4f} | "
        f"N3 {df_metrics['n3_accuracy'].mean():.4f} | "
        f"REM {df_metrics['rem_accuracy'].mean():.4f}"
    )

    # Single detailed report provides both overall metrics and the per-class report
    detailed_report = rf_model.get_detailed_report(X_test, y_test)
    test_results = detailed_report['basic_metrics']
    class_report = detailed_report['classification_report']
    calculate_per_recording_metrics(test_features, test_labels, rf_model)

    # Overall test results
    cm = test_results['confusion_matrix']
    total_samples = int(np.sum(cm))
    correct_predictions = int(np.trace(cm))
    logger.info(
        f"Test overall | f1_macro {test_results['f1_macro']:.4f} "
        f"f1_weighted {test_results['f1_weighted']:.4f} | "
        f"correct {correct_predictions:,}/{total_samples:,}"
    )

    # Confusion matrix and per-class accuracy at debug level
    header = "        " + "  ".join([f"{name:>6}" for name in class_names])
    rows = "\n".join(
        f"{name:>6}: " + "  ".join([f"{cm[i, j]:>6d}" for j in range(len(class_names))])
        for i, name in enumerate(class_names)
    )
    logger.debug(f"Confusion matrix (actual vs predicted):\n{header}\n{rows}")

    per_class = []
    for i, name in enumerate(class_names):
        if np.sum(cm[i, :]) > 0:
            per_class.append(f"{name} {cm[i, i] / np.sum(cm[i, :]):.4f}")
        else:
            per_class.append(f"{name} N/A")
    logger.debug("Per-class accuracy | " + " | ".join(per_class))

    # Detailed classification report at debug level
    report_lines = [f"{'Class':<8} {'Precision':<10} {'Recall':<8} {'F1':<10} {'Support':<8}"]
    for i, class_name in enumerate(class_names):
        if str(i) in class_report:
            m = class_report[str(i)]
            report_lines.append(
                f"{class_name:<8} {m['precision']:<10.4f} {m['recall']:<8.4f} "
                f"{m['f1-score']:<10.4f} {int(m['support']):<8d}"
            )
    
    for avg_key, avg_label in (('macro avg', 'Macro'), ('weighted avg', 'Weighted')):
        if avg_key in class_report:
            m = class_report[avg_key]
            report_lines.append(
                f"{avg_label:<8} {m['precision']:<10.4f} {m['recall']:<8.4f} "
                f"{m['f1-score']:<10.4f} {int(m['support']):<8d}"
            )
    logger.debug("Classification report:\n" + "\n".join(report_lines))

    plot_confusion_matrix(
        test_results,
        class_report,
        class_names=class_names,
        title=f"{cfg_base.model.name}",
        save_path=f"{cfg_base.model.name}_confusion_matrix.png",
    )


def plot_confusion_matrix(result_dict, class_report, class_names=None, title="Test Results", save_path=None):
    """Render a confusion matrix heatmap annotated with counts, percentages,
    and per-class precision/recall.

    Args:
        result_dict: Dict with a 'confusion_matrix' key (numpy array).
        class_report: ``classification_report`` output dict (per-class precision/recall).
        class_names: Class label names (default: ['Wake', 'N1+N2', 'N3', 'REM']).
        title: Figure title.
        save_path: If given, the figure is saved to this path (600 dpi).

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """

    # 기본 라벨 설정
    if class_names is None:
        class_names = ['Wake', 'N1+N2', 'N3', 'REM']
    
    # 데이터 추출
    cm = result_dict['confusion_matrix']
    
    # classification_report에서 precision과 recall 추출
    precision = []
    recall = []
    
    for i in range(len(class_names)):
        if str(i) in class_report:
            metrics = class_report[str(i)]
            precision.append(metrics['precision'] * 100)  # 퍼센트로 변환
            recall.append(metrics['recall'] * 100)        # 퍼센트로 변환
        else:
            precision.append(0.0)
            recall.append(0.0)
    
    # confusion matrix를 percentage로 변환
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # annotation 생성 (개수와 퍼센트)
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
    
    # DataFrame 생성
    df = pd.DataFrame(cm_percent, index=class_names, columns=class_names)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(df, annot=annot, fmt='', cmap='Purples', cbar=False,
                linewidths=0, square=False,
                annot_kws={"size": 11})
    
    # precision 텍스트를 x축 위에 추가
    for i, p in enumerate(precision):
        ax.text(i + 0.5, -0.1, f'{p:.1f}%', ha='center', va='center', fontsize=11)
    
    # recall 텍스트를 y축 오른쪽에 추가
    for i, r in enumerate(recall):
        ax.text(len(class_names) + 0.1, i + 0.5, f'{r:.1f}%', va='center', fontsize=11)
    
    # 축 라벨 추가
    ax.text(len(class_names)/2, -0.2, 'Precision', ha='center', fontsize=11)
    ax.text(len(class_names) + 0.4, len(class_names)/2, 'Recall', va='center', rotation=90, fontsize=11)
    
    plt.ylabel('Experts labels', fontsize=14)
    plt.xlabel('SleepVST_baseline Outputs', fontsize=14)
    plt.title(title, fontsize=16, pad=30)
    plt.tight_layout()
    
    # 파일 저장
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    plt.show()
    return fig
