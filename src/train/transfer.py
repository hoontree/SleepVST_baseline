from src.eval.metrics import MetricsTracker, AverageMeter
from src.models.registry import get_model
from tqdm import tqdm
from torch import nn
import torch
from src.data.datasets.KVSS import KVSSDataModule
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.models.SleepVST import SleepVST, SleepVST_BW

def plot_confusion_matrix(result_dict, class_report, class_names=None, title="Test Results", save_path=None):
    """
    Dict 형태의 결과에서 confusion matrix를 그리는 함수
    
    Args:
        result_dict (dict): 다음 키들을 포함해야 함:
            - 'confusion_matrix': numpy array 형태의 confusion matrix
        class_report (dict): classification_report 결과 dict
        class_names (list): 클래스 이름 리스트 (기본값: ['Wake', 'N1+N2', 'N3', 'REM'])
        title (str): 그래프 제목
        save_path (str): 저장할 파일 경로 (선택사항)
    
    Returns:
        matplotlib.figure.Figure: 생성된 figure 객체
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

def extract_features_from_loader(cfg, feature_extractor: torch.nn.Module, data_loader, desc):
    """
    최적화된 특징 추출 함수 - 윈도우별로 한 번만 계산하고 레이블, subject_id도 함께 추출
    """
    feature_extractor.eval()
    window_size = cfg.seq_len
    step = cfg.step_size
    center_index = (window_size-1) // 2

    all_final_features = []
    all_labels = []
    all_subject_ids = []

    with torch.no_grad():
        if 'BW' in cfg.model.name:
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=desc)):
                x_bw = batch['x_bw'].cuda().float() # (B, N, 150)
                motion = batch['motion'].cuda().float() # (B, N, 90)
                labels = batch['label'].cpu().numpy()  # (B, N)
                subject_ids = batch['subject_id']  # (B,) - list of subject IDs
                B, N, _ = x_bw.shape
                T = min(N, batch['label'].shape[1])

                # 각 윈도우의 특징을 한 번씩만 계산
                window_features = {}
                # 시작점 리스트 만들기(+꼬리 커버용 마지막 창 강제 추가)
                starts = list(range(0, max(T - window_size + 1, 0), step))
                if T >= window_size and (not starts or starts[-1] != T - window_size):
                    starts.append(T - window_size)

                for start in starts:
                    x_window = x_bw[:, start:start + window_size]  # (B, W, 150)
                    features = feature_extractor(x_window).cpu().numpy()  # (B, W, 128)
                    window_features[start] = features

                # 각 시간 스텝에서 최적의 윈도우 선택
                batch_final_features = np.zeros((B, T, cfg.d_model + cfg.motion_dim), dtype=np.float32)  # (B, T, 128 + 90)
                batch_labels = np.zeros((B, T), dtype=np.int64)  # (B, T)

                for t in range(T):
                    best_window_start = -1
                    min_distance = float('inf')

                    # t를 포함하는 모든 윈도우 확인
                    for start_pos in window_features.keys():
                        if start_pos <= t < start_pos + window_size:
                            relative_pos = t - start_pos
                            distance = abs(relative_pos - center_index)
                            if distance < min_distance:
                                min_distance = distance
                                best_window_start = start_pos

                    if best_window_start != -1:
                        relative_pos = t - best_window_start
                        feature_vec = window_features[best_window_start][:, relative_pos]  # (B, 128)
                        motion_vec = motion[:, t, :].cpu().numpy()  # motion vector at time t (B, 90)

                        # feature와 motion을 concat
                        batch_final_features[:, t, :] = np.concatenate([feature_vec, motion_vec], axis=1)
                    batch_labels[:, t] = labels[:, t]  # 해당 시점의 레이블

                for i in range(B):
                    all_final_features.append(batch_final_features[i])
                    all_labels.append(batch_labels[i])
                    all_subject_ids.append(subject_ids[i])
        else:
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=desc)):
                if 'x_hw' not in batch or 'x_bw' not in batch:
                    raise KeyError("Batch must contain both 'x_hw' and 'x_bw' keys for dual input models.")
                x_hw = batch['x_hw'].cuda().float() # (B, N, 150)
                x_bw = batch['x_bw'].cuda().float() # (B, N, 150)
                motion = batch['motion'].cuda().float() # (B, N, 90)
                labels = batch['label'].cpu().numpy()  # (B, N)
                subject_ids = batch['subject_id']  # (B,) - list of subject IDs
                B, N, _ = x_bw.shape
                T = min(N, batch['label'].shape[1])

                # 각 윈도우의 특징을 한 번씩만 계산
                window_features = {}
                starts = list(range(0, max(T - window_size + 1, 0), step))
                if T >= window_size and (not starts or starts[-1] != T - window_size):
                    starts.append(T - window_size)

                for start in starts:
                    hw_window = x_hw[:, start:start + window_size]  # (B, W, 150)
                    bw_window = x_bw[:, start:start + window_size]  # (B, W, 150)
                    features = feature_extractor(hw_window, bw_window).cpu().numpy()  # (B, W, 128)
                    window_features[start] = features

                # 각 시간 스텝에서 최적의 윈도우 선택
                batch_final_features = np.zeros((B, T, cfg.d_model + cfg.motion_dim), dtype=np.float32)  # (B, T, 128 + 90)
                batch_labels = np.zeros((B, T), dtype=np.int64)  # (B, T)

                for t in range(T):
                    best_window_start = -1
                    min_distance = float('inf')

                    # t를 포함하는 모든 윈도우 확인
                    for start_pos in window_features.keys():
                        if start_pos <= t < start_pos + window_size:
                            relative_pos = t - start_pos
                            distance = abs(relative_pos - center_index)
                            if distance < min_distance:
                                min_distance = distance
                                best_window_start = start_pos

                    if best_window_start != -1:
                        relative_pos = t - best_window_start
                        feature_vec = window_features[best_window_start][:, relative_pos]  # (B, 128)
                        motion_vec = motion[:, t, :]  # motion vector at time t (B, 90)

                        # feature와 motion을 concat
                        batch_final_features[:, t, :] = np.concatenate([feature_vec, motion_vec], axis=1)
                    batch_labels[:, t] = labels[:, t]  # 해당 시점의 레이블

                for i in range(B):
                    all_final_features.append(batch_final_features[i])
                    all_labels.append(batch_labels[i])
                    all_subject_ids.append(subject_ids[i])


    return all_final_features, all_labels, all_subject_ids  # (B, T, 128 + 90), (B, T), (N,)

def analyze_motion_features(X_train, cfg, logger):
    """Motion feature 데이터 분석"""
    
    # Visual과 Motion feature 분리
    visual_data = X_train[:, :cfg.d_model]  # 0~127
    motion_data = X_train[:, cfg.d_model:]  # 128~217 (motion_dim=90)
    
    logger.info(f"\n=== Motion Feature Data Analysis ===")
    logger.info(f"Visual features shape: {visual_data.shape}")
    logger.info(f"Motion features shape: {motion_data.shape}")
    logger.info(f"Expected motion dim: {cfg.motion_dim}")
    
    # 1. 기본 통계
    logger.info(f"\nBasic Statistics:")
    logger.info(f"Motion data range: [{np.min(motion_data):.6f}, {np.max(motion_data):.6f}]")
    logger.info(f"Motion data mean: {np.mean(motion_data):.6f}")
    logger.info(f"Motion data std: {np.std(motion_data):.6f}")
    
    # 2. 각 feature별 분석
    motion_means = np.mean(motion_data, axis=0)
    motion_stds = np.std(motion_data, axis=0)
    motion_vars = np.var(motion_data, axis=0)
    motion_mins = np.min(motion_data, axis=0)
    motion_maxs = np.max(motion_data, axis=0)
    
    # 3. 문제가 있는 feature들 찾기
    zero_var_indices = np.where(motion_vars == 0)[0]
    very_low_var_indices = np.where(motion_vars < 1e-10)[0]
    constant_indices = np.where(motion_mins == motion_maxs)[0]
    
    logger.info(f"\nProblem Features:")
    logger.info(f"Zero variance features: {len(zero_var_indices)}/{cfg.motion_dim}")
    logger.info(f"Very low variance (<1e-10): {len(very_low_var_indices)}/{cfg.motion_dim}")
    logger.info(f"Constant features: {len(constant_indices)}/{cfg.motion_dim}")
    
    if len(zero_var_indices) > 0:
        logger.info(f"Zero variance indices (first 10): {zero_var_indices[:10]}")
    
    # 4. 유효한 feature들만 확인
    valid_var_indices = np.where(motion_vars > 1e-6)[0]
    logger.info(f"Valid features (var > 1e-6): {len(valid_var_indices)}/{cfg.motion_dim}")
    
    if len(valid_var_indices) > 0:
        logger.info(f"Valid features stats:")
        logger.info(f"  - Mean range: [{np.min(motion_means[valid_var_indices]):.6f}, {np.max(motion_means[valid_var_indices]):.6f}]")
        logger.info(f"  - Std range: [{np.min(motion_stds[valid_var_indices]):.6f}, {np.max(motion_stds[valid_var_indices]):.6f}]")
        logger.info(f"  - Value range: [{np.min(motion_mins[valid_var_indices]):.6f}, {np.max(motion_maxs[valid_var_indices]):.6f}]")
    
    # 5. Visual vs Motion 스케일 비교
    logger.info(f"\nScale Comparison:")
    logger.info(f"Visual - Mean: {np.mean(visual_data):.6f}, Std: {np.std(visual_data):.6f}")
    logger.info(f"Motion - Mean: {np.mean(motion_data):.6f}, Std: {np.std(motion_data):.6f}")
    
    if np.std(motion_data) > 0:
        scale_ratio = np.std(visual_data) / np.std(motion_data)
        logger.info(f"Scale ratio (Visual/Motion): {scale_ratio:.2f}")
        
        if scale_ratio > 100:
            logger.info("WARNING: Visual features have much larger scale than motion features!")
            logger.info("Consider feature scaling or normalization.")
    
    # 6. 샘플별 motion feature 확인 (처음 몇 샘플만)
    logger.info(f"\nFirst 5 samples motion data:")
    for i in range(min(5, motion_data.shape[0])):
        sample_data = motion_data[i, :10]  # 처음 10개 feature만
        logger.info(f"Sample {i}: {sample_data}")
    
    # 7. NaN, Inf 확인
    nan_count = np.sum(np.isnan(motion_data))
    inf_count = np.sum(np.isinf(motion_data))
    logger.info(f"\nData Quality:")
    logger.info(f"NaN values: {nan_count}")
    logger.info(f"Infinite values: {inf_count}")
    
    return {
        'zero_var_indices': zero_var_indices,
        'valid_var_indices': valid_var_indices,
        'constant_indices': constant_indices,
        'scale_ratio': np.std(visual_data) / np.std(motion_data) if np.std(motion_data) > 0 else float('inf')
    }

def calculate_per_recording_metrics(features_list, labels_list, rf_model, logger=None):
    """
    Args:
        features_list: List of feature arrays, one per recording [(T1, D), (T2, D), ...]
        labels_list: List of label arrays, one per recording [(T1,), (T2,), ...]
        rf_model: Trained RF classifier
        logger: Logger for output

    Returns:
        dict: Contains Acc_T, Acc_mu, Kappa_T, Kappa_mu metrics
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

    if logger:
        logger.info("\n" + "="*60)
        logger.info("PER-RECORDING METRICS (as per paper equations 7, 8)")
        logger.info("="*60)
        logger.info(f"\n📊 ACCURACY METRICS:")
        logger.info(f"Acc_T  (Total accuracy, all epochs):     {acc_T:.4f}")
        logger.info(f"Acc_μ  (Mean per-recording accuracy):    {acc_mu:.4f}")
        logger.info(f"       Std of per-recording accuracies:  {np.std(per_recording_acc):.4f}")

        logger.info(f"\n📈 KAPPA METRICS:")
        logger.info(f"κ_T    (Total kappa, all epochs):        {kappa_T:.4f}")
        logger.info(f"κ_μ    (Mean per-recording kappa):       {kappa_mu:.4f}")
        logger.info(f"       Std of per-recording kappas:      {np.std(per_recording_kappa):.4f}")

        logger.info(f"\nNumber of recordings: {len(features_list)}")
        logger.info(f"Total epochs: {len(all_y_true)}")

    return results

def transfer_to_video(cfg, logger=None):
    from src.models.RFClassifier import SleepVSTVideoRF, SleepVSTVideoRFConfig
    if logger:
        logger.info(f"Loading model...{cfg.model.name}")
    model_mapping = {
        "SleepVST": SleepVST,
        "SleepVST_BW": SleepVST_BW
    }
    model = model_mapping[cfg.model.name](cfg.model)
    if cfg.model.use_finetuned:
        checkpoint = torch.load(cfg.model.finetuned_checkpoint, map_location=cfg.system.device)
        if logger:
            logger.info(f"Model {cfg.model.name} loaded from {cfg.model.finetuned_checkpoint}")
    else:
        checkpoint = torch.load(cfg.model.checkpoint, map_location=cfg.system.device)
        if logger:
            logger.info(f"Model {cfg.model.name} loaded from {cfg.model.checkpoint}")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()  # 모델을 GPU로 이동
    
    in_features = model.classifier.in_features
    model.classifier = nn.Identity().cuda()
    feature_extractor = model
    feature_extractor.eval()  # evaluation 모드로 설정
    
    # 테스트 단계 공통 실행 함수 (fit_test와 test 모드에서 재사용)
    def _run_test_evaluation(cfg_base, feature_extractor, rf_model, motion_feature_keys, logger):
        # test split dataloader 생성
        cfg_test = cfg_base.copy()
        cfg_test.data.split = 'test'
        kvss_test_module = KVSSDataModule(cfg_test)
        print(kvss_test_module.dataset.__len__())
        test_loader = kvss_test_module.get_dataloader()

        with torch.no_grad():
            test_features, test_labels, test_subject_ids = extract_features_from_loader(
                cfg_test, feature_extractor, test_loader, "Extracting features from KVSS test set"
            )

        X_test = np.vstack([features for features in test_features])
        y_test = np.hstack([labels for labels in test_labels])

        # 데이터셋 크기 로깅
        logger.info(f"\n=== TEST DATASET SIZE ===")
        logger.info(f"Test samples: {len(y_test):,} epochs from {len(test_features)} recordings")
        logger.info(f"Test feature shape: {X_test.shape}")

        # Sleep stage label mapping
        class_names = ['Wake', 'N1+N2', 'N3', 'REM']

        # Per-recording predictions 저장 및 subject metrics 계산
        logger.info(f"\n=== SAVING PER-RECORDING PREDICTIONS ===")
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

        # Per-epoch predictions 저장
        df_predictions = pd.DataFrame(csv_data)
        csv_path = f"{cfg_base.model.name}_test_predictions.csv"
        df_predictions.to_csv(csv_path, index=False)
        logger.info(f"Saved per-epoch predictions to CSV: {csv_path}")
        logger.info(f"Total rows saved: {len(csv_data)}")
        logger.info(f"Columns: {list(df_predictions.columns)}")

        # Subject별 metrics 저장
        df_metrics = pd.DataFrame(subject_metrics)
        metrics_path = f"{cfg_base.model.name}_test_subject_metrics.csv"
        df_metrics.to_csv(metrics_path, index=False)
        logger.info(f"\nSaved per-subject metrics to CSV: {metrics_path}")
        logger.info(f"Total subjects: {len(subject_metrics)}")

        # Subject별 metrics 요약 출력
        logger.info(f"\n=== PER-SUBJECT METRICS SUMMARY ===")
        logger.info(f"Accuracy    - Mean: {df_metrics['accuracy'].mean():.4f}, Std: {df_metrics['accuracy'].std():.4f}")
        logger.info(f"Kappa       - Mean: {df_metrics['kappa'].mean():.4f}, Std: {df_metrics['kappa'].std():.4f}")
        logger.info(f"F1 (macro)  - Mean: {df_metrics['f1_macro'].mean():.4f}, Std: {df_metrics['f1_macro'].std():.4f}")
        logger.info(f"F1 (wtd)    - Mean: {df_metrics['f1_weighted'].mean():.4f}, Std: {df_metrics['f1_weighted'].std():.4f}")
        logger.info(f"Precision   - Mean: {df_metrics['precision_macro'].mean():.4f}, Std: {df_metrics['precision_macro'].std():.4f}")
        logger.info(f"Recall      - Mean: {df_metrics['recall_macro'].mean():.4f}, Std: {df_metrics['recall_macro'].std():.4f}")

        logger.info(f"\n=== PER-CLASS ACCURACY (AVERAGED ACROSS SUBJECTS) ===")
        logger.info(f"Wake        - Mean: {df_metrics['wake_accuracy'].mean():.4f}, Std: {df_metrics['wake_accuracy'].std():.4f}")
        logger.info(f"N1+N2       - Mean: {df_metrics['n1n2_accuracy'].mean():.4f}, Std: {df_metrics['n1n2_accuracy'].std():.4f}")
        logger.info(f"N3          - Mean: {df_metrics['n3_accuracy'].mean():.4f}, Std: {df_metrics['n3_accuracy'].std():.4f}")
        logger.info(f"REM         - Mean: {df_metrics['rem_accuracy'].mean():.4f}, Std: {df_metrics['rem_accuracy'].std():.4f}")

        test_results = rf_model.evaluate(X_test, y_test)

        # Calculate per-recording metrics (paper equations 7, 8)
        logger.info("="*60)
        logger.info("TEST SET: PER-RECORDING METRICS (Paper Equations 7, 8)")
        logger.info("="*60)
        _ = calculate_per_recording_metrics(test_features, test_labels, rf_model, logger)

        # 상세한 테스트 결과 출력
        logger.info("\n" + "="*60)
        logger.info("DETAILED TEST RESULTS")
        logger.info("="*60)
        
        logger.info(f"\n🎯 F1 SCORE METRICS:")
        logger.info(f"F1_macro: {test_results['f1_macro']:.4f}")
        logger.info(f"F1_weighted: {test_results['f1_weighted']:.4f}")
        
        logger.info(f"\n📋 CONFUSION MATRIX:")
        cm = test_results['confusion_matrix']
        logger.info(f"Shape: {cm.shape}")
        
        # 클래스별로 Confusion Matrix 출력
        class_names = ['Wake', 'N1+N2', 'N3', 'REM']  # 수면 단계
        logger.info("\nConfusion Matrix (Actual vs Predicted):")
        logger.info("        " + "  ".join([f"{name:>6}" for name in class_names]))
        for i, actual_class in enumerate(class_names):
            row_str = f"{actual_class:>6}: " + "  ".join([f"{cm[i,j]:>6d}" for j in range(len(class_names))])
            logger.info(row_str)
        
        # 클래스별 정확도 계산
        logger.info(f"\n📊 PER-CLASS ACCURACY:")
        class_accuracies = []
        for i in range(len(class_names)):
            if np.sum(cm[i, :]) > 0:  # 해당 클래스가 실제로 존재하는 경우
                class_acc = cm[i, i] / np.sum(cm[i, :])
                class_accuracies.append(class_acc)
                logger.info(f"{class_names[i]:>6}: {class_acc:.4f} ({cm[i,i]:>4d}/{np.sum(cm[i,:]):>4d})")
            else:
                logger.info(f"{class_names[i]:>6}: N/A (no samples)")
        
        # 전체 샘플 수 정보
        total_samples = np.sum(cm)
        correct_predictions = np.trace(cm)
        logger.info(f"\n📈 OVERALL STATISTICS:")
        logger.info(f"Total samples: {total_samples:,}")
        logger.info(f"Correct predictions: {correct_predictions:,}")
        logger.info(f"Incorrect predictions: {total_samples - correct_predictions:,}")
        
        # 상세 리포트 생성
        detailed_report = rf_model.get_detailed_report(X_test, y_test)
        
        # Classification Report 출력
        logger.info(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        class_report = detailed_report['classification_report']
        
        logger.info(f"{'Class':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'Support':<8}")
        logger.info("-" * 50)
        
        for i, class_name in enumerate(class_names):
            if str(i) in class_report:
                metrics = class_report[str(i)]
                logger.info(f"{class_name:<8} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f} {metrics['f1-score']:<10.4f} {int(metrics['support']):<8d}")
        
        # Macro and Weighted averages
        if 'macro avg' in class_report:
            macro_avg = class_report['macro avg']
            logger.info("-" * 50)
            logger.info(f"{'Macro':<8} {macro_avg['precision']:<10.4f} {macro_avg['recall']:<8.4f} {macro_avg['f1-score']:<10.4f} {int(macro_avg['support']):<8d}")
            
        if 'weighted avg' in class_report:
            weighted_avg = class_report['weighted avg']
            logger.info(f"{'Weighted':<8} {weighted_avg['precision']:<10.4f} {weighted_avg['recall']:<8.4f} {weighted_avg['f1-score']:<10.4f} {int(weighted_avg['support']):<8d}")

        # 함수 호출 시 class_names 전달
        fig = plot_confusion_matrix(
            test_results, 
            class_report, 
            class_names=class_names,  # class_names 추가
            title=f"{cfg.model.name}", 
            save_path=f"{cfg.model.name}_confusion_matrix.png"
        )

    # fit 또는 fit 후 즉시 test 실행
    if cfg.mode.classifier.mode in ('fit', 'fit_test'):
        cfg_train = cfg.copy()
        cfg_train.data.split = 'train'
        kvss_train_module = KVSSDataModule(cfg_train)

        cfg_val = cfg.copy()
        cfg_val.data.split = 'valid'
        kvss_val_module = KVSSDataModule(cfg_val)

        # Motion feature keys 정보 가져오기
        motion_feature_keys = kvss_train_module.dataset.get_motion_feature_keys()
        
        # Feature 이름 생성 (시각적 특징 + 동작 특징)
        signal_features = [f"signal_feat_{i}" for i in range(cfg.d_model)]
        if motion_feature_keys is not None:
            motion_features = motion_feature_keys
        else:
            motion_features = [f"motion_feat_{i}" for i in range(cfg.motion_dim)]
        feature_names = signal_features + motion_features
        
        # Feature keys 정보 로깅
        if logger and motion_feature_keys is not None:
            logger.info(f"Motion feature keys loaded: {len(motion_feature_keys)} features")
            logger.info(f"First 10 motion keys: {motion_feature_keys[:10]}")
        
        train_loader = kvss_train_module.get_dataloader()
        val_loader = kvss_val_module.get_dataloader()
        
        rf_config = SleepVSTVideoRFConfig(
            n_estimators=300,
            search_params=False,
            verbose=1
        )
        rf_model = SleepVSTVideoRF(rf_config)
        
        with torch.no_grad():
            train_features, train_labels, train_subject_ids = extract_features_from_loader(cfg_train,
                feature_extractor,
                train_loader,
                "Extracting features from KVSS train set"
            )

            val_features, val_labels, val_subject_ids = extract_features_from_loader(cfg_val,
                feature_extractor,
                val_loader,
                "Extracting features from KVSS validation set"
            )
            
        X_train = np.vstack([features for features in train_features])
        X_val = np.vstack([features for features in val_features])

        y_train = np.hstack([labels for labels in train_labels])
        y_val = np.hstack([labels for labels in val_labels])

        # 데이터셋 크기 로깅
        logger.info(f"\n=== DATASET SIZES ===")
        logger.info(f"Train samples: {len(y_train):,} epochs from {len(train_features)} recordings")
        logger.info(f"Val samples: {len(y_val):,} epochs from {len(val_features)} recordings")
        logger.info(f"Train feature shape: {X_train.shape}")
        logger.info(f"Val feature shape: {X_val.shape}")

        rf_model.fit(X_train, y_train, feature_names=feature_names)  # feature 이름 전달

        # Basic evaluation (overall metrics only)
        train_results = rf_model.evaluate(X_train, y_train)
        val_results = rf_model.evaluate(X_val, y_val)

        logger.info(f"\n=== BASIC METRICS (Overall) ===")
        logger.info(f"Train Accuracy: {train_results['accuracy']:.4f}, Kappa: {train_results['kappa']:.4f}, F1: {train_results['f1']:.4f}")
        logger.info(f"Val Accuracy: {val_results['accuracy']:.4f}, Kappa: {val_results['kappa']:.4f}, F1: {val_results['f1']:.4f}")

        # Per-recording metrics (paper equations 7, 8)
        logger.info(f"\n=== TRAIN SET: PER-RECORDING METRICS ===")
        train_per_recording = calculate_per_recording_metrics(train_features, train_labels, rf_model, logger)

        logger.info(f"\n=== VALIDATION SET: PER-RECORDING METRICS ===")
        val_per_recording = calculate_per_recording_metrics(val_features, val_labels, rf_model, logger)
        
        # 모델 저장 시 motion feature keys도 함께 저장
        if cfg.model.use_finetuned:
            rf_model.save(f"./checkpoint/randomforest/{cfg.model.name}_rf_model_used_finetuned_{cfg.data.data_source}.pkl", metadata={'motion_feature_keys': motion_feature_keys})
        else:
            rf_model.save(f"./checkpoint/randomforest/{cfg.model.name}_rf_model_{cfg.data.data_source}.pkl", metadata={'motion_feature_keys': motion_feature_keys})
        
        # fit 직후 test 실행 (새 모드)
        if cfg.mode.classifier.mode == 'fit_test':
            _run_test_evaluation(cfg, feature_extractor, rf_model, motion_feature_keys, logger)
    else:
        # test 전용 모드 (이전 로직과 동일, 단 평가 로직을 함수로 호출)
        if cfg.model.use_finetuned:
            rf_model = SleepVSTVideoRF.load(f"./checkpoint/randomforest/{cfg.model.name}_rf_model_used_finetuned.pkl")
        else:
            rf_model = SleepVSTVideoRF.load(f"./checkpoint/randomforest/{cfg.model.name}_rf_model.pkl")
        metadata = rf_model.get_metadata() if hasattr(rf_model, 'get_metadata') else {}
        motion_feature_keys = metadata.get('motion_feature_keys', None)

        _run_test_evaluation(cfg, feature_extractor, rf_model, motion_feature_keys, logger)

    # # Feature importance 분석 (공통)
    # # 전체 feature importance 가져오기
    # full_importance_df = rf_model.get_feature_importance()
    
    # # signal vs Motion feature importance 계산 (전체 데이터 기준)
    # signal_mask = full_importance_df['feature'].str.startswith('signal_feat_')
    # motion_mask = full_importance_df['feature'].str.startswith(('past_', 'current_', 'future_')) | \
    #               full_importance_df['feature'].str.startswith('motion_feat_')

    # signal_importance = full_importance_df[signal_mask]['importance'].sum()
    # motion_importance = full_importance_df[motion_mask]['importance'].sum()
    
    # logger.info(f"\n🔍 FEATURE TYPE IMPORTANCE ANALYSIS:")
    # logger.info(f"Total features: {len(full_importance_df)}")
    # logger.info(f"Signal features: {signal_mask.sum()}")
    # logger.info(f"Motion features: {motion_mask.sum()}")
    # logger.info(f"Signal features total importance: {signal_importance:.4f}")
    # logger.info(f"Motion features total importance: {motion_importance:.4f}")
    
    # # 안전한 비율 계산
    # if motion_importance > 0:
    #     ratio = signal_importance / motion_importance
    #     logger.info(f"Signal/Motion ratio: {ratio:.2f}")
    # else:
    #     logger.info("Motion features have zero importance - all importance from signal features")

    # # 상위 20개 feature importance 출력
    # importance_df = rf_model.get_feature_importance(top_k=20)
    # logger.info("\n🏆 TOP 20 FEATURE IMPORTANCE:")
    # for idx, row in importance_df.iterrows():
    #     logger.info(f"{row['feature']}: {row['importance']:.6f}")
    
    # # 각 feature type별 상위 features 분석
    # top_signal = full_importance_df[signal_mask].head(10)
    # top_motion = full_importance_df[motion_mask].head(10)

    # logger.info("\n📡 TOP 10 SIGNAL FEATURES:")
    # for idx, row in top_signal.iterrows():
    #     logger.info(f"{row['feature']}: {row['importance']:.6f}")
    
    # logger.info("\n🏃 TOP 10 MOTION FEATURES:")
    # for idx, row in top_motion.iterrows():
    #     logger.info(f"{row['feature']}: {row['importance']:.6f}")
    
    # # Motion feature를 temporal type별로 분석
    # if motion_feature_keys is not None:
    #     logger.info("\n🔍 MOTION FEATURES BY TEMPORAL TYPE:")
        
    #     # Past, Current, Future로 분류
    #     past_mask = full_importance_df['feature'].str.startswith('past_')
    #     current_mask = full_importance_df['feature'].str.startswith('current_')
    #     future_mask = full_importance_df['feature'].str.startswith('future_')
        
    #     past_importance = full_importance_df[past_mask]['importance'].sum()
    #     current_importance = full_importance_df[current_mask]['importance'].sum()
    #     future_importance = full_importance_df[future_mask]['importance'].sum()
        
    #     logger.info(f"Past motion features importance: {past_importance:.4f}")
    #     logger.info(f"Current motion features importance: {current_importance:.4f}")
    #     logger.info(f"Future motion features importance: {future_importance:.4f}")
        
    #     # 각 temporal type별 top 5 features
    #     if past_mask.any():
    #         logger.info("\n⬅️  TOP 5 PAST MOTION FEATURES:")
    #         for idx, row in full_importance_df[past_mask].head(5).iterrows():
    #             logger.info(f"{row['feature']}: {row['importance']:.6f}")
        
    #     if current_mask.any():
    #         logger.info("\n⏺️  TOP 5 CURRENT MOTION FEATURES:")
    #         for idx, row in full_importance_df[current_mask].head(5).iterrows():
    #             logger.info(f"{row['feature']}: {row['importance']:.6f}")
                
    #     if future_mask.any():
    #         logger.info("\n➡️  TOP 5 FUTURE MOTION FEATURES:")
    #         for idx, row in full_importance_df[future_mask].head(5).iterrows():
    #             logger.info(f"{row['feature']}: {row['importance']:.6f}")

    # # Feature importance 통계
    # logger.info(f"\n📊 FEATURE IMPORTANCE STATISTICS:")
    # logger.info(f"Signal features - Mean: {full_importance_df[signal_mask]['importance'].mean():.6f}, "
    #             f"Std: {full_importance_df[signal_mask]['importance'].std():.6f}")
    # logger.info(f"Motion features - Mean: {full_importance_df[motion_mask]['importance'].mean():.6f}, "
    #             f"Std: {full_importance_df[motion_mask]['importance'].std():.6f}")
    
    # # Feature importance 시각화 저장
    # rf_model.plot_feature_importance(top_k=20, save_path="feature_importance.png")
    
    return rf_model

