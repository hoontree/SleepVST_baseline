import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import copy
import numpy as np
import config
from data.MESA import MESA
from data.SHHS import SHHS
from data.KVSS import KVSS
from models.SleepVST import SleepVST
from tqdm import tqdm
from utils.customlogger import Logger
from utils.util import MetricsTracker, AverageMeter
from collections import Counter
import wandb

def setup_dataloaders(args, logger):
    """데이터 로더들을 설정하고 반환합니다."""
    dataloaders = {}
    batch_size = args.batch_size
    num_workers = args.num_workers
    dataset_type = getattr(args, 'dataset', 'all')
    mode = getattr(args, 'mode', 'train_and_test')
    finetune_mode = mode == 'finetune'
    
    # Fine-tuning 모드에서는 항상 KVSS만 사용
    if finetune_mode:
        use_kvss = True
        use_shhs_mesa = False
        logger.info("Fine-tuning mode: Using only KVSS dataset")
    else:
        use_kvss = args.kvss or dataset_type == 'kvss' or dataset_type == 'all'
        use_shhs_mesa = dataset_type == 'shhs_mesa' or dataset_type == 'all'
    
    # --- 테스트 데이터 로더 ---
    if use_shhs_mesa:
        logger.info("Loading SHHS and MESA test datasets...")
        shhs_test = SHHS(split='test')
        mesa_test = MESA(split='test')
        dataloaders['shhs_test'] = DataLoader(shhs_test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        dataloaders['mesa_test'] = DataLoader(mesa_test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    if use_kvss:
        logger.info("Loading KVSS test dataset...")
        kvss_test = KVSS(split='test')
        dataloaders['kvss_test'] = DataLoader(kvss_test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    # --- 학습/검증 데이터 로더 ---
    if 'train' in mode or finetune_mode:
        train_datasets = []
        val_datasets = []

        if use_kvss:
            logger.info("Loading KVSS train/val datasets...")
            kvss_train = KVSS(split='train')
            kvss_val = KVSS(split='val')
            train_datasets.append(kvss_train)
            val_datasets.append(kvss_val)
            logger.info(f"KVSS training data size: {len(kvss_train)}")
            logger.info(f"KVSS validation data size: {len(kvss_val)}")

        # Fine-tuning 모드가 아니고 use_shhs_mesa가 활성화된 경우에만 SHHS/MESA 추가
        if use_shhs_mesa and not finetune_mode:
            if args.if_scratch:  # if_scratch 조건이 있는 경우에만 
                logger.info("Loading SHHS and MESA train/val datasets...")
                shhs_train = SHHS(split='train')
                mesa_train = MESA(split='train')
                shhs_val = SHHS(split='val')
                mesa_val = MESA(split='val')
                
                train_datasets.extend([shhs_train, mesa_train])
                val_datasets.extend([shhs_val, mesa_val])
                logger.info(f"SHHS training data size: {len(shhs_train)}")
                logger.info(f"MESA training data size: {len(mesa_train)}")
                logger.info(f"SHHS validation data size: {len(shhs_val)}")
                logger.info(f"MESA validation data size: {len(mesa_val)}")

        if not train_datasets:
            logger.warning("No training datasets selected based on arguments.")
        else:
            train_dataset = ConcatDataset(train_datasets)
            dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            logger.info(f"Total training dataset size: {len(train_dataset)}")

        if not val_datasets:
            logger.warning("No validation datasets selected based on arguments.")
        else:
            val_dataset = ConcatDataset(val_datasets)
            dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            logger.info(f"Total validation dataset size: {len(val_dataset)}")

    return dataloaders

def sliding_window_inference(model, loader, criterion, window_size=240, step_size=60):
    """
    SleepVST inference with overlapping windows and majority voting.
    """
    model.eval()
    metrics = MetricsTracker()
    losses = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Sliding Inference", leave=False)):
            x_hw = batch['x_hw'].squeeze(0).cuda()  # (T, 300)
            x_bw = batch['x_bw'].squeeze(0).cuda()  # (T, 150)
            labels = batch['label'].squeeze(0).cpu().numpy()  # (T,)
            
            T = x_hw.shape[0]  # 현재 배치의 실제 시퀀스 길이
            
            # 짧은 시퀀스 처리
            if T < window_size:
                # print(f"WARNING: Sequence {batch_idx} length ({T}) is shorter than window_size ({window_size})")
                
                if T < 7:  # 커널 크기보다 작으면 건너뜀
                    # print(f"ERROR: Sequence {batch_idx} length ({T}) is too small for convolution (min: 7)")
                    # 빈 배열 추가하고 계속 진행 (나중에 처리)
                    all_preds.append(np.zeros_like(labels))
                    all_labels.append(labels)
                    continue
                
                hw_chunk = x_hw.unsqueeze(0)  # (1, T, 300)
                bw_chunk = x_bw.unsqueeze(0)  # (1, T, 150)
                logits = model(hw_chunk, bw_chunk)  # (1, T, num_classes)
                preds = logits.squeeze(0).argmax(-1).cpu().numpy()
                
                # 손실 계산 (배치별로)
                batch_loss = criterion(logits.view(-1, logits.size(-1)), 
                                       torch.tensor(labels).cuda().view(-1).long())
                losses.update(batch_loss.item(), len(labels))
                
                all_preds.append(preds)
                all_labels.append(labels)
                continue
            
            batch_preds = [[] for _ in range(T)]
            
            # 슬라이딩 윈도우 추론
            for start in range(0, T - window_size + 1, step_size):
                end = start + window_size
                hw_chunk = x_hw[start:end].unsqueeze(0)
                bw_chunk = x_bw[start:end].unsqueeze(0)
                logits = model(hw_chunk, bw_chunk)  # (1, window_size, num_classes)
                preds = logits.squeeze(0).argmax(-1).cpu().numpy()  # (window_size,)
                
                for i in range(window_size):
                    batch_preds[start + i].append(preds[i])
            
            # 다수결로 최종 예측값 결정
            final_preds = np.zeros(T, dtype=int)
            for i in range(T):
                if len(batch_preds[i]) == 0:
                    # 예측값이 없는 경우 (끝부분)
                    non_empty_preds = [p for sublist in batch_preds if sublist for p in sublist]
                    if non_empty_preds:
                        final_preds[i] = np.argmax(np.bincount(non_empty_preds))
                    else:
                        final_preds[i] = 0  # 기본값
                else:
                    counter = Counter(batch_preds[i])
                    most_common = counter.most_common(1)[0][0]
                    final_preds[i] = most_common
            
            all_preds.append(final_preds)
            all_labels.append(labels)
    
    # 배치별 예측과 레이블을 하나로 결합
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 성능 평가
    metrics.update(torch.tensor(all_preds).cuda(), torch.tensor(all_labels).cuda().long())
    
    # 전체 손실 계산 부분 수정 (이미 배치별로 계산했으므로 여기서는 생략)
    val_loss = losses.avg
    
    acc, f1, kappa, cm, cm_norm, cr = metrics.compute()
    
    return val_loss, acc, f1, kappa, cm, cm_norm, cr
            

def sliding_window_features(model, x_hw, x_bw, window_size=240, step_size=60, device='cuda'):
    """
    SleepVST feature extractor for longer sequences (return closest-to-center vector).

    Returns:
        features: (T, D) numpy array where D = model's hidden dim
    """
    model.eval()
    T = x_hw.shape[0]
    feature_dim = model.classifier.in_features  # d_model
    feature_buffer = [[] for _ in range(T)]

    with torch.no_grad():
        for start in range(0, T - window_size + 1, step_size):
            end = start + window_size
            hw_chunk = x_hw[start:end].unsqueeze(0).cuda()
            bw_chunk = x_bw[start:end].unsqueeze(0).cuda()

            z = model.forward_features(hw_chunk, bw_chunk)  # (1, 240, D)
            z = z.squeeze(0).cpu().numpy()

            for i in range(window_size):
                feature_buffer[start + i].append((i, z[i]))

    # 중심에 가장 가까운 벡터 선택
    features = []
    for i, candidates in enumerate(feature_buffer):
        if not candidates:
            features.append(np.zeros(feature_dim))
        else:
            center = window_size // 2
            closest = min(candidates, key=lambda x: abs(x[0] - center))
            features.append(closest[1])

    return np.stack(features)  # (T, D)
        
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    
    losses = AverageMeter()
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        x_hw = batch['x_hw'].cuda()       # (B, 240, 300)
        x_bw = batch['x_bw'].cuda()       # (B, 240, 150)
        labels = batch['label'].cuda().long()  # (B, 240)

        optimizer.zero_grad()
        logits = model(x_hw, x_bw)  # (B, 240, num_classes)

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # ====== NaN/Inf 검사: loss ======
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Batch {batch_idx}] Loss is NaN or Inf!")
            print("logits stats →", logits.min().item(), logits.max().item())
            print("labels stats →", labels.min().item(), labels.max().item())
            continue  # NaN 발생 시 해당 배치 스킵
        
            # 라벨 범위 확인
            num_classes_model = logits.size(-1)
            invalid_labels = (labels < 0) | (labels >= num_classes_model)
            if invalid_labels.any():
                print(f"INVALID LABELS FOUND! Out of range [0, {num_classes_model-1}]")
                print(f"Invalid labels count: {invalid_labels.sum().item()}")
                print(f"Invalid label values: {labels[invalid_labels].unique().cpu().tolist()}")
            
            # NaN/Inf 값 확인
            if torch.isnan(logits).any():
                print("NaN values in logits:", torch.isnan(logits).sum().item())
            if torch.isinf(logits).any():
                print("Inf values in logits:", torch.isinf(logits).sum().item())

        try:
            loss.backward()
        except RuntimeError as e:
            print(f"[Batch {batch_idx}] RuntimeError during backward: {e}")
            print("Loss value:", loss.item())
            raise  # 오류 재발생 시켜서 중단

        optimizer.step()
        
        preds = logits.argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        losses.update(loss.item(), labels.numel())
          

    return losses.avg, correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    
    metrics = MetricsTracker()
    losses = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            x_hw = batch['x_hw'].cuda()
            x_bw = batch['x_bw'].cuda()
            labels = batch['label'].cuda().long()

            logits = model(x_hw, x_bw)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            losses.update(loss.item(), labels.numel())
            preds = logits.argmax(-1)
            metrics.update(preds, labels)

    val_loss = losses.avg
    acc, f1, kappa, cm, cm_norm, cr = metrics.compute()

    return val_loss, acc, f1, kappa, cm, cm_norm

def main():
    # 설정 로드
    args = config.parse_args()
    
    # args에 mode 속성이 없다면 기본값으로 'train_and_test' 사용
    mode = getattr(args, 'mode', 'train_and_test')
    
    # 공통 설정 로드
    seq_len = args.seq_len
    hw_length = args.patch_hw
    bw_length = args.patch_bw
    num_heads = args.num_heads
    num_layers = args.num_layers
    d_model = args.d_model
    kvss = args.kvss
    
    dataset = getattr(args, 'dataset', 'all')
    finetune_mode = mode == 'finetune'
    
    use_shhs_mesa = (dataset == 'shhs_mesa' or dataset == 'all') and not finetune_mode
    use_kvss = kvss or dataset == 'kvss' or dataset == 'all'
    gpu_ids = args.gpu_ids
    num_workers = args.num_workers
    checkpoint_dir = args.checkpoint_dir
    log_name = args.log_name
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    
    lr = args.lr
    weight_decay = args.weight_decay
    early_stopping = args.early_stopping
    batch_size = args.batch_size
    num_classes = args.num_classes
    end_epoch = args.end_epoch
    if_scratch = args.if_scratch
    
    if finetune_mode:
        # Fine-tuning은 항상 KVSS 데이터셋을 사용
        dataset = 'kvss'
        # Fine-tuning에선 더 작은 학습률 사용 (기존의 1/10)
        lr = lr / 10 if args.finetune_lr is None else args.finetune_lr
        # pretrained 체크포인트 경로 (기본 체크포인트와 다를 수 있음)
        pretrained_checkpoint = checkpoint_dir
        pretrained_checkpoint_path = os.path.join(pretrained_checkpoint, 'checkpoint.pth')
    

    
    # Logger 설정
    if not os.path.exists('output/log'):
        os.makedirs('output/log')
    logger = Logger(dir='output/log', name='SleepVST' + '.train')
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logger.info(f"Checkpoint path: {checkpoint_path}")
    
    # 모델 초기화
    model = SleepVST().cuda()
    
    dataloaders = setup_dataloaders(args, logger)
    
    # wandb 설정
    if 'train' in mode or finetune_mode:
        wandb_key = os.environ.get("WANDB_API_KEY", "7470ee66d23ffbe8f8ccbbf5bfeeb19966a69540")
        wandb.login(key=wandb_key)
        run = wandb.init(
            project="SleepVST",
            entity="heohun-seoul-national-university"
        )
        
        wandb_config = {
            "epochs": end_epoch,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "seq_len": seq_len,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dataset": dataset,
            "mode": mode
        }
        wandb.config.update(wandb_config)
        run_name = f"{log_name}_{mode}" if finetune_mode else log_name
        wandb.run.name = run_name
        
        # 손실 함수, 옵티마이저 설정
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # finetune 모드일 때 체크포인트 로드 또는 새로 시작
        if finetune_mode:
            if os.path.exists(pretrained_checkpoint_path):
                logger.info(f"Loading pre-trained checkpoint from {pretrained_checkpoint_path} for fine-tuning")
                checkpoint = torch.load(pretrained_checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Pre-trained model loaded successfully, starting fine-tuning on KVSS dataset")
                
                # Fine-tuning은 새로운 시작점
                begin_epoch = 0
                train_losses = []
                train_accs = []
                val_losses = []
                val_accs = []
                f1s = []
                cms = []
            else:
                logger.error(f"Pre-trained checkpoint not found at {pretrained_checkpoint_path}.")
                return
        # 일반 훈련 모드에서 체크포인트 로드 또는 새로 시작
        elif not if_scratch and os.path.exists(checkpoint_path):
            logger.info(f"loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            begin_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            train_losses = checkpoint['train_loss']
            val_losses = checkpoint['val_loss']
            train_accs = checkpoint['train_acc']
            val_accs = checkpoint['val_acc']
            f1s = checkpoint['val_f1']
            last_epoch = checkpoint['epoch']
            logger.info(f"loaded checkpoint from {checkpoint_path} at epoch {last_epoch}")
            
            cms = []
            begin_epoch = last_epoch + 1
            wandb.log(
                data={
                    "train_loss": train_losses,
                    "train_accuracy": train_accs,
                    "val_loss": val_losses,
                    "val_accuracy": val_accs,
                    "val_f1": f1s,
                },
                step=begin_epoch,
            )
        else:
            begin_epoch = 0
            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []
            f1s = []
            cms = []
        
        # 학습 시작
        best_val_loss = float('inf')
        best_model_state = None
        patience = 0
        
        dataset_text = "KVSS dataset" if finetune_mode else f"dataset: {dataset}"
        logger.info(f"{'Fine-tuning' if finetune_mode else 'Training'} started with {dataset_text}")
        training_start_time = time.time()
        
        for epoch in range(begin_epoch, end_epoch):
            start_time = time.time()
            logger.info(f"\n===== Epoch {epoch + 1}/{end_epoch} =====")
            
            train_loss, train_accuracy = train_one_epoch(model, dataloaders['train'], optimizer, criterion)
            
            wandb.log(
                data={
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                },
                step=epoch,
            )
            
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            
            val_loss, val_accuracy, val_f1, kappa, val_cm, _ = evaluate(model, dataloaders['val'], criterion)
            
            wandb.log(
                data={
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_f1": val_f1,
                    "val_kappa": kappa,
                },
                step=epoch,
            )
            
            val_losses.append(val_loss)
            val_accs.append(val_accuracy)
            f1s.append(val_f1)
            cms.append(val_cm)
            
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.")
            logger.info(f"Epoch {epoch+1}/{end_epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}",)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                
                # Fine-tuning일 경우 다른 이름으로 저장
                save_path = checkpoint_path
                if finetune_mode:
                    save_path = os.path.join(checkpoint_dir, 'kvss_finetuned_checkpoint.pth')
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_losses,
                    'val_loss': val_losses,
                    'train_acc': train_accs,
                    'val_acc': val_accs,
                    'val_f1': f1s
                }, save_path)
                logger.info(f"Saved checkpoint to {save_path}")
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping:
                    logger.info("Early stopping triggered.")
                    break
        
        total_time = time.time() - training_start_time
        logger.info(f"{'Fine-tuning' if finetune_mode else 'Training'} completed in {total_time:.2f} seconds.")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info("Best model loaded.")
    
    if 'test' in mode:
        if 'train' not in mode and not use_kvss:
            logger.info(f"Test mode: Loading model from {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model loaded successfully")
            else:
                logger.error(f"No checkpoint found at {checkpoint_path}. Cannot test.")
                return
        elif 'train' not in mode and use_kvss:
            checkpoint_path = os.path.join(checkpoint_dir, 'kvss_finetuned_checkpoint.pth')
            logger.info(f"Fine-tuning mode: Loading model from {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model loaded successfully")
            else:
                logger.error(f"No checkpoint found at {checkpoint_path}. Cannot test.")
                return
        
        # 테스트 로거 설정
        test_logger = Logger(dir='output/log', name='SleepVST' + '.test')
        
        if use_shhs_mesa:
            criterion = nn.CrossEntropyLoss().cuda()
            
            shhs_loader = dataloaders['shhs_test']
            mesa_loader = dataloaders['mesa_test']
            
            # SHHS sliding window inference
            test_logger.info("----SHHS TEST----")
            loss, acc, f1, kappa, cm, cm_norm, cr = sliding_window_inference(model, shhs_loader, criterion, window_size=240, step_size=60)
            test_logger.info(f"SHHS Test (Sliding) Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
            test_logger.info(f"SHHS Confusion Matrix:\n{cm}")
            test_logger.info(f"SHHS Confusion Matrix (Normalized):\n{cm_norm}")
            test_logger.info(f"Classification Report:\n{cr}")

            # MESA sliding window inference
            test_logger.info("----MESA TEST----")
            loss, acc, f1, kappa, cm, cm_norm, cr = sliding_window_inference(model, mesa_loader, criterion, window_size=240, step_size=60)
            test_logger.info(f"MESA Test (Sliding) Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
            test_logger.info(f"MESA Confusion Matrix:\n{cm}")
            test_logger.info(f"MESA Confusion Matrix (Normalized):\n{cm_norm}")
            test_logger.info(f"Classification Report:\n{cr}")

            if 'train' in mode and 'run' in locals():
                wandb.log({
                    "shhs_test_acc": acc,
                    "shhs_test_f1": f1,
                    "shhs_test_kappa": kappa,
                    "mesa_test_acc": acc,
                    "mesa_test_f1": f1,
                    "mesa_test_kappa": kappa,
                })
            
        if use_kvss:
            kvss_loader = dataloaders['kvss_test']
            criterion = nn.CrossEntropyLoss().cuda()
            
            test_logger.info("----KVSS TEST (sliding_window)----")
            
            loss, acc, f1, kappa, cm, cm_norm, cr = sliding_window_inference(model, kvss_loader, criterion, window_size=240, step_size=60)
            test_logger.info(f"KVSS Test (Sliding) Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
            test_logger.info(f"KVSS Confusion Matrix:\n{cm}")
            test_logger.info(f"KVSS Confusion Matrix (Normalized):\n{cm_norm}")
            test_logger.info(f"Classification Report:\n{cr}")

            if 'train' in mode and 'run' in locals():
                wandb.log({
                    "kvss_test_acc": acc,
                    "kvss_test_f1": f1,
                    "kvss_test_kappa": kappa,
                })
    
    # wandb 종료 (훈련 모드였을 경우)
    if 'train' in mode and 'run' in locals():
        wandb.finish()
    
    logger.info("Process completed.")
    
if __name__ == "__main__":    
    main()