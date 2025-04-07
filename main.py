import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import numpy as np
import config
from data.MESA import MESA
from data.SHHS import SHHS
from models.SleepVST import SleepVST
from tqdm import tqdm
from utils.logger import Logger
from utils.util import MetricsTracker, AverageMeter
from collections import Counter
import wandb
        
def sliding_window_inference(model, x_hw, x_bw, window_size=240, step_size=60, device='cuda'):
    """
    SleepVST inference with overlapping windows and majority voting.
    
    Args:
        model: trained SleepVST model
        x_hw: heart waveform (T, 300)
        x_bw: breath waveform (T, 150)
        window_size: number of epochs per window (default=240)
        step_size: sliding step size (default=60 → 30min if 2 epoch/min)
        device: inference device

    Returns:
        final_preds: (T,) array of majority-voted predictions
    """
    model.eval()
    T = x_hw.shape[0]
    all_preds = [[] for _ in range(T)]

    with torch.no_grad():
        for start in range(0, T - window_size + 1, step_size):
            end = start + window_size
            hw_chunk = x_hw[start:end].unsqueeze(0).cuda()  # (1, 240, 300)
            bw_chunk = x_bw[start:end].unsqueeze(0).cuda()  # (1, 240, 150)

            logits = model(hw_chunk, bw_chunk)
            preds = logits.argmax(-1).squeeze(0).cpu().numpy()  # (240,)

            for i in range(window_size):
                all_preds[start + i].append(preds[i])

    # 최빈값으로 결합
    final_preds = np.array([Counter(p).most_common(1)[0][0] if p else -1 for p in all_preds])
    return final_preds


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

        # ====== NaN/Inf 검사: 입력 ======
        if torch.isnan(x_hw).any() or torch.isinf(x_hw).any():
            print(f"[Batch {batch_idx}] x_hw contains NaN or Inf")
        if torch.isnan(x_bw).any() or torch.isinf(x_bw).any():
            print(f"[Batch {batch_idx}] x_bw contains NaN or Inf")
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            print(f"[Batch {batch_idx}] labels contain NaN or Inf")

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
    acc, f1, kappa, cm = metrics.compute()

    return val_loss, acc, f1, kappa, cm

def main():
    # 설정 로드
    args = config.parse_args()
    
    # 먼저 args에 mode 인자를 추가했다고 가정합니다
    # args에 mode 속성이 없다면 기본값으로 'train_and_test' 사용
    mode = getattr(args, 'mode', 'train_and_test')
    
    # 공통 설정 로드
    seq_len = args.seq_len
    hw_length = args.patch_hw
    bw_length = args.patch_bw
    num_heads = args.num_heads
    num_layers = args.num_layers
    d_model = args.d_model
    
    lr = args.lr
    weight_decay = args.weight_decay
    early_stopping = args.early_stopping
    batch_size = args.batch_size
    num_classes = args.num_classes
    end_epoch = args.end_epoch
    if_scratch = args.if_scratch
    
    gpu_ids = args.gpu_ids
    num_workers = args.num_workers
    checkpoint_dir = args.checkpoint_dir
    log_name = args.log_name
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    
    # Logger 설정
    if not os.path.exists('output/log'):
        os.makedirs('output/log')
    logger = Logger(dir='output/log', name='SleepVST' + '.train')
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logger.info(f"Checkpoint path: {checkpoint_path}")
    
    # 모델 초기화
    model = SleepVST().cuda()
    
    # 데이터셋 로드 (테스트셋은 모든 모드에서 필요)
    shhs_test = SHHS(split='test')
    mesa_test = MESA(split='test')

    shhs_test_loader = DataLoader(
        shhs_test, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    mesa_test_loader = DataLoader(
        mesa_test, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # 모드에 따라 다른 동작 수행
    if mode in ['train', 'train_and_test']:
        # 학습 모드일 경우 wandb 설정
        wandb.login(key="7470ee66d23ffbe8f8ccbbf5bfeeb19966a69540")
        run = wandb.init(
            project="SleepVST",
            entity="heohun-seoul-national-university"
        )
        
        wandb_config ={
            "epochs": end_epoch,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "seq_len": seq_len,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers
        }
        wandb.config.update(wandb_config)
        wandb.run.name = log_name
        
        # 학습 데이터셋 로드
        if if_scratch:
            shhs_train = SHHS(split='train')
            mesa_train = MESA(split='train')
            train_dataset = torch.utils.data.ConcatDataset([shhs_train, mesa_train])
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            
            shhs_val = SHHS(split='val')
            mesa_val = MESA(split='val')
            val_dataset = torch.utils.data.ConcatDataset([shhs_val, mesa_val])
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        
        # 손실 함수, 옵티마이저 설정
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 체크포인트 로드 또는 새로 시작
        if not if_scratch and os.path.exists(checkpoint_path):
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
        
        logger.info("Training started.")
        training_start_time = time.time()
        
        for epoch in range(begin_epoch, end_epoch):
            start_time = time.time()
            logger.info(f"\n===== Epoch {epoch + 1}/{end_epoch} =====")
            
            train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion)
            
            wandb.log(
                data={
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                },
                step=epoch,
            )
            
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            
            val_loss, val_accuracy, val_f1, kappa, val_cm = evaluate(model, val_loader, criterion)
            
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
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping:
                    logger.info("Early stopping triggered.")
                    break
        
        total_time = time.time() - training_start_time
        logger.info(f"Training completed in {total_time:.2f} seconds.")
                
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info("Best model loaded.")
    
    # 테스트 모드 (학습 후 테스트 또는 테스트만)
    if mode in ['test', 'train_and_test']:
        # 테스트 전용 모드일 경우 모델 로드
        if mode == 'test':
            logger.info(f"Test mode: Loading model from {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model loaded successfully")
            else:
                logger.error(f"No checkpoint found at {checkpoint_path}. Cannot test.")
                return
        
        # 테스트 로거 설정
        test_logger = Logger(dir='output/log', name='SleepVST' + '.test')
        
        criterion = nn.CrossEntropyLoss().cuda()
        
        # 테스트 실행
        test_logger.info("----SHHS TEST----")
        shhs_loss, shhs_accuracy, shhs_f1, shhs_kappa, shhs_cm = evaluate(model, shhs_test_loader, criterion)
        test_logger.info(f"SHHS Test Loss: {shhs_loss:.4f}, Acc: {shhs_accuracy:.4f}, F1: {shhs_f1:.4f}, Kappa: {shhs_kappa:.4f}")
        test_logger.info(f"SHHS Confusion Matrix:\n {shhs_cm}")
        
        test_logger.info("----MESA TEST----")
        mesa_loss, mesa_accuracy, mesa_f1, mesa_kappa, mesa_cm = evaluate(model, mesa_test_loader, criterion)
        test_logger.info(f"MESA Test Loss: {mesa_loss:.4f}, Acc: {mesa_accuracy:.4f}, F1: {mesa_f1:.4f}, Kappa: {mesa_kappa:.4f}")
        test_logger.info(f"MESA Confusion Matrix:\n {mesa_cm}")
        
        # wandb에 테스트 결과 기록 (학습 모드와 함께했을 경우만)
        if mode == 'train_and_test' and 'run' in locals():
            wandb.log({
                "shhs_test_loss": shhs_loss,
                "shhs_test_accuracy": shhs_accuracy,
                "shhs_test_f1": shhs_f1,
                "shhs_test_kappa": shhs_kappa,
                "mesa_test_loss": mesa_loss,
                "mesa_test_accuracy": mesa_accuracy,
                "mesa_test_f1": mesa_f1,
                "mesa_test_kappa": mesa_kappa,
            })
            # wandb 종료
            wandb.finish()
    
    logger.info("Process completed.")
    
if __name__ == "__main__":    
    main()