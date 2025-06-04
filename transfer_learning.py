import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from models.SleepVST import SleepVST
from tqdm import tqdm
from utils.customlogger import Logger
from utils.util import MetricsTracker, AverageMeter
import wandb
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='SleepVST Transfer Learning')
    parser.add_argument('--source_ckpt', type=str, required=True, help='Path to source model checkpoint')
    parser.add_argument('--target_dataset', type=str, required=True, help='Target dataset name')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder layers')
    parser.add_argument('--freeze_transformer', action='store_true', help='Freeze transformer layers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of target classes')
    parser.add_argument('--output_dir', type=str, default='transfer_checkpoints', help='Output directory')
    parser.add_argument('--run_name', type=str, default=time.strftime('%Y%m%d_%H%M%S'),
                        help='Unique run name for logs and checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    return parser.parse_args()

def load_target_dataset(target_dataset, batch_size, num_workers=4):
    # 여기서 타겟 데이터셋을 로드하는 코드 구현
    # 예시: 다른 수면 데이터셋 또는 새로운 데이터셋
    if target_dataset == 'kvss':
        from data.KVSS import KVSS
        train_dataset = KVSS(split='train')
        val_dataset = KVSS(split='val')
        test_dataset = KVSS(split='test')
    else:
        raise ValueError(f"Unknown dataset: {target_dataset}")
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    return dataloaders

def freeze_layers(model, freeze_encoder=False, freeze_transformer=False):
    """
    선택적으로 모델의 일부 계층을 고정(freeze)합니다.
    """
    if freeze_encoder:
        # Waveform encoder 계층 고정
        for param in model.heart_encoder.parameters():
            param.requires_grad = False
        for param in model.breath_encoder.parameters():
            param.requires_grad = False
        for param in model.proj_heart.parameters():
            param.requires_grad = False
        for param in model.proj_breath.parameters():
            param.requires_grad = False
            
    if freeze_transformer:
        # Transformer 계층 고정
        for param in model.transformer.parameters():
            param.requires_grad = False

def transfer_learning(args):
    # 로거 설정
    logger = Logger(dir='output/log', name=f'SleepVST_transfer_{args.target_dataset}', run_name=args.run_name)
    
    # 모델 로드
    logger.info(f"Loading source model from {args.source_ckpt}")
    model = SleepVST(num_classes=args.num_classes).cuda()
    
    # 소스 모델 체크포인트 로드
    checkpoint = torch.load(args.source_ckpt)
    
    # 분류기를 제외한 가중치만 로드 (분류기의 클래스 수가 다를 수 있음)
    source_state_dict = checkpoint['model_state_dict']
    target_state_dict = model.state_dict()
    
    # 분류기를 제외한 가중치만 복사
    for k in source_state_dict:
        if 'classifier' not in k:
            target_state_dict[k] = source_state_dict[k]
    
    model.load_state_dict(target_state_dict, strict=False)
    logger.info("Loaded pre-trained weights (excluding classifier)")
    
    # 선택적으로 계층 고정
    freeze_layers(model, freeze_encoder=args.freeze_encoder, freeze_transformer=args.freeze_transformer)
    
    # 학습 가능한 파라미터 로깅
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    # 데이터 로더 설정
    dataloaders = load_target_dataset(args.target_dataset, args.batch_size)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # wandb 설정
    if args.use_wandb:
        wandb.init(
            project="SleepVST-Transfer",
            name=f"{args.target_dataset}_{'frozen' if args.freeze_encoder else 'unfrozen'}"
        )
        wandb.config.update(vars(args))
    
    # 체크포인트 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 학습 루프
    best_val_loss = float('inf')
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        # 학습
        model.train()
        train_loss = AverageMeter()
        correct = 0
        total = 0
        
        for batch in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{args.epochs}"):
            x_hw = batch['x_hw'].cuda()
            x_bw = batch['x_bw'].cuda()
            labels = batch['label'].cuda().long()
            
            optimizer.zero_grad()
            logits = model(x_hw, x_bw)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            train_loss.update(loss.item(), labels.numel())
        
        train_acc = correct / total
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc:.4f}")
        
        # 검증
        model.eval()
        val_metrics = MetricsTracker()
        val_loss = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(dataloaders['val'], desc="Validating"):
                x_hw = batch['x_hw'].cuda()
                x_bw = batch['x_bw'].cuda()
                labels = batch['label'].cuda().long()
                
                logits = model(x_hw, x_bw)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                val_loss.update(loss.item(), labels.numel())
                preds = logits.argmax(-1)
                val_metrics.update(preds, labels)
        
        val_acc, val_f1, val_kappa, _, _ = val_metrics.compute()
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss.avg,
                'train_acc': train_acc,
                'val_loss': val_loss.avg,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_kappa': val_kappa,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # 스케줄러 업데이트
        scheduler.step(val_loss.avg)
        
        # 체크포인트 저장
        if val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_val_acc = val_acc
            
            # 체크포인트 저장
            checkpoint_path = os.path.join(args.output_dir, f"{args.run_name}_best_model_{args.target_dataset}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss.avg,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'args': args
            }, checkpoint_path)
            logger.info(f"Saved best model with val_loss: {val_loss.avg:.4f}, val_acc: {val_acc:.4f}")
    
    # 테스트 평가
    logger.info("Evaluating on test set...")
    model.eval()
    test_metrics = MetricsTracker()
    test_loss = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloaders['test'], desc="Testing"):
            x_hw = batch['x_hw'].cuda()
            x_bw = batch['x_bw'].cuda()
            labels = batch['label'].cuda().long()
            
            logits = model(x_hw, x_bw)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            test_loss.update(loss.item(), labels.numel())
            preds = logits.argmax(-1)
            test_metrics.update(preds, labels)
    
    test_acc, test_f1, test_kappa, test_cm, test_cm_norm = test_metrics.compute()
    
    logger.info(f"Test Results - Loss: {test_loss.avg:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Kappa: {test_kappa:.4f}")
    logger.info(f"Confusion Matrix:\n{test_cm}")
    
    if args.use_wandb:
        wandb.log({
            'test_loss': test_loss.avg,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_kappa': test_kappa
        })
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    transfer_learning(args)