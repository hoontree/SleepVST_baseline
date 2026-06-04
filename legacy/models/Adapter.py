import torch
import torch.nn as nn
from models.SleepVST import SleepVST

class AdapterLayer(nn.Module):
    """
    Adapter 계층: 적은 파라미터로 모델을 효율적으로 조정할 수 있게 함
    """
    def __init__(self, d_model, bottleneck_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 초기화: up_proj는 0에 가깝게 초기화하여 학습 초기에 원래 모델의 동작을 보존
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = x + residual  # 잔차 연결
        x = self.layer_norm(x)
        return x

class SleepVST_Adapter(nn.Module):
    """
    기존 SleepVST 모델에 Adapter 계층을 추가하는 모델
    """
    def __init__(self, pretrained_model_path, num_classes=5, freeze_base=True):
        super().__init__()
        # 기존 모델 로드
        self.base_model = SleepVST()
        checkpoint = torch.load(pretrained_model_path)
        self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 기존 모델 파라미터 고정 (선택적)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        d_model = self.base_model.classifier.in_features
        
        # Adapter 계층 추가
        self.adapter = AdapterLayer(d_model)
        
        # 새로운 분류기
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x_hw, x_bw):
        # 기본 모델에서 특징 추출
        features = self.base_model.forward_features(x_hw, x_bw)
        
        # Adapter 계층 통과
        adapted_features = self.adapter(features)
        
        # 분류기
        logits = self.classifier(adapted_features)
        
        return logits
    
    def forward_features(self, x_hw, x_bw):
        # 기본 모델에서 특징 추출
        features = self.base_model.forward_features(x_hw, x_bw)
        
        # Adapter 계층 통과
        adapted_features = self.adapter(features)
        
        return adapted_features