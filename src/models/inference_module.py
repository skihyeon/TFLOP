import torch
import torch.nn as nn
from typing import Dict, Any
from .tflop import TFLOP

class TFLOPInferenceModule(nn.Module):
    """Inference 전용 TFLOP 모듈"""
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
        # TFLOP 모델 초기화 (inference mode)
        self.model = TFLOP(config, inference_mode=True)
        
    def load_from_checkpoint(self, checkpoint_path: str, device: torch.device):
        """체크포인트에서 가중치 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # state_dict에서 'model.' 접두사 제거 (Lightning 모듈에서 저장된 형태)
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                state_dict[key[6:]] = value  # 'model.' 제거
        
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """추론 전용 forward"""
        return self.model(batch) 