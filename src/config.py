from dataclasses import dataclass
import torch
from typing import Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """TFLOP 모델 설정"""
    feature_dim: int = 1024      # 모델의 hidden dimension 크기
    total_sequence_length: int = 1376  # 최대 시퀀스(토큰) 길이
    image_size: int = 768       # 입력 이미지의 크기 (height=width)  # 논문 768
    temperature: float = 0.1    # Layout Pointer의 softmax temperature
    
    # Swin Transformer 설정
    swin_model_name: str = "microsoft/swin-base-patch4-window7-224"
    # swin_model_name: str = "microsoft/swin-tiny-patch4-window7-224"
    # swin_model_name: str = "microsoft/swinv2-tiny-patch4-window16-256"
    
    # BART model 설정
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    dropout: float = 0.1

    
    # Loss weights (논문 Section 3.7)
    lambda_cls: float = 1.0
    lambda_ptr: float = 1.0
    lambda_empty_ptr: float = 1.0
    lambda_row_contr: float = 0.5
    lambda_col_contr: float = 0.5

    scale_embedding: bool = True
    def to_dict(self):
        """설정을 딕셔너리로 변환"""
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}

@dataclass
class TrainingConfig:
    """학습 설정"""
    exp_name: str = "TFLOP"
    use_wandb: bool = False
    
    # Resume training
    resume_training: bool = False
    resume_checkpoint_path: Optional[str] = None  # 재개할 체크포인트 경로
    
    # Data
    data_dir: str = "./data/pubtabnet"
    train_split: str = "train"
    val_split: str = "val"
    
    # Training
    total_steps: int = 100000
    eval_steps: int = 1000
    save_steps: int = 2000
    checkpoint_dir: str = "./checkpoints"
    
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    warmup_steps: int = 0
    max_grad_norm: float = 0.5
    
    # Device & Hardware
    gpu_id: int = 0
    num_workers: int = 4
    pin_memory: bool = True
    
    # Trainer 설정
    accelerator: str = "gpu"
    devices: list = None
    strategy: str = "auto"
    precision: str = "32"
    accumulate_grad_batches: int = 8
    gradient_clip_val: float = 0.5
    num_sanity_val_steps: int = 2
    
    def __post_init__(self):
        # devices 설정
        if self.devices is None:
            self.devices = [self.gpu_id] if torch.cuda.is_available() else None
        
        # accelerator 설정
        if not torch.cuda.is_available():
            self.accelerator = "cpu"
            self.devices = None
        
        # device 설정
        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # Logging
    log_every_n_steps: int = 10
    
    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}

@dataclass
class InferenceConfig:
    """추론 관련 설정"""
    
    # 모델 관련
    checkpoint_path: str = 'checkpoints/best_model.pt'
    device: str = 'cuda'
    
    # 데이터 관련
    batch_size: int = 16
    num_workers: int = 4
    
    # 추론 관련
    use_beam_search: bool = True
    beam_size: int = 5
    max_length: int = 1376