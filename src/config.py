from dataclasses import dataclass
import torch
from typing import Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """TFLOP 모델 설정"""
    # feature_dim: int = 768      # 모델의 hidden dimension 크기       # 논문 1024
    # total_sequence_length: int = 512  # 최대 시퀀스(토큰) 길이        # 논문 1376 # bart position embedding최대 길이 1024
    # otsl_max_length: int = 30
    # image_size: int = 224  
    # swin_model_name: str = "microsoft/swin-tiny-patch4-window7-224"
    
    
    feature_dim: int = 768      # 모델의 hidden dimension 크기       # 논문 1024
    total_sequence_length: int = 1024  # 최대 시퀀스(토큰) 길이        # 논문 1376 # bart position embedding최대 길이 1024
    otsl_max_length: int = 30
    image_size: int = 672    
    swin_model_name: str = "microsoft/swin-base-patch4-window7-224"
    
    
    
    temperature: float = 0.1    # Layout Pointer의 softmax temperature
    
    dropout: float = 0.1

    # Loss weights (논문 Section 3.7)
    lambda_cls: float = 1.0
    lambda_ptr: float = 1.0
    lambda_empty_ptr: float = 1.0
    lambda_row_contr: float = 0.5
    lambda_col_contr: float = 0.5

    def to_dict(self):
        """설정을 딕셔너리로 변환"""
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}

@dataclass
class TrainingConfig:
    """학습 설정"""
    exp_name: str = "TFLOP"
    use_wandb: bool = False
    gpu_id: int = 2
    
    # Resume training
    resume_training: bool = False
    resume_checkpoint_path: Optional[str] = "/mnt/hdd1/sgh/TFLOP/src/checkpoints/20241216_1450_TFLOP_tiny_224_len30_deep_layoutEncoder/checkpoints/TFLOP_tiny_672_len30_deep_layoutEncoder_epoch=20.ckpt"
    # resume_checkpoint_path: Optional[str] = "/mnt/hdd1/sgh/TFLOP/src/checkpoints/20241211_1551_TFLOP_tiny/checkpoints/TFLOP_epoch=130.ckpt"
    
    # Data
    data_dir: str = "./data/pubtabnet"
    train_split: str = "train"
    val_split: str = "val"
    
    # Training
    num_epochs: int = 200
    checkpoint_dir: str = "./checkpoints"
    
    batch_size: int = 2 
    accumulate_grad_batches: int = 16
    learning_rate: float = 1e-3
    
    # Device & Hardware
    gpu_id: int = 1  # 기본 GPU ID (단일 GPU 사용시)
    devices: list = None  # 이 부분을 수정
    accelerator: str = "gpu"
    strategy: str = "ddp"  # multi-GPU 사용시 "ddp"로 설정
    
    num_workers: int = 12
    pin_memory: bool = True
    
    # Trainer 설정
    precision: str = "32"
    gradient_clip_val: float = 0.5
    num_sanity_val_steps: int = 2
    check_val_every_n_epoch: int = 1
    
    def __post_init__(self):
        # devices 설정 수정
        if self.devices is None:
            self.devices = [0, 1]  # 0번과 1번 GPU만 사용하도록 설정
        
        # accelerator 설정
        if not torch.cuda.is_available():
            self.accelerator = "cpu"
            self.devices = None
        
        # device 설정
        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}

@dataclass
class InferenceConfig:
    """추론 관련 설정"""
    
    # 모델 관련
    checkpoint_path: str = '/mnt/hdd1/sgh/TFLOP/src/checkpoints/20241216_1450_TFLOP_tiny_224_len30_deep_layoutEncoder/checkpoints/TFLOP_tiny_224_len30_deep_layoutEncoder_epoch=44.ckpt'
    device: str = 'cuda'
    
    feature_dim: int = 768      # 모델의 hidden dimension 크기       # 논문 1024
    total_sequence_length: int = 512  # 최대 시퀀스(토큰) 길이        # 논문 1376 # bart position embedding최대 길이 1024
    otsl_max_length: int = 30
    image_size: int = 224  
    swin_model_name: str = "microsoft/swin-tiny-patch4-window7-224"
    
    
    image_path: str = './infer_images'
    output_path: str = './infer_result'
    
    batch_size: int = 1
    num_workers: int = 32