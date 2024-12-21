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
    
    
    feature_dim: int = 1024      # 모델의 hidden dimension 크기       # 논문 1024
    total_sequence_length: int = 1376  # 최대 시퀀스(토큰) 길이        # 논문 1376 # bart position embedding최대 길이 1024
    otsl_max_length: int = 100
    image_size: int = 768  
    # swin_model_name: str = "microsoft/swin-base-patch4-window7-224"
    swin_model_name: str = "microsoft/swinv2-base-patch4-window8-256"
    
    
    
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
    exp_name: str = "TFLOP_base"
    use_wandb: bool = False
    
    # Resume training
    resume_training: bool = True
    resume_checkpoint_path: Optional[str] = "/mnt/hdd1/sgh/TFLOP/src/checkpoints/20241220_1832_TFLOP_base/checkpoints/TFLOP_base_epoch=6.ckpt"
    # resume_checkpoint_path: Optional[str] = "/mnt/hdd1/sgh/TFLOP/src/checkpoints/20241211_1551_TFLOP_tiny/checkpoints/TFLOP_epoch=130.ckpt"
    
    # Data
    data_dir: str = "./data/pubtabnet"
    train_split: str = "train"
    val_split: str = "val"
    
    # Training
    num_epochs: int = 1000
    checkpoint_dir: str = "./checkpoints"
    
    batch_size: int = 3
    accumulate_grad_batches: int = 12
    learning_rate: float = 1e-4
    
    # Device & Hardware
    gpu_id: int = 1  # 기본 GPU ID (단일 GPU 사용시)
    devices: list = None  # for multi-GPU
    accelerator: str = "gpu"
    strategy: str = "ddp"  # 미사용 파라미터 감지 활성화
    sync_batchnorm: bool = True  # DDP에서 배치 정규화 동기화
    replace_sampler_ddp: bool = True  # DDP에서 샘플러 자동 교체
    
    num_workers: int = 12
    pin_memory: bool = True
    
    # Trainer 설정
    precision: str = "32-true"
    gradient_clip_val: float = 1.0
    num_sanity_val_steps: int = 2
    check_val_every_n_epoch: int = 1
    
    def __post_init__(self):
        # devices 설정 수정
        if self.devices is None:
            self.devices = [0]  
        
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
class TestConfig:
    """추론 관련 설정"""
    data_dir: str = "./data/pubtabnet"
    train_split: str = "train"
    val_split: str = "val"
    
    # 모델 관련
    checkpoint_path: str = './checkpoints/20241218_2313_TFLOP_tiny/checkpoints/TFLOP_tiny_epoch=203.ckpt'
    device: str = 'cuda'
    gpu_id: int = 3
    
    feature_dim: int = 768      # 모델의 hidden dimension 크기       # 논문 1024
    total_sequence_length: int = 512  # 최대 시퀀스(토큰) 길이        # 논문 1376 # bart position embedding최대 길이 1024
    otsl_max_length: int = 30
    image_size: int = 224  
    swin_model_name: str = "microsoft/swin-tiny-patch4-window7-224"
    
    # feature_dim: int = 768      # 모델의 hidden dimension 크기       # 논문 1024
    # total_sequence_length: int = 1024  # 최대 시퀀스(토큰) 길이        # 논문 1376 # bart position embedding최대 길이 1024
    # otsl_max_length: int = 30
    # image_size: int = 672  
    # swin_model_name: str = "microsoft/swin-base-patch4-window7-224"
    
    
    image_path: str = './infer_images'
    output_path: str = './infer_result'
    
    batch_size: int = 1
    num_workers: int = 12
    
    
    
    # Loss weights (논문 Section 3.7)
    lambda_cls: float = 1.0
    lambda_ptr: float = 1.0
    lambda_empty_ptr: float = 1.0
    lambda_row_contr: float = 0.5
    lambda_col_contr: float = 0.5

    temperature: float = 0.1

@dataclass
class PredictConfig:
    """Predict 관련 설정"""
    
    # 모델 관련
    checkpoint_path: str = '/mnt/hdd1/sgh/TFLOP/src/checkpoints/20241219_1008_TFLOP_base/checkpoints/TFLOP_base_epoch=51.ckpt'
    
    # 모델 설정
    feature_dim: int = 768
    total_sequence_length: int = 1024
    otsl_max_length: int = 30
    image_size: int = 672
    swin_model_name: str = "microsoft/swin-base-patch4-window7-224"
    temperature: float = 0.1
    
    # 입출력 경로
    input_dir: str = './infer_images'  # 입력 이미지 디렉토리
    output_dir: str = './infer_results'  # 결과 저장 디렉토리
    
    # GPU 설정
    gpu_id: int = 3  # 사용할 GPU ID
    precision: str = "32-true"
    
    # DataLoader 설정
    batch_size: int = 1
    num_workers: int = 12
    pin_memory: bool = True
    
    
    # Loss weights (논문 Section 3.7)
    lambda_cls: float = 1.0
    lambda_ptr: float = 1.0
    lambda_empty_ptr: float = 1.0
    lambda_row_contr: float = 0.5
    lambda_col_contr: float = 0.5
    