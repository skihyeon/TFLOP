from dataclasses import dataclass
import torch
from typing import Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """TFLOP 모델의 기본 설정"""
    feature_dim: int = 1024
    total_sequence_length: int = 1376
    otsl_max_length: int = 200
    image_size: int = 768
    swin_model_name: str = "microsoft/swinv2-base-patch4-window8-256"
    
    temperature: float = 0.1
    dropout: float = 0.1

    # Loss weights
    lambda_cls: float = 1.0
    lambda_ptr: float = 1.0
    lambda_empty_ptr: float = 1.0
    lambda_row_contr: float = 0.5
    lambda_col_contr: float = 0.5

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}

@dataclass
class TrainingConfig:
    """학습 관련 설정"""
    exp_name: str = "TFLOP_len200_bf16"
    use_wandb: bool = True
    
    # Resume training
    resume_training: bool = True
    resume_checkpoint_path: Optional[str] = "/mnt/hdd1/sgh/TFLOP/src/checkpoints/20250103_1823_TFLOP_len200_bf16/step_checkpoints/TFLOP_noShift_step=49999.ckpt"
    
    # Data
    data_dir: str = "./data/pubtabnet"
    train_split: str = "train"
    val_split: str = "val"
    
    # Training
    num_epochs: int = 1000
    checkpoint_dir: str = "./checkpoints"
    batch_size: int = 4
    accumulate_grad_batches: int = 4
    learning_rate: float = 1e-4
    
    # Hardware
    gpu_id: int = 3
    devices: list = None        # dataclass 는 기본적으로 list 안받음
    accelerator: str = "gpu"
    strategy: str = "ddp"
    sync_batchnorm: bool = True
    replace_sampler_ddp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Trainer
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 0.5
    num_sanity_val_steps: int = 2
    check_val_every_n_epoch: int = 5
    
    def __post_init__(self):
        if self.devices is None:
            self.devices = [1,2,3]
        if not torch.cuda.is_available():
            self.accelerator = "cpu"
            self.devices = None
        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}

@dataclass
class PredictConfig:
    """예측 전용 설정"""
    # 모델 체크포인트
    checkpoint_path: str = "/mnt/hdd1/sgh/TFLOP/src/checkpoints/20250103_1823_TFLOP_len200_bf16/checkpoints/TFLOP_len200_bf16_best.pt"
    
    # 입출력 설정
    input_dir: str = './infer_images'
    output_dir: str = './infer_results'
    
    # 하드웨어 설정
    gpu_id: int = 0
    precision: str = "bf16-mixed"
    num_workers: int = 4
    pin_memory: bool = True
    batch_size: int = 1
    
    # OCR 결과 (선택사항)
    ocr_results: Optional[dict] = None
    
    def __post_init__(self):
        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}
    