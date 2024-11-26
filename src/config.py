from dataclasses import dataclass
import torch
from typing import Optional

@dataclass
class TFLOPConfig:
    """TFLOP 모델 설정"""
    vocab_size: int = 50265
    feature_dim: int = 768
    max_seq_length: int = 1376
    image_size: int = 768
    temperature: float = 0.1
    
    # Swin Transformer 설정
    swin_model_name: str = "microsoft/swin-base-patch4-window7-224"
    
    # Special tokens (OTSLTokenizer와 동일하게 설정)
    pad_token_id: int = 50262  # vocab_size - 3
    unk_token_id: int = 50263  # vocab_size - 2
    eos_token_id: int = 50264  # vocab_size - 1
    bos_token_id: int = 50261  # vocab_size - 4
    
    # BART tokenizer 설정
    pad_token_id: int = 1  # BART default
    bos_token_id: int = 0  # BART default
    eos_token_id: int = 2  # BART default
    decoder_start_token_id: int = 2  # BART default
    
    # BART model 설정
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_ffn_dim: int = 3072
    decoder_ffn_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    max_position_embeddings: int = 1024
    
    # Loss weights (논문 Section 3.7)
    lambda_cls: float = 1.0
    lambda_ptr: float = 1.0
    lambda_empty_ptr: float = 1.0
    lambda_row_contr: float = 0.5
    lambda_col_contr: float = 0.5

@dataclass
class TrainingConfig:
    """학습 설정"""
    # Data
    data_dir: str = "./data/pubtabnet"
    train_split: str = "train"
    val_split: str = "val"
    
    # Training
    num_epochs: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Device
    device: torch.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    # Logging & Checkpoints
    log_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 5000
    checkpoint_dir: str = "./checkpoints"
    
    # Will be set in train.py
    steps_per_epoch: Optional[int] = None

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