import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Optional
import torch
from config import TrainingConfig, TFLOPConfig
from trainer import TFLOPTrainer
from datasets import TableDataset, create_dataloader

def main() -> None:
    # 설정 로드
    model_config = TFLOPConfig()
    train_config = TrainingConfig()
    
    # 데이터셋 생성 (논문 Section 4.1)
    train_dataset = TableDataset(
        data_dir=train_config.data_dir,
        split=train_config.train_split,
        max_seq_length=model_config.max_seq_length,
        image_size=model_config.image_size,
    )
    
    val_dataset = TableDataset(
        data_dir=train_config.data_dir,
        split=train_config.val_split,
        max_seq_length=model_config.max_seq_length,
        image_size=model_config.image_size,
    )
    
    # steps_per_epoch 계산
    train_config.steps_per_epoch = len(train_dataset) // (
        train_config.batch_size * train_config.gradient_accumulation_steps
    )
    
    # 데이터로더 생성
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 트레이너 초기화
    trainer = TFLOPTrainer(
        model_config=model_config,
        train_config=train_config
    )
    
    # 학습 시작
    trainer.train(train_dataloader, val_dataloader)

if __name__ == '__main__':
    main() 