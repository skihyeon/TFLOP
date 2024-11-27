from pathlib import Path
import sys
import os
# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import TrainingConfig, TFLOPConfig 
from trainer import TFLOPTrainer
from datasets import TableDataset, create_dataloader
from utils import init_wandb

def create_dataset(config: TrainingConfig, model_config: TFLOPConfig, split: str) -> TableDataset:
    """데이터셋 생성 함수"""
    return TableDataset(
        data_dir=config.data_dir,
        split=split,
        max_seq_length=model_config.max_seq_length,
        image_size=model_config.image_size,
    )


def main() -> None:
    # 설정 초기화
    model_config = TFLOPConfig()
    train_config = TrainingConfig()
    
    # 데이터셋 생성
    train_dataset = create_dataset(train_config, model_config, train_config.train_split)
    val_dataset = create_dataset(train_config, model_config, train_config.val_split)
    
    # 데이터로더 생성 (balanced dataloader 사용)
    dataloader_kwargs = dict(batch_size=train_config.batch_size, num_workers=4)
    train_dataloader = create_dataloader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = create_dataloader(val_dataset, shuffle=False, **dataloader_kwargs)
    
    # 학습 실행
    trainer = TFLOPTrainer(model_config=model_config, train_config=train_config)
    
    # wandb 초기화
    if train_config.use_wandb:
        init_wandb(model_config, train_config)
    
    trainer.train(train_dataloader, val_dataloader)


if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = "0a7cca3a906f5c34a06fe63623461725e2278ef3"
    os.environ['WANDB_ENTITY'] = "hero981001"
    main()
    
    
    
    
