# test.py
import pytorch_lightning as pl
import torch
from pathlib import Path
from datetime import datetime
import json

from config import ModelConfig, TrainingConfig
from models import TFLOPLightningModule
from datasets.dataset import TableDataset
from datasets.dataloader import create_dataloader
from models.otsl_tokenizer import OTSLTokenizer


def main():
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # 1. GPU 설정 명시
    if torch.cuda.is_available():
        torch.cuda.set_device(train_config.gpu_id)
        device = torch.device(f'cuda:{train_config.gpu_id}')
    else:
        device = torch.device('cpu')
    
    # 2. 토크나이저 초기화
    tokenizer = OTSLTokenizer(
        otsl_sequence_length=model_config.otsl_max_length
    )
    
    # 2. Validation 데이터로 테스트 데이터셋 생성
    # TODO: 추후 실제 테스트 데이터셋으로 교체 필요
    test_dataset = TableDataset(
        data_dir=train_config.data_dir,
        split='val',  # 현재는 validation 데이터 사용
        image_size=model_config.image_size,
        tokenizer=tokenizer
    )
    
    # 3. 테스트용 데이터로더 생성
    test_loader = create_dataloader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        drop_last=False
    )
    
    # 4. 체크포인트에서 모델 로드 및 GPU 설정
    model = TFLOPLightningModule.load_from_checkpoint(
        train_config.resume_checkpoint_path,
        model_config=model_config,
        train_config=train_config,
    )
    model = model.to(device)
    
    # 5. 테스트를 위한 Trainer 설정
    trainer = pl.Trainer(
        accelerator='gpu',  # 명시적으로 GPU 설정
        devices=[train_config.gpu_id],
        # strategy='single_device',
        precision=train_config.precision,
        logger=False,
        enable_checkpointing=False,
        deterministic=True
    )
    
    # 6. 테스트 실행
    results = trainer.test(model, dataloaders=test_loader)
    
    # 7. 결과 저장
    checkpoint_dir = Path(train_config.resume_checkpoint_path).parent.parent
    output_dir = checkpoint_dir / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_file = output_dir / f"test_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "test_config": {
                "checkpoint_path": train_config.resume_checkpoint_path,
                "data_split": "validation",  # 현재는 validation 데이터 사용
                "num_samples": len(test_dataset),
                # "model_config": model_config.to_dict(),
                # "train_config": train_config.to_dict()
            },
            "metrics": results[0]
        }, f, indent=2)
    
    print(f"Test results saved to {results_file}")


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    config = TrainingConfig()
    torch.cuda.set_device(config.gpu_id)  # 명시적으로 GPU 설정
    main()