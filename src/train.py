import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from datetime import datetime
import torch

from config import ModelConfig, TrainingConfig
from models import TFLOPLightningModule
from datasets.datamodule import TableDataModule
from utils.callbacks import ValidationVisualizationCallback


def main():
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    if train_config.resume_training and train_config.resume_checkpoint_path:
        checkpoint_dir = Path(train_config.resume_checkpoint_path).parent.parent
        exp_dir = checkpoint_dir
    else:
        exp_dir = Path(train_config.checkpoint_dir) / (timestamp + "_" + train_config.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
    
    datamodule = TableDataModule(
        data_dir=train_config.data_dir,
        model_config=model_config,
        train_config=train_config
    )
    
    model = TFLOPLightningModule(
        model_config=model_config,
        train_config=train_config,
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=exp_dir / "checkpoints",
            filename=f'{train_config.exp_name}' + '_{epoch}',
            save_top_k=-1,  # 모든 체크포인트 저장
            every_n_epochs=1,  # 매 epoch마다 저장
            save_on_train_epoch_end=True,
            save_last=True,  # 마지막 체크포인트는 따로 저장
            # save_weights_only=True  # 모델 가중치만 저장
        ),
        ValidationVisualizationCallback(viz_dir=exp_dir / "visualizations")
    ]
    
    if train_config.use_wandb:
        logger = WandbLogger(
            project="TFLOP",
            name=timestamp + "_" + train_config.exp_name,
            config={
                "model_config": model_config.to_dict(),
                "train_config": train_config.to_dict()
            }
        )
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    else:
        logger = False
    
    if train_config.resume_training:
        # 기존 config 파일들 확인
        existing_configs = list(exp_dir.glob("config*.txt"))
        next_num = len(existing_configs) + 1
        config_path = exp_dir / f"config_{next_num}.txt"
    else:
        config_path = exp_dir / "config.txt"
        
    with open(config_path, "w") as f:
        # 모델 설정 저장
        f.write("=" * 50 + "\n")
        f.write("Model Configuration\n") 
        f.write("=" * 50 + "\n")
        model_config_dict = model_config.to_dict()
        for key, value in model_config_dict.items():
            f.write(f"{key:25s}: {value}\n")
            
        # 학습 설정 저장 
        f.write("\n" + "=" * 50 + "\n")
        f.write("Training Configuration\n")
        f.write("=" * 50 + "\n")
        train_config_dict = train_config.to_dict()
        for key, value in train_config_dict.items():
            f.write(f"{key:25s}: {value}\n")
    
    trainer = pl.Trainer(
        max_epochs=train_config.num_epochs,  # step 대신 epoch 사용
        accelerator=train_config.accelerator,
        devices=train_config.devices,
        strategy=train_config.strategy,
        precision=train_config.precision,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=train_config.gradient_clip_val,
        check_val_every_n_epoch=train_config.check_val_every_n_epoch,  # 매 epoch마다 validation 수행
        num_sanity_val_steps=train_config.num_sanity_val_steps,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        deterministic=True,
        benchmark=False
    )

    if train_config.resume_training and train_config.resume_checkpoint_path:
        print(f"Resuming training from checkpoint: {train_config.resume_checkpoint_path}")
        # 체크포인트는 trainer.fit에서만 로딩
        trainer.fit(model, datamodule=datamodule, ckpt_path=train_config.resume_checkpoint_path)
    else:
        trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    from setproctitle import setproctitle
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    setproctitle(f"TFLOP_{timestamp}")
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    main()