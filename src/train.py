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
    
    model = TFLOPLightningModule(model_config=model_config, train_config=train_config, inference_mode=False)
    
    datamodule = TableDataModule(
        data_dir=train_config.data_dir,
        model_config=model_config,
        train_config=train_config
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=exp_dir / "checkpoints",
            filename=f'{train_config.exp_name}'+'_{step}',
            save_top_k=3,
            monitor='val/loss',
            mode='min',
            every_n_train_steps=train_config.save_steps if train_config.save_steps > 0 else None,
        ),
        ValidationVisualizationCallback(viz_dir=exp_dir / "visualizations")
    ]
    
    if train_config.use_wandb:
        logger = WandbLogger(
            project="TFLOP",
            name=timestamp,
            config={
                "model_config": model_config.to_dict(),
                "train_config": train_config.to_dict()
            }
        )
        callbacks.append(LearningRateMonitor(logging_interval='step'))
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
        max_steps=train_config.total_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[train_config.gpu_id] if torch.cuda.is_available() else None,
        accumulate_grad_batches=train_config.gradient_accumulation_steps,
        gradient_clip_val=train_config.max_grad_norm,
        val_check_interval=train_config.eval_steps,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=None,
        max_epochs=-1,
        limit_train_batches=train_config.total_steps,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=train_config.log_every_n_steps if logger else None,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    if train_config.resume_training and train_config.resume_checkpoint_path:
        print(f"Resuming training from checkpoint: {train_config.resume_checkpoint_path}")
        trainer.fit(
            model, 
            datamodule=datamodule,
            ckpt_path=train_config.resume_checkpoint_path
        )
    else:
        trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()