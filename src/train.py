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
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(train_config.checkpoint_dir) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    model = TFLOPLightningModule(model_config=model_config, train_config=train_config)
    
    datamodule = TableDataModule(
        data_dir=train_config.data_dir,
        model_config=model_config,
        train_config=train_config
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=exp_dir / "checkpoints",
            filename='step_{step}-loss_{val/loss:.2f}',
            save_top_k=3,
            monitor='val/loss',
            mode='min',
            every_n_train_steps=train_config.save_steps,
        ),
        LearningRateMonitor(logging_interval='step'),
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
    else:
        logger = True
    
    
    
    with open(exp_dir / "config.txt", "w") as f:
        f.write("Model Config:\n")
        f.write(str(model_config.to_dict()))
        f.write("\n\nTraining Config:\n")
        f.write(str(train_config.to_dict()))
    
    trainer = pl.Trainer(
        max_steps=train_config.total_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[train_config.gpu_id] if torch.cuda.is_available() else None,
        accumulate_grad_batches=train_config.gradient_accumulation_steps,
        gradient_clip_val=train_config.max_grad_norm,
        val_check_interval=train_config.eval_steps,  # steps 단위로 validation
        num_sanity_val_steps=2,  # sanity check 수 지정
        check_val_every_n_epoch=None,  # epoch 기반 validation 비활성화
        max_epochs=-1,  # infinite epochs
        limit_train_batches=train_config.total_steps,  # 전체 step 수 제한
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=train_config.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()