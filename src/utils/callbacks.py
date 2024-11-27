from pytorch_lightning.callbacks import Callback
from .visualize import visualize_validation_sample
import pytorch_lightning as pl
from typing import Dict
import torch
from pathlib import Path

class ValidationVisualizationCallback(Callback):
    def __init__(self, viz_dir: str):
        super().__init__()
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
                
    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        if batch_idx == 0:  # 첫 번째 배치만 시각화
            # val/ 접두사 제거한 loss components 준비
            loss_components = {
                k.replace('val/', ''): v.item() if torch.is_tensor(v) else v 
                for k, v in outputs.items() 
                if k in ['val/loss', 'val/cls_loss', 'val/ptr_loss', 'val/empty_ptr_loss']
            }
            
            visualize_validation_sample(
                image=batch['images'][0],
                boxes=batch['bboxes'][0],
                pred_html=outputs['pred_html'],
                true_html=outputs['true_html'],
                pred_otsl=outputs['pred_otsl'],
                true_otsl=outputs['true_otsl'],
                pointer_logits=outputs.get('pointer_logits', None),
                step=trainer.global_step,
                loss_components=loss_components,
                viz_dir=self.viz_dir
            )