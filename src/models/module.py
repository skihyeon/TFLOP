import pytorch_lightning as pl
import torch
from typing import Dict, Any
from .tflop import TFLOP
from .losses import TFLOPLoss
from metrics.teds import compute_teds, compute_teds_struct

class TFLOPLightningModule(pl.LightningModule):
    def __init__(self, model_config: Any, train_config: Any):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.train_config = train_config
        
        # Model components
        self.model = TFLOP(model_config)
        self.criterion = TFLOPLoss(
            temperature=model_config.temperature,
            lambda_cls=model_config.lambda_cls,
            lambda_ptr=model_config.lambda_ptr,
            lambda_empty_ptr=model_config.lambda_empty_ptr,
            lambda_row_contr=model_config.lambda_row_contr,
            lambda_col_contr=model_config.lambda_col_contr
        )
        
    def forward(self, batch):
        # 키 이름 변환
        model_inputs = {
            'images': batch['images'],
            'text_regions': batch['bboxes'],  # bboxes -> text_regions
            'labels': batch.get('tokens', None),
            'attention_mask': batch.get('attention_mask', None),
            'row_spans': batch.get('row_spans', None),
            'col_spans': batch.get('col_spans', None)
        }
        return self.model(**model_inputs)
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(batch)
        batch_size = batch['images'].size(0)
        
        loss_dict = self.criterion(
            tag_logits=outputs['tag_logits'],
            tag_targets=batch['tokens'],
            box_features=outputs['box_features'],
            tag_features=outputs['tag_features'],
            box_indices=batch['tokens'],
            data_tag_mask=outputs['data_tag_mask'],
            empty_mask=(batch['tokens'] == self.model.tokenizer.pad_token_id),
            row_spans=batch['row_spans'],
            col_spans=batch['col_spans']
        )
        
        # step 단위로만 로깅
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, 
                    batch_size=batch_size,
                    on_step=True,
                    on_epoch=False,  # epoch 로깅 비활성화
                    prog_bar=(name == 'loss'))
        
        return loss_dict
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(batch)
        batch_size = batch['images'].size(0)
        
        loss_dict = self.criterion(
            tag_logits=outputs['tag_logits'],
            tag_targets=batch['tokens'],
            box_features=outputs['box_features'],
            tag_features=outputs['tag_features'],
            box_indices=batch['tokens'],
            data_tag_mask=outputs['data_tag_mask'],
            empty_mask=(batch['tokens'] == self.model.tokenizer.pad_token_id),
            row_spans=batch['row_spans'],
            col_spans=batch['col_spans']
        )
            
        # TEDS 계산
        pred_tokens = outputs['tag_logits'].argmax(dim=-1)
        pred_otsl = self.model.tokenizer.decode(pred_tokens[0].cpu().tolist())
        true_otsl = self.model.tokenizer.decode(batch['tokens'][0].cpu().tolist())
        
        try:
            pred_html = self.model.tokenizer.convert_otsl_to_html(pred_otsl)
        except ValueError:
            pred_html = "<table><tr><td></td></tr></table>"

        true_html = self.model.tokenizer.convert_otsl_to_html(true_otsl)
        
        teds = compute_teds(pred_html, true_html)
        teds_struct = compute_teds_struct(pred_html, true_html)
        
        # Visualization용 데이터 추가
        if batch_idx == 0:
            outputs.update({
                'pred_html': pred_html,
                'true_html': true_html,
                'pred_otsl': pred_otsl,
                'true_otsl': true_otsl
            })
        
        # 메트릭 로깅
        metrics = {
            'val/loss': loss_dict['loss'],
            'val/teds': teds,
            'val/teds_struct': teds_struct
        }
        
        self.log_dict(
            metrics, 
            batch_size=batch_size,
            on_step=True,   # step 레벨 메트릭 활성화
            on_epoch=False, # epoch 레벨 메트릭 비활성화
            prog_bar=True,
            sync_dist=True
        )
        
        return {**outputs, **metrics}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_config.total_steps,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        } 