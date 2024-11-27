import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .tflop import TFLOP
from .losses import TFLOPLoss
from metrics.teds import compute_teds, compute_teds_struct
import wandb
from pathlib import Path

class TFLOPModule(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
        # Model components
        self.model = TFLOP(config)
        self.criterion = TFLOPLoss(
            temperature=config.temperature,
            lambda_cls=config.lambda_cls,
            lambda_ptr=config.lambda_ptr,
            lambda_empty_ptr=config.lambda_empty_ptr,
            lambda_row_contr=config.lambda_row_contr,
            lambda_col_contr=config.lambda_col_contr
        )
        
        # Optimizer
        self.configure_optimizers()
        
        # Metrics
        self.best_val_loss = float('inf')
        
    def configure_optimizers(self):
        """Optimizer 및 Scheduler 설정"""
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.total_steps,
            eta_min=1e-6
        )
        
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """단일 학습 스텝"""
        outputs = self.model(
            images=batch['images'],
            text_regions=batch['bboxes'],
            labels=batch['tokens'],
            attention_mask=batch['attention_mask'],
            row_spans=batch['row_spans'],
            col_spans=batch['col_spans']
        )
        
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
        
        return {'loss': loss_dict['loss'], **loss_dict}
    
    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """단일 검증 스텝"""
        outputs = self.model(
            images=batch['images'],
            text_regions=batch['bboxes'],
            labels=batch['tokens'],
            attention_mask=batch['attention_mask']
        )
        
        # Loss 계산
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
        
        pred_html = self.model.tokenizer.convert_otsl_to_html(pred_otsl)
        true_html = self.model.tokenizer.convert_otsl_to_html(true_otsl)
        
        teds = compute_teds(pred_html, true_html)
        teds_struct = compute_teds_struct(pred_html, true_html)
        
        return {
            'val_loss': loss_dict['loss'],
            'teds': teds,
            'teds_struct': teds_struct,
            **loss_dict
        }
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = str(Path(path).parent / 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """메트릭 로깅"""
        if self.config.use_wandb:
            wandb.log(metrics, step=step) 