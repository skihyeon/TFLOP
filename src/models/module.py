import pytorch_lightning as pl
import torch
from typing import Dict, Any
from .tflop import TFLOP
from .losses import TFLOPLoss
from metrics.teds import compute_teds, compute_teds_struct
from utils.util import construct_table_html_pred, construct_table_html_gt
import torchmetrics
import math
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TFLOPLightningModule(pl.LightningModule):
    def __init__(self, model_config: Any, train_config: Any, inference_mode: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.train_config = train_config
        self.inference_mode = inference_mode

        # Model components
        self.model = TFLOP(model_config, inference_mode)
        self.criterion = TFLOPLoss(
            lambda_cls=model_config.lambda_cls,
            lambda_ptr=model_config.lambda_ptr,
            lambda_empty_ptr=model_config.lambda_empty_ptr,
            lambda_row_contr=model_config.lambda_row_contr,
            lambda_col_contr=model_config.lambda_col_contr,
            tokenizer=self.model.tokenizer
        ) 
                # Metric 초기화
        self.val_teds = torchmetrics.MeanMetric()
        self.val_teds_struct = torchmetrics.MeanMetric()
        
        self.teds_eval_interval = 10
        
    def forward(self, batch):
        return self.model(batch)
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(batch)
        loss_dict = self.criterion(batch, outputs)
        # Loss logging (step과 epoch 단위로 분리)
        for name, value in loss_dict.items():
            # Epoch 단위 로깅
            self.log(f"train/{name.replace('_loss', '')}", value,
                    batch_size=batch['images'].size(0),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True)
        
        
        return loss_dict
        
    def validation_step(self, batch, batch_idx):
        # Validation에서는 teacher forcing 사용하지 않음
        batch['teacher_forcing_ratio'] = 0.0
        
        outputs = self(batch)
        loss_dict = self.criterion(batch, outputs)
        batch_size = batch['images'].size(0)
        
        # Loss logging
        for name, value in loss_dict.items():
            self.log(f"val/{name}", value, 
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(name == 'loss'),
                    sync_dist=True)
        
        visualization_outputs = {}
        
        # 첫 번째 배치의 첫 번째 샘플에 대해서만 처리
        try:
            # 배치의 첫 번째 샘플에 대해서만 처리
            for i in range(batch_size):
                # 예측 토큰과 실제 토큰 디코딩
                pred_tokens = outputs['tag_logits'][i].argmax(dim=-1)
                true_tokens = batch['token_ids'][i]
                
                # 패딩된 부분 제거
                pred_tokens = pred_tokens[pred_tokens != self.model.tokenizer.pad_token_id]
                true_tokens = true_tokens[true_tokens != self.model.tokenizer.pad_token_id]
                
                # OTSL 문자열로 변환
                pred_otsl = self.model.tokenizer.decode(pred_tokens.cpu().tolist())
                true_otsl = self.model.tokenizer.decode(true_tokens.cpu().tolist())
                
                # 토큰 분포 로깅
                log_text = self._print_token_distribution(pred_tokens, true_tokens)

                # HTML 생성
                bbox_with_text = batch['bbox_with_text'][i]
                try:
                    pred_html = construct_table_html_pred(
                        pred_otsl,
                        bbox_with_text,
                        outputs['pointer_logits'][i]
                    )
                    true_html = construct_table_html_gt(batch['html'][i])
                    
                    # TEDS 계산
                    teds = compute_teds(pred_html, true_html)
                    teds_struct = compute_teds_struct(pred_html, true_html)
                    print(f"pred_otsl: {pred_otsl}")
                    print(f"true_otsl: {true_otsl}")
                    print(f"teds: {teds}, teds_struct: {teds_struct}")
                    self.val_teds.update(teds)
                    self.val_teds_struct.update(teds_struct)
                    
                    self.log("val/teds", teds, 
                            on_epoch=True, 
                            prog_bar=True, 
                            batch_size=batch_size,
                            sync_dist=True)
                    self.log("val/teds_struct", teds_struct, 
                            on_epoch=True, 
                            prog_bar=True, 
                            batch_size=batch_size,
                            sync_dist=True)
                    
                except Exception as e:
                    teds = 0.0
                    teds_struct = 0.0
                    pred_html = ""
                    true_html = ""
                
                # 시각화용 출력 저장
                visualization_outputs = {
                    'image_name': batch['image_names'][i],
                    'pred_html': pred_html,
                    'true_html': true_html,
                    'pred_otsl': pred_otsl,
                    'true_otsl': true_otsl,
                    'pointer_logits': outputs['pointer_logits'][i],
                    'empty_pointer_logits': outputs['empty_pointer_logits'][i],
                    'teds': teds,
                    'teds_s': teds_struct,
                    'log_text': log_text
                }
        
        except Exception as e:
            print(f"검증 단계 처리 중 오류 발생: {str(e)}")
            self.val_teds.update(0.0)
            self.val_teds_struct.update(0.0)
        
        return {**loss_dict, **visualization_outputs}


    def _print_token_distribution(self, pred_tokens, true_tokens):
        """토큰 분포 출력"""
        log_text = ""
        token_dist = torch.bincount(pred_tokens.view(-1), 
                                minlength=self.model.tokenizer.vocab_size)
        true_dist = torch.bincount(true_tokens.view(-1),
                                minlength=self.model.tokenizer.vocab_size)
        
        log_text += "\n=== Token Distribution Debug ===\n"
        log_text += "Token Distributions (Pred | True):\n"
        for token_id in range(self.model.tokenizer.vocab_size):
            pred_count = token_dist[token_id].item()
            true_count = true_dist[token_id].item()
            if pred_count > 0 or true_count > 0:
                token = self.model.tokenizer.id2token[token_id]
                log_text += f"  {token:8s}: {pred_count:4d} | {true_count:4d}\n"
        # print(log_text)
        return log_text
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # lr을 절반으로 감소
            patience=5,   # 5 에폭동안 개선이 없으면 lr 감소
            verbose=True,
            min_lr=1e-6  # 최소 lr
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",  # validation loss 기준
                "interval": "epoch",    # epoch 단위로 업데이트
                "frequency": 1
            }
        }
