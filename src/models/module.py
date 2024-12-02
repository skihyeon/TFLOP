import pytorch_lightning as pl
import torch
from typing import Dict, Any
from .tflop import TFLOP
from .losses import TFLOPLoss
from metrics.teds import compute_teds, compute_teds_struct
from utils.util import construct_table_html_pred, construct_table_html_gt

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
        
    def forward(self, batch):
        # 키 이름 변환
        # model_inputs = {
        #     'images': batch['images'],
        #     'text_regions': batch['bboxes'],  # bboxes -> text_regions
        #     'labels': batch.get('tokens', None),
        #     'attention_mask': batch.get('attention_mask', None),
        #     'row_span_coef': batch.get('row_span_coef', None),
        #     'col_span_coef': batch.get('col_span_coef', None),
        #     'data_tag_mask': batch.get('data_tag_mask', None),
        #     'box_indices': batch.get('box_indices', None),
        #     'cells': batch.get('cells', None),
        #     'html': batch.get('html', None)
        # }
        return self.model(batch)
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(batch)
        batch_size = batch['images'].size(0)
        loss_dict = self.criterion(batch, outputs)
        
        # step 단위로만 로깅
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, 
                    batch_size=batch_size,
                    on_step=True,
                    on_epoch=False,  # epoch 로�� 비활성화
                    prog_bar=(name == 'loss')
                    )
        
        return loss_dict
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        batch_size = batch['images'].size(0)
        loss_dict = self.criterion(batch, outputs)  
        
        # 1. Loss 로깅 - 메트릭 이름에서 _step 제거
        for name, value in loss_dict.items():
            self.log(f"val/{name}", value, 
                    batch_size=batch_size,
                    on_step=False,     # 스텝별 로깅 비활성화
                    on_epoch=True,     # 에폭 평균 계산
                    prog_bar=(name == 'loss'),
                    sync_dist=True)
        
        # 2. TEDS 메트릭 계산 및 로깅
        try:
            pred_otsl = self.model.tokenizer.decode(
                outputs['tag_logits'][0].argmax(dim=-1).cpu().tolist()
            )
            true_otsl = self.model.tokenizer.decode(
                batch['tokens'][0].cpu().tolist()
            )
            
            # text regions 정보 준비 (첫 번째 샘플의 cells 정보)
            text_regions = batch['cells'][0]
            
            try:
                # 예측 HTML 생성
                pred_html = construct_table_html_pred(
                    pred_otsl,
                    text_regions,
                    outputs['pointer_logits'][0]
                )
                
                # Ground Truth HTML 생성 (text 정보 포함)
                true_html = construct_table_html_gt(
                    batch['html'][0]
                )
            except ValueError as e:
                print(f"Warning: Failed to construct HTML: {str(e)}")
                pred_html = "<table><tr><td>Invalid OTSL sequence</td></tr></table>"
                true_html = "<table><tr><td>Invalid OTSL sequence</td></tr></table>"
            
            # TEDS 및 TEDS-Structure 계산
            teds = compute_teds(pred_html, true_html)
            teds_struct = compute_teds_struct(pred_html, true_html)
            
            # TEDS 메트릭 로깅 - _step 없이 로깅
            self.log('val/teds', teds,
                    batch_size=batch_size,
                    on_step=False,     # 스텝별 로깅 비활성화
                    on_epoch=True,     # 에폭 평균 계산
                    prog_bar=True,
                    sync_dist=True)
            self.log('val/teds_s', teds_struct,
                    batch_size=batch_size,
                    on_step=False,     # 스텝별 로깅 비활성화
                    on_epoch=True,     # 에폭 평균 계산
                    prog_bar=True,
                    sync_dist=True)
            
            # 3. 첫 번째 배치의 첫 번째 샘플에 대해서만 시각화 데이터 준비
            if batch_idx == 0:
                # 시각화용 출력 준비
                outputs.update({
                    'pred_otsl': pred_otsl,
                    'true_otsl': true_otsl,
                    'pred_html': pred_html,
                    'true_html': true_html,
                    'complete_html': pred_html,  # complete_html은 이제 pred_html과 동일
                    'teds': teds,
                    'teds_s': teds_struct
                })
                
        except Exception as e:
            print(f"Warning: Validation step failed: {str(e)}")
            # 기본값으로 출력 구성
            outputs.update({
                'pred_otsl': "Invalid sequence",
                'true_otsl': "Invalid sequence",
                'pred_html': "<table><tr><td>Error</td></tr></table>",
                'true_html': "<table><tr><td>Error</td></tr></table>",
                'complete_html': "<table><tr><td>Error</td></tr></table>",
                'teds': 0.0,
                'teds_s': 0.0
            })
            # 에러 시에도 동일한 형식으로 로깅
            self.log('val_teds', 0.0, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True)
            self.log('val_teds_s', 0.0, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True)
        
        return outputs

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