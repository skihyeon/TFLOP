import pytorch_lightning as pl
import torch
from typing import Dict, Any
from .tflop import TFLOP
from .losses import TFLOPLoss
from metrics.teds import compute_teds, compute_teds_struct
from utils.util import construct_table_html_pred, construct_table_html_gt
import torchmetrics

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
        
    def forward(self, batch):
        return self.model(batch)
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(batch)
        loss_dict = self.criterion(batch, outputs)
        
        # Loss logging (epoch 단위)
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value,
                    batch_size=batch['images'].size(0),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True)
        
        return loss_dict
    
    def validation_step(self, batch, batch_idx):
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
        
        # 첫 번째 배치에 대해서만 시각화 정보 준비
        visualization_outputs = {}
        if batch_idx == 0:
            try:
                # 첫 번째 샘플에 대한 예측/정답 준비
                pred_tokens = outputs['tag_logits'][0].argmax(dim=-1)
                true_tokens = batch['token_ids'][0]
                
                # 토큰 분포 계산
                token_dist = torch.bincount(pred_tokens.view(-1), 
                                        minlength=self.model.tokenizer.vocab_size)
                true_dist = torch.bincount(true_tokens.view(-1),
                                        minlength=self.model.tokenizer.vocab_size)
                
                # 토큰 분포 출력
                print("\n=== Token Distribution Debug ===")
                print("Token Distributions (Pred | True):")
                for token_id in range(self.model.tokenizer.vocab_size):
                    pred_count = token_dist[token_id].item()
                    true_count = true_dist[token_id].item()
                    if pred_count > 0 or true_count > 0:
                        token = self.model.tokenizer.id2token[token_id]
                        print(f"  {token:8s}: {pred_count:4d} | {true_count:4d}")

                pred_otsl = self.model.tokenizer.decode(pred_tokens.cpu().tolist())
                
                # 필요한 정보들이 모두 있는지 확인
                if all(k in batch for k in ['cells', 'html', 'otsl']):
                    text_regions = batch['cells'][0]
                    
                    # HTML 생성
                    pred_html = construct_table_html_pred(
                        pred_otsl,
                        text_regions,
                        outputs['pointer_logits'][0]
                    )
                    true_html = construct_table_html_gt(batch['html'][0])
                    
                    # TEDS 계산
                    teds = compute_teds(pred_html, true_html)
                    teds_struct = compute_teds_struct(pred_html, true_html)
                    
                    # 시각화에 필요한 정보들 저장
                    visualization_outputs.update({
                        'pred_html': pred_html,
                        'true_html': true_html,
                        'pred_otsl': pred_otsl,
                        'true_otsl': batch['otsl'][0],
                        'pointer_logits': outputs['pointer_logits'][0],
                        'empty_pointer_logits': outputs['empty_pointer_logits'][0],
                        'teds': teds,
                        'teds_s': teds_struct
                    })
            
            except Exception as e:
                print(f"Error constructing pred HTML: {str(e)}")
        
        # TEDS 메트릭 업데이트 (전체 배치)
        for i in range(batch_size):
            try:
                pred_tokens = outputs['tag_logits'][i].argmax(dim=-1)
                pred_otsl = self.model.tokenizer.decode(pred_tokens.cpu().tolist())
                text_regions = batch['cells'][i]
                pred_html = construct_table_html_pred(
                    pred_otsl,
                    text_regions,
                    outputs['pointer_logits'][i]
                )
                true_html = construct_table_html_gt(batch['html'][i])
                
                teds = compute_teds(pred_html, true_html)
                teds_struct = compute_teds_struct(pred_html, true_html)
                
                self.val_teds.update(teds)
                self.val_teds_struct.update(teds_struct)
                
            except Exception as e:
                print(f"Warning: TEDS calculation failed for sample {i}: {str(e)}")
                self.val_teds.update(0.0)
                self.val_teds_struct.update(0.0)
        
        return {**loss_dict, **visualization_outputs}

    def on_validation_epoch_end(self):
        # Log metrics
        self.log("val/teds", self.val_teds.compute(), prog_bar=True)
        self.log("val/teds_struct", self.val_teds_struct.compute(), prog_bar=True)
        
        # Reset metrics
        self.val_teds.reset()
        self.val_teds_struct.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_config.num_epochs,  # total_steps 대신 num_epochs 사용
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"  # step 대신 epoch 단위로 변경
            }
        }
    
    def valid_step_progressor(self, batch, outputs, batch_idx):
        batch_size = batch['images'].size(0)
        
        # 2. TEDS 메트릭 계산 및 로깅
        try:
            # 첫 번째 배치의 모든 샘플에 대해 토큰 분포 디버깅
            if batch_idx == 0:
                batch_results = []
                
                # 예측된 토큰 분포 (한 번에 계산)
                pred_tokens = outputs['tag_logits'][0].argmax(dim=-1)
                token_dist = torch.bincount(pred_tokens.view(-1), 
                                            minlength=self.model.tokenizer.vocab_size)
                
                # Ground Truth 토큰 분포 (한 번에 계산)
                true_tokens = batch['token_ids'][0]
                true_dist = torch.bincount(true_tokens.view(-1),
                                        minlength=self.model.tokenizer.vocab_size)
                
                for i in range(batch_size):
                    with torch.no_grad():
                        # 예측된 토큰 분포
                        pred_tokens = outputs['tag_logits'][i].argmax(dim=-1)
                        token_dist = torch.bincount(pred_tokens.view(-1), 
                                                minlength=self.model.tokenizer.vocab_size)
                        
                        # Ground Truth 토큰 분포
                        true_tokens = batch['token_ids'][i]
                        true_dist = torch.bincount(true_tokens.view(-1),
                                                minlength=self.model.tokenizer.vocab_size)
                        
                        print("\n=== Token Distribution Debug ===")
                        print("Token Distributions (Pred | True):")
                        for token_id in range(self.model.tokenizer.vocab_size):
                            pred_count = token_dist[token_id].item()
                            true_count = true_dist[token_id].item()
                            if pred_count > 0 or true_count > 0:
                                token = self.model.tokenizer.id2token[token_id]
                                print(f"  {token:8s}: {pred_count:4d} | {true_count:4d}")
                                
                        # OTSL 시퀀스 디코딩
                        pred_otsl = self.model.tokenizer.decode(
                            pred_tokens.cpu().tolist()
                        )
                        true_otsl = self.model.tokenizer.decode(
                            true_tokens.cpu().tolist()
                        )
                        
                        # HTML 생성
                        text_regions = batch['cells'][i]
                        try:
                            pred_html = construct_table_html_pred(
                                pred_otsl,
                                text_regions,
                                outputs['pointer_logits'][i],  # 새로 계산된 pointer_logits 사용
                                # outputs['empty_logits'][i]     # 새로 계산된 empty_logits 사용
                            )
                            true_html = construct_table_html_gt(
                                batch['html'][i]
                            )
                        except ValueError as e:
                            print(f"Warning: Failed to construct HTML for sample {i}: {str(e)}")
                            pred_html = "<table><tr><td>Invalid OTSL sequence</td></tr></table>"
                            true_html = "<table><tr><td>Invalid OTSL sequence</td></tr></table>"
                        
                        # TEDS 계산
                        teds = compute_teds(pred_html, true_html)
                        teds_struct = compute_teds_struct(pred_html, true_html)
                        
                        # 결과 저장
                        outputs.update({
                            'sample_idx': i,
                            'pred_otsl': pred_otsl,
                            'true_otsl': true_otsl,
                            'pred_html': pred_html,
                            'true_html': true_html,
                            'teds': teds,
                            'teds_s': teds_struct,
                            'token_dist': token_dist,
                            'true_dist': true_dist
                        })
                
                # 전체 배치 결과 출력
                print("\n=== Validation Batch 0 Results ===")
                for result in batch_results:
                    print(f"\nSample {result['sample_idx']}:")
                    print(f"TEDS: {result['teds']:.4f}, TEDS-S: {result['teds_s']:.4f}")
                    print("\nToken Distribution (Pred | True):")
                    for token_id in range(self.model.tokenizer.vocab_size):
                        pred_count = result['token_dist'][token_id].item()
                        true_count = result['true_dist'][token_id].item()
                        if pred_count > 0 or true_count > 0:
                            token = self.model.tokenizer.id2token[token_id]
                            print(f"  {token:8s}: {pred_count:4d} | {true_count:4d}")
                
                # 첫 번째 배치의 모든 결과를 outputs에 추가
                outputs.update({
                    'batch_results': batch_results
                })
        except Exception as e:
            print(f"Warning: Validation step failed: {str(e)}")
            outputs.update({
                'pred_otsl': "Invalid sequence",
                'true_otsl': "Invalid sequence",
                'pred_html': "<table><tr><td>Error</td></tr></table>",
                'true_html': "<table><tr><td>Error</td></tr></table>",
                'complete_html': "<table><tr><td>Error</td></tr></table>",
                'teds': 0.0,
                'teds_s': 0.0
            })
            self.log('val_teds', 0.0, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True)
            self.log('val_teds_s', 0.0, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True)
        
        return outputs
