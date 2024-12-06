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
        
        # Dataset의 max_boxes를 모델 설정에 추가
        if hasattr(train_config, 'dataset') and train_config.dataset is not None:
            self.model_config.max_boxes = train_config.dataset.max_boxes
        
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
        
        for name, value in loss_dict.items():
            # self.log(f"train/{name}", value, 
            #         batch_size=batch_size,
            #         on_step=True,
            #         on_epoch=False,  # epoch 로 비활성화
            #         prog_bar=(name == 'loss')
            #         )
            if name == 'loss': name = 'l'
            elif name == 'cls_loss': name = 'cls'
            elif name == 'ptr_loss': name = 'p'
            elif name == 'empty_ptr_loss': name = 'e'
            elif name == 'row_contr_loss': name = 'r'
            elif name == 'col_contr_loss': name = 'c'
            self.log(f"{name}", value,
                    batch_size=batch_size,
                    on_step=True,
                    on_epoch=False,  # epoch 로깅 비활성화
                    prog_bar=True
                    )
        
        return loss_dict
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        batch_size = batch['images'].size(0)
        loss_dict = self.criterion(batch, outputs)
        
        # pointer logits와 empty logits 계산
        box_features = outputs['box_proj']      # (B, N, D)
        tag_features = outputs['tag_proj']      # (B, T, D)
        empty_proj = outputs['empty_proj']      # (B, 1, D)
        
        # pointer logits 계산 (temperature=0.1 사용)
        pointer_logits = torch.matmul(box_features, tag_features.transpose(-2, -1)) / 0.1  # (B, N, T)
        empty_logits = torch.matmul(tag_features, empty_proj.transpose(-2, -1)).squeeze(-1)  # (B, T)
        
        # 시각화를 위해 outputs에 추가
        outputs['pointer_logits'] = pointer_logits
        outputs['empty_logits'] = empty_logits
        
        # 1. Loss 로깅
        for name, value in loss_dict.items():
            self.log(f"val/{name}", value, 
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(name == 'loss'),
                    sync_dist=True)
        
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