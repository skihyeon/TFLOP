from typing import Any, Dict

import torch
import pytorch_lightning as pl
import torchmetrics
from torch.optim import lr_scheduler

from .tflop import TFLOP
from .losses import TFLOPLoss
from metrics.teds import compute_teds, compute_teds_struct, compute_teds_both
from utils.util import construct_table_html_pred, construct_table_html_gt
from metrics.teds import teds_calculator

class TFLOPLightningModule(pl.LightningModule):
    def __init__(self, model_config: Any, train_config: Any):
        super().__init__()
        self.save_hyperparameters({
            "config": model_config.__dict__,
            "train_config": train_config.__dict__
        })
        self.model_config = model_config
        self.train_config = train_config
        
        # 메모리 관리 설정
        self.automatic_optimization = True
        torch.backends.cudnn.benchmark = True
        
        # Model components
        self.model = TFLOP(model_config)
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
        # 메모리 최적화를 위한 context manager 사용
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self(batch)
            loss_dict = self.criterion(batch, outputs)
        
        # 스텝 단위 로깅 추가
        self.log('step', self.global_step, prog_bar=False)
        for name, value in loss_dict.items():
            self.log(f"train/step_{name.replace('_loss', '')}", value,
                    on_step=True, on_epoch=False, prog_bar=True)
        
        # 메모리 정리
        del outputs
        torch.cuda.empty_cache()
        
        batch_size = batch['images'].size(0)
        
        # Loss logging
        for name, value in loss_dict.items():
            # Step 단위 로깅 (train_step/...)
            # self.log(f"train_step/{name.replace('_loss', '')}", value,
            #         batch_size=batch_size,
            #         on_step=True,
            #         on_epoch=False,
            #         prog_bar=False,
            #         sync_dist=True)
            
            # Epoch 단위 로깅 (train/...)
            self.log(f"train/{name.replace('_loss', '')}", value,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True)
        
        # CUDA 동기화 추가
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        return loss_dict
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if batch_idx % 1000 == 0:  # 메모리 주기적 정리
            torch.cuda.empty_cache()
            
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self(batch)
            loss_dict = self.criterion(batch, outputs)
            
        # 배치당 하나의 샘플만 상세 평가 (첫 번째 샘플)
        visualization_outputs = {}
        try:
            # 전체 배치에 대한 TEDS 계산 (병렬)
            pred_tokens_batch = outputs['tag_logits'].argmax(dim=-1)
            true_tokens_batch = batch['token_ids']
            
            # 패딩 마스크 생성 (배치 전체)
            pad_mask = pred_tokens_batch != self.model.tokenizer.pad_token_id
            
            # 배치의 첫 번째 샘플에 대해서만 상세 로깅
            i = 0
            pred_tokens = pred_tokens_batch[i][pad_mask[i]]
            true_tokens = true_tokens_batch[i][true_tokens_batch[i] != self.model.tokenizer.pad_token_id]
            
            # OTSL 문자열로 변환
            pred_otsl = self.model.tokenizer.decode(pred_tokens.cpu().tolist())
            true_otsl = self.model.tokenizer.decode(true_tokens.cpu().tolist())
            
            # 토큰 분포 로깅 (첫 번째 샘플만)
            log_text = self._print_token_distribution(pred_tokens, true_tokens)

            # HTML 생성 (첫 번째 샘플만)
            bbox_with_text = batch['bbox_with_text'][i]
            try:
                pred_html = construct_table_html_pred(
                    pred_otsl,
                    bbox_with_text,
                    outputs['pointer_logits'][i] / self.model.config.temperature
                )
                true_html = construct_table_html_gt(batch['html'][i])
                
                # 한 번의 호출로 두 메트릭 모두 계산
                teds, teds_struct = compute_teds_both(pred_html, true_html)
                
                # 메트릭 업데이트
                self.val_teds.update(teds)
                self.val_teds_struct.update(teds_struct)
                
                self.log("val/teds", teds, 
                        on_epoch=True, 
                        prog_bar=True, 
                        batch_size=batch['images'].size(0),
                        sync_dist=True)
                self.log("val/teds_struct", teds_struct, 
                        on_epoch=True, 
                        prog_bar=True, 
                        batch_size=batch['images'].size(0),
                        sync_dist=True)
                
            except Exception as e:
                teds = 0.0
                teds_struct = 0.0
                pred_html = ""
                true_html = ""
            
            # 시각화용 출력 저장 (첫 번째 샘플만)
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
            
            # 중간 결과물 즉시 제거
            del pred_tokens, pred_tokens_batch
            
        except Exception as e:
            print(f"검증 단계 처리 중 오류 발생: {str(e)}")
            
        # 큰 출력값들 제거
        del outputs
        torch.cuda.empty_cache()
        
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
        
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=1000,  # 초기 주기 (스텝 단위)
            T_mult=1, 
            eta_min=1e-10
        )
        
        # 옵티마이저 상태 메모리 최적화
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # epoch에서 step으로 변경
                "frequency": 1
            }
        }
        
    def test_step(self, batch, batch_idx):
        self.model.train(False)
        outputs = self(batch)
        loss_dict = self.criterion(batch, outputs)
        batch_size = batch['images'].size(0)
        
        # 배치 단위로 TEDS 계산
        pred_htmls = []
        true_htmls = []
        
        for i in range(batch_size):
            pred_tokens = outputs['tag_logits'][i].argmax(dim=-1)
            pred_tokens = pred_tokens[pred_tokens != self.model.tokenizer.pad_token_id]
            pred_otsl = self.model.tokenizer.decode(pred_tokens.cpu().tolist())
            
            try:
                pred_html = construct_table_html_pred(
                    pred_otsl,
                    batch['bbox_with_text'][i],
                    outputs['pointer_logits'][i] / self.model.config.temperature
                )
                true_html = construct_table_html_gt(batch['html'][i])
                
                pred_htmls.append(pred_html)
                true_htmls.append(true_html)
                
            except Exception as e:
                print(f"Error in test_step: {str(e)}")
                pred_htmls.append("")
                true_htmls.append("")
        
        # 배치 단위로 TEDS와 TEDS-struct 동시 계산
        teds_scores = []
        teds_struct_scores = []
        
        for pred_html, true_html in zip(pred_htmls, true_htmls):
            teds, teds_struct = compute_teds_both(pred_html, true_html)
            teds_scores.append(teds)
            teds_struct_scores.append(teds_struct)
        
        # 평균 TEDS 점수 로깅
        self.log("test/teds", sum(teds_scores) / len(teds_scores), 
                on_epoch=True, 
                batch_size=batch_size,
                sync_dist=True)
        self.log("test/teds_struct", sum(teds_struct_scores) / len(teds_struct_scores), 
                on_epoch=True, 
                batch_size=batch_size,
                sync_dist=True)
        
        return loss_dict
    
    def predict_step(self, batch, batch_idx):
        self.model.train(False)
        outputs = self(batch)
        batch_size = batch['images'].size(0)
        predictions = []

        for i in range(batch_size):
            # 1. OTSL 시퀀스 생성
            pred_tokens = outputs['tag_logits'][i].argmax(dim=-1)
        
            pred_otsl = self.model.tokenizer.decode(pred_tokens.cpu().tolist()) ## 어차피 decode 시 특수 토큰들은 제거됨
            
            # 2. HTML 생성
            try:
                pointer_logits = outputs['pointer_logits'][i] / self.model.config.temperature
                pred_html = construct_table_html_pred(
                    pred_otsl,
                    batch['bbox_with_text'][i],
                    pointer_logits,
                    confidence_threshold=0.2
                )
                print(f"pred_otsl: {pred_otsl}")
        
            except Exception as e:
                print(f"Error constructing HTML for {batch['image_names'][i]}: {str(e)}")
                pred_html = "<table><tr><td>Error occurred</td></tr></table>"
            
            predictions.append({
                'image_name': batch['image_names'][i],
                'pred_html': pred_html,
                'pred_otsl': pred_otsl,
                'pointer_logits': pointer_logits.cpu().numpy() if pointer_logits is not None else None
            })
        
        return predictions

    def on_train_epoch_end(self):
        # 에폭 종료시 메모리 정리
        torch.cuda.empty_cache()
        
    def on_validation_epoch_end(self):
        # 검증 종료시 메모리 정리
        torch.cuda.empty_cache()