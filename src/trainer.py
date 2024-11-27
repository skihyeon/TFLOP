import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Tuple
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from datetime import datetime

from models import TFLOP, TFLOPLoss
from config import TrainingConfig, TFLOPConfig
from models.otsl_tokenizer import OTSLTokenizer
from utils.html_to_otsl import HTMLtoOTSLConverter
from metrics.teds import compute_teds, compute_teds_struct
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from utils.visualize import visualize_validation_sample

class TFLOPTrainer:
    def __init__(
        self,
        model_config: TFLOPConfig,
        train_config: TrainingConfig,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.device = train_config.device
        
        # 모델 초기화
        self.model = TFLOP(config=model_config).to(self.device)
        
        # Tokenizer 초기화
        self.tokenizer = OTSLTokenizer(vocab_size=model_config.vocab_size)
        
        # Loss 함수 초기화
        self.criterion = TFLOPLoss(
            temperature=model_config.temperature,
            lambda_cls=model_config.lambda_cls,
            lambda_ptr=model_config.lambda_ptr,
            lambda_empty_ptr=model_config.lambda_empty_ptr,
            lambda_row_contr=model_config.lambda_row_contr,
            lambda_col_contr=model_config.lambda_col_contr
        ).to(self.device)
        
        # Optimizer 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Scheduler를 CosineAnnealingLR로 변경
        self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=train_config.warmup_steps,
            num_training_steps=train_config.total_steps
        )
        
        # 체크포인트 디렉토리 생성 및 하위 폴더 설정
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(train_config.checkpoint_dir, current_time)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 모델 체크포인트 디렉토리
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'models')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 시각화 결과 저장 디렉토리
        self.viz_dir = os.path.join(self.experiment_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.config_save_dir = os.path.join(self.experiment_dir, 'configs')
        os.makedirs(self.config_save_dir, exist_ok=True)
        for config in [self.model_config, self.train_config]:
            with open(os.path.join(self.config_save_dir, 'config.txt'), 'a') as f:
                f.write(f'{config.__class__.__name__}\n')
                for k, v in config.__dict__.items():
                    f.write(f'{k}: {v}\n')
                f.write('\n')
                
        # Step 카운터 추가
        self.current_step = 0
        self.gradient_accumulation_steps = train_config.gradient_accumulation_steps
        
        # HTML to OTSL 컨버터 추가
        self.html_converter = HTMLtoOTSLConverter()
        
        
    def train(self, train_dataloader, val_dataloader):
        best_val_loss = float('inf')
        self.current_step = 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(total=self.train_config.total_steps, desc="Training")
        
        while self.current_step < self.train_config.total_steps:
            for batch in train_dataloader:
                self.model.train()
                
                # Forward pass
                outputs = self.model(
                    images=batch['images'].to(self.device),
                    text_regions=batch['bboxes'].to(self.device),
                    labels=batch['tokens'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    row_spans=batch['row_spans'].to(self.device),
                    col_spans=batch['col_spans'].to(self.device)
                )
                
                # Loss 계산
                loss_dict = self.criterion(
                    tag_logits=outputs['tag_logits'],
                    tag_targets=batch['tokens'].to(self.device),
                    box_features=outputs['box_features'],
                    tag_features=outputs['tag_features'],
                    box_indices=batch['tokens'].to(self.device),
                    data_tag_mask=outputs['data_tag_mask'],
                    empty_mask=(batch['tokens'] == self.tokenizer.pad_token_id).to(self.device),
                    row_spans=batch['row_spans'].to(self.device),
                    col_spans=batch['col_spans'].to(self.device)
                )
                
                # Gradient accumulation
                loss = loss_dict['loss'] / self.gradient_accumulation_steps
                loss.backward()
                
                # Logging
                current_lr = self.scheduler.get_last_lr()[0]
                
                if self.current_step > 0 and self.current_step % self.train_config.save_steps == 0:
                    self.save_checkpoint(loss.item(), is_best=False)
                
                # Validation
                if self.current_step > 0 and self.current_step % self.train_config.eval_steps == 0:
                    training = self.model.training
                    val_loss, val_components = self.validate(val_dataloader)
                    if training:
                        self.model.train()
                    
                    # Best model 체크 및 저장
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        self.logger.info(f"New best model with val_loss: {val_loss:.4f}")
                        self.save_checkpoint(val_loss, is_best=True)
                
                # Optimizer step (gradient accumulation 완료 시에만)
                if (self.current_step + 1) % self.gradient_accumulation_steps == 0:
                    if self.train_config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.train_config.max_grad_norm
                        )
                    self.optimizer.step()
                    
                    self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    
                    # wandb 로깅
                    if self.train_config.use_wandb:
                        wandb.log({
                            'train/loss': loss.item() * self.gradient_accumulation_steps,
                            'train/cls_loss': loss_dict['cls_loss'].item(),
                            'train/ptr_loss': loss_dict['ptr_loss'].item(),
                            'train/empty_ptr_loss': loss_dict['empty_ptr_loss'].item(),
                            'train/row_contr_loss': loss_dict['row_contr_loss'].item(),
                            'train/col_contr_loss': loss_dict['col_contr_loss'].item(),
                            'train/learning_rate': current_lr
                        }, step=self.current_step)
                
                # Progress bar 업데이트
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    # 'cls': f'{loss_dict["cls_loss"].item():.4f}',
                    # 'ptr': f'{loss_dict["ptr_loss"].item():.4f}',
                    # 'empty_ptr': f'{loss_dict["empty_ptr_loss"].item():.4f}',
                    # 'row_contr': f'{loss_dict["row_contr_loss"].item():.4f}',
                    # 'col_contr': f'{loss_dict["col_contr_loss"].item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })
                
                self.current_step += 1
                pbar.update(1)
                
                if self.current_step >= self.train_config.total_steps:
                    break
        
        pbar.close()
        if self.train_config.use_wandb:
            wandb.finish()
        self.logger.info("Training completed!")
        
    @torch.no_grad()
    def validate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        """검증 (loss 계산만 수행)"""
        self.model.eval()
        total_loss = 0
        loss_components = {
            'cls': 0, 'ptr': 0, 'empty_ptr': 0, 
            'row_contr': 0, 'col_contr': 0
        }
        
        first_batch = True
        pbar = tqdm(dataloader, desc=f'Validation step {self.current_step}')
        for batch_idx, batch in enumerate(pbar):
            # 1. Forward pass
            outputs = self.model(
                images=batch['images'].to(self.device),
                text_regions=batch['bboxes'].to(self.device),
                labels=batch['tokens'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
            )
            
            # 2. Loss 계산
            loss_dict = self.criterion(
                tag_logits=outputs['tag_logits'],
                tag_targets=batch['tokens'].to(self.device),
                box_features=outputs['box_features'],
                tag_features=outputs['tag_features'],
                box_indices=batch['tokens'].to(self.device),
                data_tag_mask=outputs['data_tag_mask'],
                empty_mask=(batch['tokens'] == self.tokenizer.pad_token_id).to(self.device),
                row_spans=batch['row_spans'].to(self.device),
                col_spans=batch['col_spans'].to(self.device)
            )
            
            loss = loss_dict['loss']
            total_loss += loss.item()
            
            # Loss components 기록
            for k in loss_components.keys():
                if f'{k}_loss' in loss_dict:
                    loss_components[k] += loss_dict[f'{k}_loss'].item()
            
            # Progress bar 업데이트
            current_loss = total_loss / (batch_idx + 1)
            current_components = {
                k: v / (batch_idx + 1) 
                for k, v in loss_components.items()
            }
            current_components['loss'] = current_loss
            
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'cls': f'{current_components["cls"]:.4f}',
                'ptr': f'{current_components["ptr"]:.4f}',
                'empty_ptr': f'{current_components["empty_ptr"]:.4f}'
            })
            
            # 첫 번째 배치의 첫 번째 샘플 시각화
            if first_batch:
                pred_tokens = outputs['tag_logits'][0].argmax(dim=-1)
                pred_otsl = self.tokenizer.decode(pred_tokens.cpu().tolist())
                
                true_tokens = batch['tokens'][0]
                true_otsl = self.tokenizer.decode(true_tokens.cpu().tolist())
                
                pred_html = self.html_converter.convert_otsl_to_html(pred_otsl)
                true_html = self.html_converter.convert_otsl_to_html(true_otsl)
                
                visualize_validation_sample(
                    image=batch['images'][0],
                    boxes=batch['bboxes'][0],
                    pred_html=pred_html,
                    true_html=true_html,
                    pred_otsl=pred_otsl,
                    true_otsl=true_otsl,
                    pointer_logits=outputs['pointer_logits'][0] if 'pointer_logits' in outputs else None,
                    step=self.current_step,
                    loss_components=current_components,
                    viz_dir=self.viz_dir
                )
                first_batch = False
        
        # 평균 계산
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        avg_components['loss'] = avg_loss
        
        # wandb 로깅
        if self.train_config.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/cls_loss': avg_components['cls'],
                'val/ptr_loss': avg_components['ptr'],
                'val/empty_ptr_loss': avg_components['empty_ptr'],
                'val/row_contr_loss': avg_components['row_contr'],
                'val/col_contr_loss': avg_components['col_contr']
            }, step=self.current_step)
        
        # 로깅
        self.logger.info(
            f"Validation Step {self.current_step} - "
            f"Loss: {avg_loss:.4f}, "
            f"CLS: {avg_components['cls']:.4f}, "
            f"PTR: {avg_components['ptr']:.4f}, "
            f"Empty PTR: {avg_components['empty_ptr']:.4f}"
        )
        
        return avg_loss, avg_components
        
    def save_checkpoint(
        self,
        loss: float,
        is_best: bool = False
    ) -> None:
        """체크포인트 저장"""
        checkpoint = {
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.train_config
        }
        
        # 일반 체크포인트 저장
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_step_{self.current_step}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Best 모델 저장
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            
            
