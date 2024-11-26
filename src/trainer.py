import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Tuple, Optional, Any, List
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from models import TFLOP, TFLOPLoss
from config import TrainingConfig, TFLOPConfig
from models.otsl_tokenizer import OTSLTokenizer
from utils.html_to_otsl import HTMLtoOTSLConverter
from metrics.teds import compute_teds, compute_teds_struct
import seaborn as sns


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
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.num_epochs,
            eta_min=1e-7
        )
        
        # 체크포인트 디렉토리 생성 및 하위 폴더 설정
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(train_config.checkpoint_dir, current_time)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 로그 파일 설정
        logging.basicConfig(
            filename=os.path.join(self.experiment_dir, 'train.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Tensorboard 설정
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.experiment_dir, 'tensorboard')
        )
        
        # 모델 체크포인트 디렉토리
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'models')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 시각화 결과 저장 디렉토리
        self.viz_dir = os.path.join(self.experiment_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Step 카운터 추가
        self.current_step = 0
        
        # HTML to OTSL 컨버터 추가
        self.html_converter = HTMLtoOTSLConverter()
        
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """한 에폭 학습 (논문 Section 3.7 참조)"""
        self.model.train()
        total_loss = 0
        loss_components = {
            'cls': 0, 'ptr': 0, 'empty_ptr': 0, 
            'row_contr': 0, 'col_contr': 0
        }
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # 1. Forward pass
            outputs = self.model(
                images=batch['images'].to(self.device),
                text_regions=batch['bboxes'].to(self.device),
                labels=batch['tokens'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                row_spans=batch['row_spans'].to(self.device),
                col_spans=batch['col_spans'].to(self.device)
            )
            
            # 2. Loss 계산 (논문 Equation 7)
            loss_dict = self.criterion(
                tag_logits=outputs['tag_logits'],
                tag_targets=batch['tokens'].to(self.device),
                box_features=outputs['box_features'],
                tag_features=outputs['tag_features'],
                box_indices=batch['tokens'].to(self.device),  # box와 매칭되는 token indices
                data_tag_mask=outputs['data_tag_mask'],
                empty_mask=(batch['tokens'] == self.tokenizer.pad_token_id).to(self.device),
                row_spans=batch['row_spans'].to(self.device),
                col_spans=batch['col_spans'].to(self.device)
            )
            
            loss = loss_dict['loss']
            
            # 3. Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 4. Gradient clipping
            if self.train_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_config.max_grad_norm
                )
            
            # 5. Optimizer step
            self.optimizer.step()
            
            # Loss components 기록
            total_loss += loss.item()
            for k in loss_components.keys():
                if f'{k}_loss' in loss_dict:
                    loss_components[k] += loss_dict[f'{k}_loss'].item()
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Tensorboard 로깅
            step = epoch * len(dataloader) + batch_idx
            self.writer.add_scalar('Loss/train_step', loss.item(), step)
            for k, v in loss_components.items():
                self.writer.add_scalar(f'Loss/{k}_step', loss_dict[f'{k}_loss'].item(), step)
            
            # 로깅
            if batch_idx % self.train_config.log_steps == 0:
                self.logger.info(
                    f"Epoch {epoch} Step {batch_idx}/{len(dataloader)} "
                    f"Loss: {loss.item():.4f}"
                )
        
        # 에폭 평균 loss 계산
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
        
    @torch.no_grad()
    def validate(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """검증 (loss 계산만 수행)"""
        self.model.eval()
        total_loss = 0
        loss_components = {
            'cls': 0, 'ptr': 0, 'empty_ptr': 0, 
            'row_contr': 0, 'col_contr': 0
        }
        
        first_batch = True
        pbar = tqdm(dataloader, desc=f'Validation epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # 1. Forward pass
            outputs = self.model(
                images=batch['images'].to(self.device),
                text_regions=batch['bboxes'].to(self.device),
                labels=batch['tokens'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                row_spans=batch['row_spans'].to(self.device),
                col_spans=batch['col_spans'].to(self.device)
            )
            
            # 2. Loss 계산
            loss_dict = self.criterion(
                tag_logits=outputs['tag_logits'],
                tag_targets=batch['tokens'].to(self.device),
                box_features=outputs['box_features'],
                tag_features=outputs['tag_features'],
                box_indices=outputs['tag_logits'].argmax(dim=-1),
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
            current_components['loss'] = current_loss  # total loss 추가
            
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'cls': f'{current_components["cls"]:.4f}',
                'ptr': f'{current_components["ptr"]:.4f}',
                'empty_ptr': f'{current_components["empty_ptr"]:.4f}'
            })
            
            # 첫 번째 배치의 첫 번째 샘플 시각화
            if first_batch:
                # 예측된 토큰 시퀀스 디코딩
                pred_tokens = outputs['tag_logits'][0].argmax(dim=-1)
                pred_otsl = self.tokenizer.decode(pred_tokens.cpu().tolist())
                
                # 실제 토큰 시퀀스 디코딩
                true_tokens = batch['tokens'][0]
                true_otsl = self.tokenizer.decode(true_tokens.cpu().tolist())
                
                # OTSL을 HTML로 변환
                pred_html = self.html_converter.convert_otsl_to_html(pred_otsl)
                true_html = self.html_converter.convert_otsl_to_html(true_otsl)
                
                # 시각화
                self._visualize_validation_sample(
                    image=batch['images'][0],
                    boxes=batch['bboxes'][0],
                    pred_html=pred_html,
                    true_html=true_html,
                    pred_otsl=pred_otsl,
                    true_otsl=true_otsl,
                    pointer_logits=outputs['pointer_logits'][0] if 'pointer_logits' in outputs else None,
                    epoch=epoch,
                    loss_components=current_components
                )
                first_batch = False
        
        # 평균 계산
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        avg_components['loss'] = avg_loss  # total loss 추가
        
        # Tensorboard 로깅
        self.writer.add_scalar('Validation/loss', avg_loss, epoch)
        for k, v in avg_components.items():
            if k != 'loss':  # total loss는 이미 기록됨
                self.writer.add_scalar(f'Validation/{k}_loss', v, epoch)
        
        # 로깅
        self.logger.info(
            f"Validation Epoch {epoch} - "
            f"Loss: {avg_loss:.4f}, "
            f"CLS: {avg_components['cls']:.4f}, "
            f"PTR: {avg_components['ptr']:.4f}, "
            f"Empty PTR: {avg_components['empty_ptr']:.4f}"
        )
        
        return avg_loss, avg_components
    
    def _visualize_validation_sample(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        pred_html: str,
        true_html: str,
        pred_otsl: str,
        true_otsl: str,
        pointer_logits: Optional[torch.Tensor],
        epoch: int,
        loss_components: Dict[str, float]
    ) -> None:
        """검증 샘플 시각화"""
        plt.figure(figsize=(20, 8))  # 높이를 줄임 (2x2에서 2x1로 변경)
        
        # 1. GT 박스와 Pred 박스를 별도의 이미지에 시각화
        # 이미지 역정규화
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = image.cpu().numpy()
        for i in range(3):
            img[i] = img[i] * std[i].item() + mean[i].item()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        
        H, W = image.shape[1:]
        
        # 1-1. GT 박스 시각화 (왼쪽)
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.set_title('Ground Truth Boxes')
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle(
                (x1 * W, y1 * H),
                (x2 - x1) * W,
                (y2 - y1) * H,
                linewidth=2,
                edgecolor='r',
                facecolor='none',
                alpha=0.7
            )
            ax1.add_patch(rect)
            
            # 박스 번호 표시
            ax1.text(x1 * W, y1 * H, f"Box {i}", 
                    color='black', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # 1-2. Pred 박스 시각화 (오른쪽)
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(img)
        ax2.set_title('Predicted Box-Tag Matching')
        
        if pointer_logits is not None:
            max_indices = pointer_logits.argmax(dim=-1)
            max_values = pointer_logits.max(dim=-1)[0]
            
            # confidence에 따른 색상 맵 생성
            cmap = plt.cm.get_cmap('YlOrRd')
            
            for i, box in enumerate(boxes):
                if max_indices[i] > 0:  # 0은 패딩
                    x1, y1, x2, y2 = box.cpu().numpy()
                    confidence = torch.sigmoid(max_values[i]).item()  # 0~1 사이 값으로 변환
                    
                    rect = patches.Rectangle(
                        (x1 * W, y1 * H),
                        (x2 - x1) * W,
                        (y2 - y1) * H,
                        linewidth=2,
                        edgecolor=cmap(confidence),
                        facecolor=cmap(confidence * 0.3),  # 투명도 있는 채우기
                        alpha=0.7,
                        linestyle='--'
                    )
                    ax2.add_patch(rect)
                    
                    # OTSL 태그로 변환
                    otsl_tag = 'C'  # 기본값
                    if max_indices[i] in self.tokenizer.id2token:
                        otsl_tag = self.tokenizer.id2token[max_indices[i].item()]
                    
                    # 박스 번호, 매칭된 OTSL 태그, confidence 표시
                    ax2.text(x1 * W, y1 * H, 
                            f"Box {i}\n{otsl_tag}\n{confidence:.2f}", 
                            color='black', fontsize=8, 
                            bbox=dict(facecolor='white', alpha=0.7))
        
        # 2. Loss components와 OTSL/HTML 정보를 하단에 표시
        plt.figtext(0.1, 0.02, 
                    f"Loss: {loss_components['loss']:.4f} | "
                    f"CLS: {loss_components['cls']:.4f} | "
                    f"PTR: {loss_components['ptr']:.4f} | "
                    f"Empty PTR: {loss_components['empty_ptr']:.4f}",
                    fontsize=10)
        
        plt.figtext(0.1, -0.08,
                    f"Pred OTSL: {pred_otsl[:100]}...\n"
                    f"True OTSL: {true_otsl[:100]}...\n"
                    f"Pred HTML: {pred_html[:100]}...\n"
                    f"True HTML: {true_html[:100]}...",
                    fontsize=8)
        
        plt.tight_layout()
        
        # 저장
        save_path = os.path.join(self.viz_dir, f'val_epoch_{epoch}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.2)
        plt.close()
        
    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> None:
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.train_config
        }
        
        # 일반 체크포인트 저장
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Best 모델 저장
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def train(self, train_dataloader, val_dataloader):
        """전체 학습 과정 (논문 Section 4.2)"""
        best_val_loss = float('inf')
        self.current_step = 0
        
        # 논문: 250K steps 학습
        total_steps = 250000
        current_epoch = 0
        
        while self.current_step < total_steps:
            self.logger.info(f"\nEpoch {current_epoch+1}")
            
            # Train
            train_loss, train_components = self.train_epoch(train_dataloader, current_epoch)
            
            # Validate
            if self.current_step % self.train_config.eval_steps == 0:
                val_loss, val_components = self.validate(val_dataloader, current_epoch)
                
                # Tensorboard logging
                self.writer.add_scalar('Loss/train', train_loss, self.current_step)
                self.writer.add_scalar('Loss/val', val_loss, self.current_step)
                
                for k in train_components:
                    self.writer.add_scalar(f'Loss_Components/train_{k}', 
                                         train_components[k], 
                                         self.current_step)
                    self.writer.add_scalar(f'Loss_Components/val_{k}', 
                                         val_components[k], 
                                         self.current_step)
                
                # Save checkpoint
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    self.logger.info(f"New best model with val_loss: {val_loss:.4f}")
                
                if self.current_step % self.train_config.save_steps == 0:
                    self.save_checkpoint(current_epoch, val_loss, is_best)
            
            current_epoch += 1
            
            # total_steps에 도달하면 학습 종료
            if self.current_step >= total_steps:
                self.logger.info(f"Reached total steps {total_steps}. Training completed!")
                break
        
        self.writer.close()
        self.logger.info("Training completed!") 