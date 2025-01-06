import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .otsl_tokenizer import OTSLTokenizer
from config import ModelConfig

class TFLOPLoss(nn.Module):
    """
    TFLOP Loss Function (논문 Section 3.7)
    
    L = λ1*Lcls + λ2*Lptr + λ3*Lempty_ptr + λ4*Σ(Lrow_contr)/B + λ5*Σ(Lcol_contr)/B
    """
    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_ptr: float = 1.0,
        lambda_empty_ptr: float = 1.0,
        lambda_row_contr: float = 0.5,
        lambda_col_contr: float = 0.5,
        temperature: float = 0.1,
        tokenizer: OTSLTokenizer = None,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided")
        self.config = ModelConfig()
        self.tokenizer = tokenizer
        self.lambda_cls = lambda_cls
        self.lambda_ptr = lambda_ptr
        self.lambda_empty_ptr = lambda_empty_ptr
        self.lambda_row_contr = lambda_row_contr
        self.lambda_col_contr = lambda_col_contr
        self.temperature = temperature
        
        # length 지정
        self.layout_prompt_length = self.config.total_sequence_length - self.config.otsl_max_length

    def compute_pointer_loss(self, pointer_logits: torch.Tensor, 
                            box_indices: torch.Tensor, 
                            data_tag_mask: torch.Tensor,
                            temperature: float = 0.1) -> torch.Tensor:
        B, num_boxes, seq_len = pointer_logits.shape
        
        # Temperature scaling in softmax
        log_probs = F.log_softmax(pointer_logits / temperature, dim=-1)
        
        # data_tag_mask는 이미 layout_prompt 이후의 전체 시퀀스를 커버
        valid_tag_positions = data_tag_mask[:, self.layout_prompt_length:]
        valid_tag_positions = valid_tag_positions[:, :seq_len]
        mask = valid_tag_positions.unsqueeze(1).expand(-1, num_boxes, -1)
        log_probs = log_probs.masked_fill(~mask, float('-inf'))
        
        # box_indices는 이미 BOS를 고려하여 조정된 상태
        valid_boxes = box_indices != -1
        
        total_loss = 0.0
        total_count = 0
        
        for b in range(B):
            for n in range(num_boxes):
                valid_targets = box_indices[b, n][valid_boxes[b, n]]
                if len(valid_targets) > 0:
                    curr_loss = -log_probs[b, n, valid_targets].mean()
                    total_loss += curr_loss
                    total_count += 1
        
        return total_loss / (total_count + 1e-6)

    def compute_empty_pointer_loss(self, empty_logits: torch.Tensor,
                             data_tag_mask: torch.Tensor,
                             empty_tag_mask: torch.Tensor) -> torch.Tensor:
        if empty_logits.dim() == 3:
            empty_logits = empty_logits.squeeze(1)
        
        # layout_prompt 이후의 전체 시퀀스에 대해 마스크 적용
        otsl_mask = (data_tag_mask | empty_tag_mask)[:, self.layout_prompt_length:]
        targets = empty_tag_mask[:, self.layout_prompt_length:]
        
        # empty cell 비율에 따른 가중치 계산
        num_empty = (targets * otsl_mask).sum()
        num_total = otsl_mask.sum()
        pos_weight = (num_total - num_empty) / (num_empty + 1e-8)
        
        bce_loss = F.binary_cross_entropy_with_logits(
            empty_logits,
            targets.float(),
            pos_weight=pos_weight * torch.ones_like(targets),
            reduction='none'
        )
        
        # C 태그 위치에 대해서만 loss 계산
        masked_loss = bce_loss * otsl_mask
        num_c_tags = otsl_mask.sum(dim=1, keepdim=True) + 1e-8
        batch_losses = masked_loss.sum(dim=1) / num_c_tags.squeeze()
        
        return batch_losses.mean()
    
    def compute_span_aware_contrastive_loss(
            self,
            sim_matrix: torch.Tensor,
            span_coef_matrix: torch.Tensor,
        ) -> torch.Tensor:
        device = sim_matrix.device
        span_coef_matrix = span_coef_matrix.to(device)
        
        sim_matrix = sim_matrix / self.temperature
        
        exp_sim = torch.exp(sim_matrix)
        
        denominator = exp_sim.sum(dim=-1, keepdim=True)
        
        log_probs = torch.log(exp_sim / denominator + 1e-8)
        
        valid_mask = (span_coef_matrix >= 0)
        
        weighted_log_probs = span_coef_matrix * log_probs * valid_mask
        
        coef_sum = span_coef_matrix.sum(dim=-1) + 1e-8
        box_losses = -(weighted_log_probs.sum(dim=-1) / coef_sum)
        
        valid_boxes = valid_mask.any(dim=-1)
        final_loss = (box_losses * valid_boxes).sum() / (valid_boxes.sum() + 1e-8)
        
        return final_loss
            
    def compute_tag_loss(self, tag_logits, tag_targets):
        """
        Args:
            tag_logits: (B, S1, V) - 생성된 시퀀스의 logits
            tag_targets: (B, S2) - target 시퀀스 (S2가 S1보다 길 수 있음)
        """
        B, S1, V = tag_logits.shape
        _, S2 = tag_targets.shape
        
        if S1 < S2:
            pad_logits = torch.full(
                (B, S2-S1, V), 
                float('-inf'),
                device=tag_logits.device
            )
            pad_logits[:, :, self.tokenizer.pad_token_id] = 0
            tag_logits = torch.cat([tag_logits, pad_logits], dim=1)
        
        # EOS 토큰 위치 찾기
        eos_positions = (tag_targets == self.tokenizer.eos_token_id)
        
        # 가중치 텐서 생성 (기본값 1.0)
        weights = torch.ones_like(tag_targets, dtype=torch.float)
        # EOS 토큰 위치의 가중치를 1.2으로 설정
        weights[eos_positions] = 1.4
        
        # Cross entropy loss 계산
        loss = F.cross_entropy(
            tag_logits.reshape(-1, V),
            tag_targets.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=0.0,
            reduction='none'  # 각 토큰별 loss 반환
        )
        
        # 가중치 적용
        weighted_loss = loss * weights.reshape(-1)
        
        # 유효한 토큰 수로 나누어 평균 계산
        valid_tokens = (tag_targets != self.tokenizer.pad_token_id).sum()
        return weighted_loss.sum() / (valid_tokens + 1e-8)
        
    def forward(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # Batch inputs
        tag_targets: torch.Tensor = batch['token_ids']                 # (B, 688)
        box_indices: torch.Tensor = batch['box_indices']            # (B, N, M)
        data_tag_mask: torch.Tensor = batch['data_tag_mask']        # (B, 1376)
        empty_tag_mask: torch.Tensor = batch['empty_tag_mask']      # (B, 1376)
        
        # Model outputs
        tag_logits: torch.Tensor = outputs['tag_logits']            # (B, 688, V)
        pointer_logits: torch.Tensor = outputs['pointer_logits']    # (B, 688, 688)
        empty_pointer_logits: torch.Tensor = outputs['empty_pointer_logits']    # (B, 1, 688)
        row_sim_matrix: torch.Tensor = outputs['row_sim_matrix']    # (B, 688, 688)
        col_sim_matrix: torch.Tensor = outputs['col_sim_matrix']    # (B, 688, 688)
        row_span_coef: torch.Tensor = outputs['row_span_coef']      # (B, 688, 688)
        col_span_coef: torch.Tensor = outputs['col_span_coef']      # (B, 688, 688)
        
        # 유효한 토큰에 대해서만 loss 계산
        cls_loss = self.compute_tag_loss(
            tag_logits=tag_logits,
            tag_targets=tag_targets,
        )

        # 2. Layout Pointer Loss (Equation 2)
        ptr_loss = self.compute_pointer_loss(
            pointer_logits=pointer_logits,
            box_indices=box_indices,
            data_tag_mask=data_tag_mask,
            temperature=self.temperature,
        )

        # 3. Empty Pointer Loss (Equation 3)
        empty_ptr_loss = self.compute_empty_pointer_loss(
            empty_logits=empty_pointer_logits,
            data_tag_mask=data_tag_mask,
            empty_tag_mask=empty_tag_mask
        )
        
        # 4. Span-aware Contrastive Loss (Equation 5)
        row_contr_loss = torch.tensor(0.0, device=tag_logits.device)
        col_contr_loss = torch.tensor(0.0, device=tag_logits.device)
        
        if row_sim_matrix is not None:
            row_contr_loss = self.compute_span_aware_contrastive_loss(
                sim_matrix=row_sim_matrix,
                span_coef_matrix=row_span_coef,
            )

        if col_sim_matrix is not None:
            col_contr_loss = self.compute_span_aware_contrastive_loss(
                sim_matrix=col_sim_matrix,
                span_coef_matrix=col_span_coef,
            )
        # 5. Total Loss (Equation 7)
        total_loss = (
            self.lambda_cls * cls_loss +
            self.lambda_ptr * ptr_loss +
            self.lambda_empty_ptr * empty_ptr_loss +
            self.lambda_row_contr * row_contr_loss +
            self.lambda_col_contr * col_contr_loss
        )
        
        loss_dict = {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'ptr_loss': ptr_loss,
            'empty_ptr_loss': empty_ptr_loss,
            'row_contr_loss': row_contr_loss,
            'col_contr_loss': col_contr_loss
        }
        
        # for key, value in loss_dict.items():
            # print(f"{key}: {value}")
        
        return loss_dict
