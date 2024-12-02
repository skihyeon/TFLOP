import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .otsl_tokenizer import OTSLTokenizer

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
        tokenizer: OTSLTokenizer = None
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_ptr = lambda_ptr
        self.lambda_empty_ptr = lambda_empty_ptr
        self.lambda_row_contr = lambda_row_contr
        self.lambda_col_contr = lambda_col_contr
        self.temperature = temperature
        self.tokenizer = tokenizer
        
    def compute_pointer_loss(
        self,
        pointer_logits: torch.Tensor,  # (B, N, T)
        box_indices: torch.Tensor,     # (B, N)
        data_tag_mask: torch.Tensor,   # (B, T)
    ) -> torch.Tensor:
        """Layout pointer loss (논문 Equation 2)"""

        # 1. data tag positions에 대해서만 logits 계산
        pointer_logits = pointer_logits.masked_fill(~data_tag_mask.unsqueeze(1), -float('inf'))
        # 2. 각 box에 대해 data tag positions에 대한 log softmax 계산
        log_probs = F.log_softmax(pointer_logits, dim=-1)  # (B, N, T)
        
        # 3. box_indices를 data_tag_mask 내에서의 상대적 위치로 변환
        # -1은 padding된 box를 의미
        valid_box_mask = (box_indices != -1)  # (B, N)
        
        # 4. valid boxes에 대해서만 target probability 계산
        target_log_probs = torch.zeros_like(box_indices, dtype=torch.float)
        for b in range(pointer_logits.size(0)):  # batch 순회
            for n in range(pointer_logits.size(1)):  # box 순회
                if valid_box_mask[b, n]:
                    target_idx = box_indices[b, n]
                    if target_idx < log_probs.size(-1):
                        target_log_probs[b, n] = log_probs[b, n, target_idx]
        
        # 5. loss 계산 - valid boxes에 대해서만
        loss = -target_log_probs * valid_box_mask.float()  # (B, N)
        
        return loss.mean()

    def compute_empty_pointer_loss(
        self,
        empty_logits: torch.Tensor,    # (B, T)
        empty_mask: torch.Tensor       # (B, T)
    ) -> torch.Tensor:
        """Empty Pointer Loss (논문 Equation 3)"""
        # 1. data tag positions에서만 loss 계산
        loss = F.binary_cross_entropy_with_logits(
            empty_logits,
            empty_mask.float(),
            reduction='none'
        )
        
        # 2. data tag positions의 평균만 계산
        valid_positions = empty_mask.float()  # empty cell인 위치만 사용
        num_valid = valid_positions.sum().clamp(min=1)  # 최소 1로 clamp
        loss = (loss * valid_positions).sum() / num_valid
        
        return loss
    

    def compute_span_aware_contrastive_loss(
        self, 
        sim_matrix: torch.Tensor,  # (B, N, N)
        span_coef: torch.Tensor,   # (B, N, N)
    ) -> torch.Tensor:
        """Span-aware Contrastive Loss (논문 Equation 5)"""
        B, N = sim_matrix.size(0), sim_matrix.size(1)
        
        span_coef = span_coef.to(sim_matrix.device)
        diag_mask = torch.eye(N, dtype=torch.bool, device=sim_matrix.device).unsqueeze(0)
        
        total_loss = torch.tensor(0.0, device=sim_matrix.device)
        
        for b in range(B):
            curr_span_coef = span_coef[b]  # (N, N)
            curr_sim = sim_matrix[b]  # (N, N)
            curr_sim = curr_sim / self.temperature
            # Positive/Negative mask
            positive_mask = (curr_span_coef > 0) & ~diag_mask[0]
            negative_mask = ~positive_mask & ~diag_mask[0]
            
            for j in range(N):
                cp_sum = curr_span_coef[j][positive_mask[j]].sum()
                if cp_sum == 0:  # positive sample이 없는 경우 skip
                    continue
                
                # Positive term
                pos_sim = curr_sim[j][positive_mask[j]]  # (P,)
                pos_coef = curr_span_coef[j][positive_mask[j]]  # (P,)
                
                # Negative term
                neg_sim = curr_sim[j][negative_mask[j]]  # (N,)
                
                if len(pos_sim) == 0 or len(neg_sim) == 0:
                    continue
                    
                # Numerically stable 계산
                max_val = torch.max(torch.cat([pos_sim, neg_sim]))
                
                # logsumexp 사용하여 numerical stability 향상
                pos_term = torch.logsumexp(pos_sim - max_val, dim=0) + torch.log(pos_coef.sum())
                neg_term = torch.logsumexp(neg_sim - max_val, dim=0)
                
                # Loss clipping으로 안정성 확보
                loss_j = torch.clamp(-(pos_term - neg_term) / cp_sum, min=-100, max=100)
                total_loss = total_loss + loss_j
        
        return total_loss / (B * N)  # 배치 크기와 bbox 수로 normalize

    def forward(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Batch inputs
        tag_targets: torch.Tensor = batch['tokens']                 # (B, L)
        box_indices: torch.Tensor = batch['box_indices']            # (B, N)
        empty_mask: torch.Tensor = batch['empty_mask']              # (B, T)

        
        # Model outputs
        tag_logits: torch.Tensor = outputs['tag_logits']            # (B, L, V)
        pointer_logits: torch.Tensor = outputs['pointer_logits']    # (B, N, T)
        empty_logits: torch.Tensor = outputs['empty_logits']        # (B, T)
        data_tag_mask: torch.Tensor = outputs['data_tag_mask']      # (B, T)
        row_sim_matrix: torch.Tensor = outputs['row_sim_matrix']    # (B, N, N)
        col_sim_matrix: torch.Tensor = outputs['col_sim_matrix']    # (B, N, N)
        row_span_coef: torch.Tensor = outputs['row_span_coef']      # (B, N, N)
        col_span_coef: torch.Tensor = outputs['col_span_coef']      # (B, N, N)
        
        
        self.special_token_ids = [self.tokenizer.pad_token_id, self.tokenizer.unk_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
        
        """Total Loss (논문 Equation 7)"""
        # 1. Classification Loss
        # logits: (B, L, V) -> (B*L, V)
        # targets: (B, L) -> (B*L)
        B, L, V = tag_logits.shape
        tag_logits = tag_logits.contiguous().view(B*L, V)
        tag_targets = tag_targets.contiguous().view(B*L)
        
        # 모든 special token을 무시하는 마스크 생성
        special_token_ids = [
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id
        ]
        valid_token_mask = ~torch.isin(tag_targets, torch.tensor(special_token_ids, device=tag_targets.device))
        
        # special token을 제외한 위치에서만 loss 계산
        cls_loss = F.cross_entropy(
            tag_logits,                  # (B*L, V)
            tag_targets,                 # (B*L)
            reduction='none'
        )
        cls_loss = (cls_loss * valid_token_mask.float()).sum() / valid_token_mask.sum().clamp(min=1)
        
        # 2. Layout Pointer Loss
        ptr_loss = self.compute_pointer_loss(
            pointer_logits=pointer_logits,  # (B, N, T)
            box_indices=box_indices,        # (B, N)
            data_tag_mask=data_tag_mask,    # (B, T)
        )
        
        # 3. Empty Pointer Loss
        empty_ptr_loss = self.compute_empty_pointer_loss(
            empty_logits=empty_logits,
            empty_mask=empty_mask,
        )
        
        if row_sim_matrix is not None:
            row_contr_loss = self.compute_span_aware_contrastive_loss(
                sim_matrix=row_sim_matrix,
                span_coef=row_span_coef,
            )
        if col_sim_matrix is not None:
            col_contr_loss = self.compute_span_aware_contrastive_loss(
                sim_matrix=col_sim_matrix,
                span_coef=col_span_coef,
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
        # loss 값들 출력 및 검증
        # for key, value in loss_dict.items():
        #     print(f'{key}: {value.item()}')
        # for key, value in loss_dict.items():
        #     if torch.isnan(value):
        #         raise ValueError(f'NaN detected in {key}')
        return loss_dict