import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

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
        pad_token_id: int = 1
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_ptr = lambda_ptr
        self.lambda_empty_ptr = lambda_empty_ptr
        self.lambda_row_contr = lambda_row_contr
        self.lambda_col_contr = lambda_col_contr
        self.pad_token_id = pad_token_id
        

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
        
        # 6. batch와 valid boxes에 대해 평균
        num_valid_boxes = valid_box_mask.sum(dim=1).clamp(min=1)  # (B,)
        loss = loss.sum(dim=1) / num_valid_boxes  # (B,)
        
        return loss.mean()

    def compute_empty_pointer_loss(
        self,
        empty_logits: torch.Tensor,    # (B, T)
        empty_mask: torch.Tensor       # (B, T)
    ) -> torch.Tensor:
        """Empty Pointer Loss (논문 Equation 3)
        Lempty_ptr = -1/|D| * Σ(k'∈D) BCE(σ(b̄0·t̄k'), I(k'))
        """
        # Binary cross entropy loss
        num_data_tags = empty_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # |D|
        loss = F.binary_cross_entropy_with_logits(
            empty_logits,
            empty_mask.float(),
            reduction='none'
        )
        
        # Average over data tags only
        loss = (loss * empty_mask.float()).sum(dim=-1) / num_data_tags
        loss = loss.mean()
        
        return loss
    

    def compute_span_aware_contrastive_loss(
        self,
        sim_matrix: torch.Tensor,     # (B, N, N)
        span_coef: torch.Tensor,      # (B, N, N)
    ) -> torch.Tensor:
        """Span-aware Contrastive Loss (논문 Equation 5)"""
        B, N, _ = sim_matrix.size()
        
        # 1. Compute masks
        mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device).unsqueeze(0)
        positive_mask = (span_coef > 0) & mask  # P(j)
        negative_mask = mask & ~positive_mask   # A(j)
        
        # 2. Compute exp(sim/τ)
        exp_sim = torch.exp(sim_matrix)  # exp(b̂j·b̂p/τ)
        
        # 3. Compute denominator (Σ(a∈A(j))exp(b̂j·b̂a/τ))
        denominator = torch.sum(exp_sim * negative_mask.float(), dim=-1, keepdim=True)
        
        # 4. Compute log(exp(b̂j·b̂p/τ) / Σ(a∈A(j))exp(b̂j·b̂a/τ))
        log_probs = torch.log(exp_sim / (denominator + 1e-8))
        
        # 5. Compute loss for each positive pair
        loss = -(log_probs * positive_mask.float())
        
        # 6. Average over positive pairs
        num_positive = positive_mask.float().sum(dim=-1)
        loss = loss.sum(dim=-1) / (num_positive + 1e-8)
        
        return loss.mean()

    def forward(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Batch inputs
        tag_targets: torch.Tensor = batch['tokens']                 # (B, L)
        box_indices: torch.Tensor = batch['box_indices']            # (B, N)
        empty_mask: torch.Tensor = batch['empty_mask']              # (B, T)
        row_span_coef: torch.Tensor = batch['row_span_coef']        # (B, N, N)
        col_span_coef: torch.Tensor = batch['col_span_coef']        # (B, N, N)
        
        # Model outputs
        tag_logits: torch.Tensor = outputs['tag_logits']            # (B, L, V)
        pointer_logits: torch.Tensor = outputs['pointer_logits']    # (B, N, T)
        empty_logits: torch.Tensor = outputs['empty_logits']        # (B, T)
        data_tag_mask: torch.Tensor = outputs['data_tag_mask']      # (B, T)
        row_sim_matrix: torch.Tensor = outputs['row_sim_matrix']    # (B, N, N)
        col_sim_matrix: torch.Tensor = outputs['col_sim_matrix']    # (B, N, N)
        
        """Total Loss (논문 Equation 7)"""
        # 1. Classification Loss
        cls_loss = F.cross_entropy(
            tag_logits.reshape(-1, tag_logits.size(-1)).to(torch.float32),
            tag_targets.reshape(-1),
            ignore_index=self.pad_token_id
        )
        
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

        return loss_dict