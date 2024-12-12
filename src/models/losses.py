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
        tokenizer: OTSLTokenizer = None,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided")
        self.tokenizer = tokenizer
        self.lambda_cls = lambda_cls
        self.lambda_ptr = lambda_ptr
        self.lambda_empty_ptr = lambda_empty_ptr
        self.lambda_row_contr = lambda_row_contr
        self.lambda_col_contr = lambda_col_contr
        self.temperature = temperature
        
        # length 지정
        self.layout_prompt_length = self.otsl_sequence_length = tokenizer.otsl_sequence_length
        
    def compute_pointer_loss(self, pointer_logits, box_indices, data_tag_mask):
        """
        Args:
            pointer_logits: (B, :688, 688:) 
            box_indices: (B, N, M)
            data_tag_mask: (B, 1376) C태그이면서 has_data=True인 위치만 True
        """
        batch_size = pointer_logits.size(0)
        device = pointer_logits.device
        
        # Temperature scaling
        scaled_logits = pointer_logits  # (B, 688, 688) # 이미 스케일링 되어있음
        
        # data_tag_mask 준비 (B, 688)
        sequence_mask = data_tag_mask[:, self.layout_prompt_length:]
        
        # 수치 안정성을 위한 max normalization
        max_logits = torch.max(scaled_logits, dim=-1, keepdim=True)[0]
        scaled_logits = scaled_logits - max_logits
        
        # exp(logits) 계산 (B, 688, 688)
        exp_logits = torch.exp(scaled_logits)
        
        # 분모 계산: masked sum (B, 688, 1)
        denominator = torch.sum(
            exp_logits * sequence_mask.unsqueeze(1),
            dim=-1,
            keepdim=True
        ) + 1e-10
        
        # Valid한 box indices에 대한 마스크 생성 (B, 688)
        valid_box_mask = (box_indices != -1).any(dim=-1)
        
        # Loss 계산을 위한 target 마스크 준비
        target_mask = torch.zeros_like(exp_logits, dtype=torch.bool)
        for b in range(batch_size):
            for box_idx in range(self.layout_prompt_length):
                if valid_box_mask[b, box_idx]:
                    valid_targets = box_indices[b, box_idx][box_indices[b, box_idx] != -1]
                    target_mask[b, box_idx, valid_targets] = True
        
        # Masked loss 계산
        valid_probs = exp_logits / denominator
        losses = -torch.log(valid_probs + 1e-10)
        masked_losses = losses * (target_mask & sequence_mask.unsqueeze(1))
        
        # 유효한 샘플 수 계산
        valid_count = (masked_losses != 0).sum()
        
        if valid_count == 0:
            return torch.tensor(0.0, device=device)
        
        total_loss = masked_losses.sum()
        
        return total_loss / valid_count


    def compute_empty_pointer_loss(self, empty_pointer_logits, data_tag_mask, attention_mask):
        """
        Args:
            empty_pointer_logits: (B, 1, 688:)
            data_tag_mask: (B, 1376)
            attention_mask: (B, 1376)
        """
        device = empty_pointer_logits.device
        
        # 마스크 준비 (B, 688)
        sequence_mask = attention_mask[:, self.layout_prompt_length:]
        data_mask = data_tag_mask[:, self.layout_prompt_length:]
        
        # Target 계산 (B, 688)
        targets = sequence_mask & ~data_mask
        
        # Masked BCE loss
        logits = empty_pointer_logits.squeeze(1)  # (B, 688)
        valid_mask = sequence_mask
        
        loss = F.binary_cross_entropy_with_logits(
            logits[valid_mask],
            targets[valid_mask].float(),
            reduction='mean'
        )
        
        return loss

    def compute_span_aware_contrastive_loss(
        self,
        sim_matrix: torch.Tensor,
        span_coef_matrix: torch.Tensor,
    ) -> torch.Tensor:
        device = sim_matrix.device
        span_coef_matrix = span_coef_matrix.to(device)
        # Mask out invalid positions (where span_coef == -1)
        valid_mask = (span_coef_matrix >= 0)
        pos_mask = (span_coef_matrix > 0)
        
        # Temperature scaling
        sim_matrix = sim_matrix / self.temperature
        
        # Numerical stability
        sim_matrix_max, _ = torch.max(sim_matrix * valid_mask - 1e8 * (~valid_mask), dim=-1, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()
        
        # exp(sim) with masking
        exp_sim = torch.exp(sim_matrix) * valid_mask  # zero out invalid positions
        
        # Denominator calculation (exclude self connections)
        diag_mask = ~torch.eye(exp_sim.size(1), device=device, dtype=torch.bool).unsqueeze(0)
        denominator = torch.sum(exp_sim * diag_mask, dim=-1, keepdim=True) + 1e-8
        
        # Log probability calculation
        log_probs = sim_matrix - torch.log(denominator)
        
        # Mask out invalid positions in final loss
        weighted_loss = -(span_coef_matrix * log_probs * pos_mask)
        
        # Normalize by positive coefficients
        norm_factor = torch.sum(span_coef_matrix * pos_mask, dim=-1) + 1e-8
        row_losses = torch.sum(weighted_loss, dim=-1) / norm_factor
        
        # Only consider valid rows
        valid_rows = (norm_factor > 1e-8)
        final_losses = row_losses * valid_rows
    
        valid_count = valid_rows.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=device)
        
        return final_losses.sum() / valid_count   
        
    def compute_tag_loss(self, tag_logits, tag_targets):
        B, S, C = tag_logits.shape
        tag_logits = tag_logits.contiguous().view(-1, C)
        tag_targets = tag_targets.contiguous().view(-1)
        
        # PAD 토큰 포함하여 loss 계산
        cls_loss = F.cross_entropy(
            tag_logits,
            tag_targets,
            reduction='mean',
        )
        return cls_loss


    def forward(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # Batch inputs
        tag_targets: torch.Tensor = batch['token_ids']                 # (B, 688)
        box_indices: torch.Tensor = batch['box_indices']            # (B, N, M)
        data_tag_mask: torch.Tensor = batch['data_tag_mask']        # (B, 1376)
        attention_mask: torch.Tensor = batch['attention_mask']      # (B, 1376)
        
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
        )

        # 3. Empty Pointer Loss (Equation 3)
        empty_ptr_loss = self.compute_empty_pointer_loss(
            empty_pointer_logits=empty_pointer_logits,
            data_tag_mask=data_tag_mask,
            attention_mask=attention_mask,
        )
        
        return {
            'loss': cls_loss + ptr_loss + empty_ptr_loss,
            'cls_loss': cls_loss,
            'ptr_loss': ptr_loss,
            'empty_ptr_loss': empty_ptr_loss,
            'row_contr_loss': torch.tensor(0.0, device=tag_logits.device),
            'col_contr_loss': torch.tensor(0.0, device=tag_logits.device)
        }
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
        
        return loss_dict
