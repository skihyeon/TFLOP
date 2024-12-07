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
        
        # Special tokens 미리 설정
        self.special_token_ids = [
            self.tokenizer.pad_token_id,
            # self.tokenizer.unk_token_id,
            # self.tokenizer.bos_token_id,
            # self.tokenizer.eos_token_id
        ]
        
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
        total_loss = 0
        valid_count = 0
        
        for batch_idx in range(batch_size):
            batch_data_tag_mask = data_tag_mask[batch_idx, self.layout_prompt_length:]
            
            for box_idx in range(self.layout_prompt_length):
                if not (box_indices[batch_idx, box_idx] != -1).any():
                    continue
                    
                logits = pointer_logits[batch_idx, box_idx] / self.temperature
                exp_logits = torch.exp(logits)
                denominator = exp_logits[batch_data_tag_mask].sum()
                
                for target_idx in box_indices[batch_idx, box_idx]:
                    if target_idx == -1:
                        break
                    
                    if batch_data_tag_mask[target_idx]:
                        numerator = exp_logits[target_idx]
                        loss = -torch.log(numerator / (denominator + 1e-10))
                        total_loss += loss
                        valid_count += 1
        
        return total_loss / valid_count if valid_count > 0 else torch.tensor(0.0)

    def compute_empty_pointer_loss(self, empty_pointer_logits, data_tag_mask, attention_mask):
        """
        Args:
            empty_pointer_logits: (B, 1, 688:)
            data_tag_mask: C태그이면서 has_data=True인 위치만 True
            attention_mask: 실제 토큰이 있는 위치는 True, 패딩은 False
        """
        B = empty_pointer_logits.size(0)
        total_loss = 0
        valid_count = 0
        
        for batch_idx in range(B):
            batch_data_tag_mask = data_tag_mask[batch_idx, self.layout_prompt_length:]
            batch_attention_mask = attention_mask[batch_idx, self.layout_prompt_length:]
            
            logits = empty_pointer_logits[batch_idx].squeeze(0)
            targets = batch_attention_mask & ~batch_data_tag_mask
            
            loss = F.binary_cross_entropy_with_logits(
                logits[batch_attention_mask],
                targets[batch_attention_mask].float(),
                reduction='mean'
            )
            
            total_loss += loss
            valid_count += 1
        
        return total_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=total_loss.device)
    
    def compute_span_aware_contrastive_loss(
        self,
        sim_matrix: torch.Tensor,      # (B, layout_prompt_length, layout_prompt_length)
        span_coef_matrix: torch.Tensor, # (B, layout_prompt_length, layout_prompt_length)
    ) -> torch.Tensor:
        """Span-aware contrastive loss 계산 (논문 Equation 5)의 최적화 버전
        
        Args:
            sim_matrix: similarity matrix
            span_coef_matrix: coefficient matrix (-1로 패딩된 부분은 무시)
        """
        device = sim_matrix.device
        span_coef_matrix = span_coef_matrix.to(device)
        
        # Temperature scaling & 수치 안정성을 위한 max normalization
        sim_matrix = sim_matrix / self.temperature
        max_sim = torch.max(sim_matrix, dim=-1, keepdim=True)[0]
        sim_matrix = sim_matrix - max_sim
        exp_sim = torch.exp(sim_matrix)  # (B, N, N)
        
        # Valid한 coefficient mask 계산 (B, N)
        valid_coef_mask = (span_coef_matrix >= 0).any(dim=-1)
        
        # Positive sample mask 계산 (B, N)
        pos_mask = (span_coef_matrix > 0)
        
        # 분모 계산: exp(sim) 합, 자기 자신 제외 (B, N)
        diag_mask = torch.eye(exp_sim.size(1), device=device, dtype=torch.bool).unsqueeze(0)
        denominator = exp_sim.sum(dim=-1) - exp_sim.masked_select(diag_mask).view(exp_sim.size(0), -1)
        
        # 각 샘플에 대한 loss 계산을 벡터화
        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        # 유효한 샘플에 대해서만 계산
        for b in range(sim_matrix.size(0)):
            valid_indices = torch.where(valid_coef_mask[b] & pos_mask[b].any(dim=-1))[0]
            
            if valid_indices.size(0) == 0:
                continue
                
            for j in valid_indices:
                curr_pos_mask = pos_mask[b, j]
                pos_exp_sim = exp_sim[b, j][curr_pos_mask]
                pos_coef = span_coef_matrix[b, j][curr_pos_mask]
                
                log_probs = torch.log(pos_exp_sim / (denominator[b, j] + 1e-10))
                loss = -(pos_coef * log_probs).sum() / (pos_coef.sum() + 1e-10)
                
                total_loss += loss
                valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=device)
            
        return total_loss / valid_samples
    
    def compute_tag_loss(self, tag_logits, tag_targets):
        B, S, C = tag_logits.shape
        tag_logits = tag_logits.view(-1, C)      # (B*688, 9)
        tag_targets = tag_targets.view(-1)        # (B*688)
        
        # PAD와 UNK만 무시
        ignore_tokens = [self.tokenizer.pad_token_id, self.tokenizer.unk_token_id]
        valid_mask = ~torch.isin(tag_targets, torch.tensor(ignore_tokens, device=tag_targets.device))
        
        # 유효한 토큰에 대해서만 loss 계산
        valid_logits = tag_logits[valid_mask]
        valid_targets = tag_targets[valid_mask]
        
        # OTSL 토큰들에 대해서만 weight 계산
        otsl_token_ids = [
            self.tokenizer.c_token_id,
            self.tokenizer.l_token_id,
            self.tokenizer.u_token_id,
            self.tokenizer.x_token_id,
            self.tokenizer.nl_token_id
        ]
        
        # Class weights 계산 (OTSL 토큰만)
        unique_labels, counts = torch.unique(valid_targets, return_counts=True)
        total = counts.sum()
        weights = torch.ones(tag_logits.shape[-1], device=tag_logits.device)
        
        for label, count in zip(unique_labels, counts):
            label_item = label.item()
            if label_item in otsl_token_ids:
                weights[label_item] = total/count.item()
        
        cls_loss = F.cross_entropy(
            valid_logits,
            valid_targets,
            weight=weights,
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