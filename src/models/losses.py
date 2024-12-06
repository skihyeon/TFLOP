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
        focal_alpha: float = 0.25,  # focal loss alpha parameter
        focal_gamma: float = 2.0,   # focal loss gamma parameter
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
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Special tokens 미리 설정
        self.special_token_ids = [
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id
        ]
        
        # length 지정
        self.layout_prompt_length = self.otsl_sequence_length = tokenizer.otsl_sequence_length
        
        # Class weights for tag classification
        self.tag_weights = None  # 동적으로 계산될 예정
        self.tag_counts = torch.zeros(tokenizer.vocab_size)  # 태그별 카운트 저장
        self.total_tags = 0

    def update_tag_weights(self, tag_targets: torch.Tensor):
        """태그 분포에 따른 가중치 동적 업데이트"""
        # Count tags
        for tag in tag_targets.unique():
            count = (tag_targets == tag).sum().item()
            self.tag_counts[tag] += count
            self.total_tags += count
        
        # Calculate weights (excluding special tokens)
        valid_tag_counts = self.tag_counts.clone()
        for special_token in self.special_token_ids:
            valid_tag_counts[special_token] = 0
        
        # Inverse frequency weighting
        non_zero_counts = valid_tag_counts > 0
        weights = torch.zeros_like(valid_tag_counts)
        weights[non_zero_counts] = self.total_tags / (valid_tag_counts[non_zero_counts] * non_zero_counts.sum())
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights[non_zero_counts])
        self.tag_weights = weights.to(tag_targets.device)
        
    def compute_pointer_loss(self, box_features, tag_features, box_indices, data_tag_mask):
        """
        data_tag_mask: C태그이면서 has_data=True인 위치만 True
        """
        batch_size = box_features.size(0)
        total_loss = 0
        valid_count = 0
        
        for batch_idx in range(batch_size):
            # OTSL sequence 부분의 data tag mask
            batch_data_tag_mask = data_tag_mask[batch_idx, self.layout_prompt_length:]  # D: data tag 집합 [B, 688: ]
            
            for box_idx in range(self.layout_prompt_length):  # (0~688)
                if not (box_indices[batch_idx, box_idx] != -1).any():
                    continue
                    
                box_feat = box_features[batch_idx, box_idx]
                logits = torch.matmul(box_feat, tag_features[batch_idx].t())
                logits = logits / self.temperature
                
                exp_logits = torch.exp(logits)
                # denominator는 모든 data tag에 대해 계산
                denominator = exp_logits[batch_data_tag_mask].sum()
                
                for target_idx in box_indices[batch_idx, box_idx]:
                    if target_idx == -1:
                        break
                    
                    relative_idx = target_idx - self.layout_prompt_length
                    if batch_data_tag_mask[relative_idx]:  # target이 data tag인 경우만
                        numerator = exp_logits[relative_idx]
                        loss = -torch.log(numerator / (denominator + 1e-10))
                        total_loss += loss
                        valid_count += 1
        
        return total_loss / valid_count if valid_count > 0 else torch.tensor(0.0)
    
    def compute_empty_pointer_loss(self, empty_embedding, tag_features, data_tag_mask, attention_mask):
        """Empty cell detection loss 계산
        Args:
            data_tag_mask: C태그이면서 has_data=True인 위치만 True
            attention_mask: 실제 토큰이 있는 위치는 True, 패딩은 False
        """
        B = empty_embedding.size(0)
        total_loss = 0
        valid_count = 0
        
        for batch_idx in range(B):
            batch_data_tag_mask = data_tag_mask[batch_idx, self.layout_prompt_length:]
            batch_attention_mask = attention_mask[batch_idx, self.layout_prompt_length:]
            
            logits = torch.matmul(empty_embedding[batch_idx], tag_features[batch_idx].t())
            logits = logits.squeeze(0)
            
            # 실제 토큰이 있는 위치에서만 empty cell 판단
            # True = empty cell (C태그이지만 has_data=False)
            # False = non-empty cell (C태그이면서 has_data=True)
            targets = batch_attention_mask & ~batch_data_tag_mask
            
            # 패딩된 위치는 loss 계산에서 제외
            loss = F.binary_cross_entropy_with_logits(
                logits[batch_attention_mask],
                targets[batch_attention_mask].float(),
                reduction='mean'
            )
            
            total_loss += loss
            valid_count += 1
        
        return total_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=total_loss.device)
    
    # def compute_span_aware_contrastive_loss(
    #     self,
    #     sim_matrix: torch.Tensor,      # (B, 688, 688)
    #     span_coef: torch.Tensor,       # (B, 688, 688)
    #     attention_mask: torch.Tensor,  # (B, 688)
    # ) -> torch.Tensor:
    #     """
    #     논문 Equation 5의 Span-aware contrastive loss 구현
    #     2D mask를 사용하여 패딩된 위치 제외
    #     """
    #     B, N, _ = sim_matrix.shape
    #     device = sim_matrix.device
        
    #     # Temperature scaling 적용
    #     sim_matrix = sim_matrix / self.temperature
        
    #     # 수치 안정성을 위한 max normalization
    #     max_sim = torch.max(sim_matrix, dim=-1, keepdim=True)[0]
    #     sim_matrix = sim_matrix - max_sim
        
    #     # exp(sim) 계산
    #     exp_sim = torch.exp(sim_matrix)  # (B, N, N)
        
    #     # attention mask를 2D로 확장 (B, N, N)
    #     attention_mask = attention_mask[:, self.layout_prompt_length:]
    #     mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # (B, N, N)
        
    #     total_loss = torch.tensor(0.0, device=device)
    #     valid_samples = 0
        
    #     for b in range(B):
    #         for j in range(N):
    #             # 패딩된 위치면 건너뛰기
    #             if not attention_mask[b, j]:
    #                 continue
                    
    #             # positive pairs 찾기 (span coefficient > 0 && valid position)
    #             pos_mask = (span_coef[b, j] > 0) & mask_2d[b, j]
    #             if not pos_mask.any():
    #                 continue
                    
    #             # A(j): 유효한 위치의 샘플들에 대한 exp(sim)
    #             all_exp_sim = exp_sim[b, j]  # (N,)
                
    #             # 분모 계산: 패딩된 위치 제외하고 마스킹 적용
    #             masked_exp_sim = all_exp_sim * mask_2d[b, j]  # 현재 행에 대한 마스크 적용
    #             denominator = masked_exp_sim.sum() - masked_exp_sim[j]
                
    #             # positive pairs에 대한 loss 계산
    #             pos_indices = pos_mask.nonzero().squeeze(-1)
    #             pos_exp_sim = all_exp_sim[pos_indices]
    #             pos_coef = span_coef[b, j, pos_indices]
                
    #             log_probs = torch.log(pos_exp_sim / (denominator + 1e-10))
    #             weighted_log_probs = pos_coef * log_probs
    #             loss = -weighted_log_probs.sum() / (pos_coef.sum() + 1e-10)
                
    #             total_loss += loss
    #             valid_samples += 1
        
    #     if valid_samples == 0:
    #         return torch.tensor(0.0, device=device)
            
    #     return total_loss / valid_samples
    
    def compute_span_aware_contrastive_loss(
        self,
        sim_matrix: torch.Tensor,      # (B, layout_prompt_length, layout_prompt_length)
        span_coef_matrix: torch.Tensor, # (B, layout_prompt_length, layout_prompt_length)
    ) -> torch.Tensor:
        """Span-aware contrastive loss 계산 (논문 Equation 5)
        
        Args:
            sim_matrix: similarity matrix
            span_coef_matrix: coefficient matrix (-1로 패딩된 부분은 무시)
        """
        B, N, _ = sim_matrix.shape
        device = sim_matrix.device
        
        # Temperature scaling
        sim_matrix = sim_matrix / self.temperature
        
        # 수치 안정성을 위한 max normalization
        max_sim = torch.max(sim_matrix, dim=-1, keepdim=True)[0]
        sim_matrix = sim_matrix - max_sim
        exp_sim = torch.exp(sim_matrix)  # (B, N, N)
        
        total_loss = torch.tensor(0.0, device=device)
        span_coef_matrix = span_coef_matrix.to(device)
        valid_samples = 0
        
        for b in range(B):
            for j in range(N):
                # 현재 행에서 valid한 coefficient가 있는지 확인 (-1이 아닌 값)
                valid_coef_mask = span_coef_matrix[b, j] >= 0
                if not valid_coef_mask.any():
                    continue
                
                # P(j): positive samples (coefficient > 0)
                pos_mask = span_coef_matrix[b, j] > 0
                if not pos_mask.any():
                    continue
                
                # A(j): j를 제외한 모든 samples
                # 분모 계산: exp(sim) 합, 자기 자신 제외
                denominator = exp_sim[b, j].sum() - exp_sim[b, j, j]
                
                # Positive pairs에 대한 loss 계산
                pos_exp_sim = exp_sim[b, j][pos_mask]  # numerator
                pos_coef = span_coef_matrix[b, j][pos_mask]  # coefficients
                
                # log(exp(sim_p)/Σexp(sim_a))
                log_probs = torch.log(pos_exp_sim / (denominator + 1e-10))
                
                # coefficient로 가중치 부여
                loss = -(pos_coef * log_probs).sum() / (pos_coef.sum() + 1e-10)
                
                total_loss += loss
                valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=device)
        
        return total_loss / valid_samples

    def forward(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Batch inputs
        tag_targets: torch.Tensor = batch['token_ids']                 # (B, 688)
        box_indices: torch.Tensor = batch['box_indices']            # (B, N, M)
        data_tag_mask: torch.Tensor = batch['data_tag_mask']        # (B, 1376)
        attention_mask: torch.Tensor = batch['attention_mask']      # (B, 1376)
        
        # Model outputs
        tag_logits: torch.Tensor = outputs['tag_logits']            # (B, 688, V)
        box_proj: torch.Tensor = outputs['box_proj']                  # (B, :688, 1024)
        tag_proj: torch.Tensor = outputs['tag_proj']                  # (B, 688:, 1024)
        empty_proj: torch.Tensor = outputs['empty_proj']              # (B, 1, 1024)
        row_sim_matrix: torch.Tensor = outputs['row_sim_matrix']    # (B, 688, 688)
        col_sim_matrix: torch.Tensor = outputs['col_sim_matrix']    # (B, 688, 688)
        row_span_coef: torch.Tensor = outputs['row_span_coef']      # (B, 688, 688)
        col_span_coef: torch.Tensor = outputs['col_span_coef']      # (B, 688, 688)
        
        # Reshape tensors
        B, S, C = tag_logits.shape
        tag_logits = tag_logits.view(-1, C)      # (B*688, 9)
        tag_targets = tag_targets.view(-1)        # (B*688)
        valid_token_mask = ~torch.isin(tag_targets, torch.tensor(self.special_token_ids, device=tag_targets.device))
        
        # 유효한 토큰에 대해서만 loss 계산
        cls_loss = F.cross_entropy(
            tag_logits[valid_token_mask],     # (N, V) where N is number of valid tokens
            tag_targets[valid_token_mask],     # (N)
            reduction='mean',
            label_smoothing=0.1
        )
    
        # 2. Layout Pointer Loss (Equation 2)
        ptr_loss = self.compute_pointer_loss(
            box_features=box_proj,
            tag_features=tag_proj,
            box_indices=box_indices,
            data_tag_mask=data_tag_mask,
        )
        
        # 3. Empty Pointer Loss (Equation 3)
        empty_ptr_loss = self.compute_empty_pointer_loss(
            empty_embedding=empty_proj,
            tag_features=tag_proj,
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