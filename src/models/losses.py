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
        box_indices: torch.Tensor,     # (B, N, M)
        data_tag_mask: torch.Tensor,   # (B, T)
    ) -> torch.Tensor:
        """Layout pointer loss (논문 Equation 2)
        Many-to-one 관계를 지원하도록 수정된 버전
        각 box는 여러 개의 tag를 가리킬 수 있음
        """
        # 1. data tag positions에 대해서만 logits 계산 (k'∈D)
        pointer_logits = pointer_logits.masked_fill(~data_tag_mask.unsqueeze(1), -float('inf'))
        
        # 2. valid boxes와 mappings에 대해서만 loss 계산
        valid_box_mask = (box_indices != -1)  # (B, N, M)
        
        # 3. log softmax로 한번에 계산 (numerical stability)
        log_probs = F.log_softmax(pointer_logits, dim=-1)  # (B, N, T)
        
        total_loss = torch.tensor(0.0, device=pointer_logits.device)
        total_mappings = 0
        
        # 4. 각 batch, box, mapping에 대해 loss 계산
        for b in range(pointer_logits.size(0)):
            for j in range(pointer_logits.size(1)):
                valid_mappings = box_indices[b, j][valid_box_mask[b, j]]  # 유효한 매핑만 선택
                if len(valid_mappings) > 0:
                    # 각 매핑에 대한 loss 계산
                    for k_star in valid_mappings:
                        total_loss -= log_probs[b, j, k_star]  # negative log likelihood
                        total_mappings += 1
        
        # 5. 전체 매핑 수로 평균
        return total_loss / max(total_mappings, 1)

    def compute_empty_pointer_loss(
        self,
        empty_logits: torch.Tensor,    # (B, T)
        box_indices: torch.Tensor,     # (B, N, M)
        data_tag_mask: torch.Tensor    # (B, T)
    ) -> torch.Tensor:
        """Empty Pointer Loss (논문 Equation 3)
        Many-to-one 관계를 지원하도록 수정된 버전
        """
        # box_indices로부터 empty cell label 생성
        empty_label = torch.ones_like(data_tag_mask, dtype=torch.float32)  # 기본값 1 (empty)
        
        # 각 batch에 대해 처리
        for i in range(box_indices.size(0)):
            # 모든 유효한 매핑 위치를 찾음 (-1이 아닌 모든 위치)
            valid_mappings = box_indices[i][box_indices[i] != -1]
            if len(valid_mappings) > 0:
                empty_label[i][valid_mappings] = 0  # non-empty 위치는 0으로 설정
        
        # data tag positions에서만 loss 계산
        loss = F.binary_cross_entropy_with_logits(
            empty_logits.clamp(min=-100, max=100),  # clipping 추가
            empty_label,
            reduction='none'
        )
        
        # data tag positions의 평균만 계산
        valid_positions = data_tag_mask.float()
        num_valid = valid_positions.sum().clamp(min=1)
        loss = (loss * valid_positions).sum() / num_valid
        
        return loss
    

    def compute_span_aware_contrastive_loss(
        self, 
        sim_matrix: torch.Tensor,  # (B, N, N)
        span_coef: torch.Tensor,   # (B, N, N)
    ) -> torch.Tensor:
        batch_size, num_boxes = sim_matrix.size(0), sim_matrix.size(1)
        
        span_coef = span_coef.to(sim_matrix.device)
        sim_matrix = sim_matrix / self.temperature  # (B, N, N)
        
        diag_mask = torch.eye(num_boxes, dtype=torch.bool, device=sim_matrix.device).unsqueeze(0)
        
        total_loss = torch.tensor(0.0, device=sim_matrix.device)
        valid_count = 0
        
        # 배치 단위로는 유지하고 bbox 연산만 벡터화
        for b in range(batch_size):
            curr_span_coef = span_coef[b]  # (N, N)
            curr_sim = sim_matrix[b]  # (N, N)
            
            # Positive/Negative mask
            positive_mask = (curr_span_coef > 0) & ~diag_mask[0]  # (N, N)
            negative_mask = ~positive_mask & ~diag_mask[0]  # (N, N)
            
            # 각 bbox의 cp_sum 계산 (N,)
            cp_sums = (curr_span_coef * positive_mask.float()).sum(dim=1)  # (N,)
            valid_boxes = cp_sums > 0  # (N,)
            
            if not valid_boxes.any():
                continue
                
            # 유효한 bbox에 대해서만 계산
            valid_indices = torch.where(valid_boxes)[0]
            
            for j in valid_indices:
                pos_mask_j = positive_mask[j]  # (N,)
                neg_mask_j = negative_mask[j]  # (N,)
                
                if not (pos_mask_j.any() and neg_mask_j.any()):
                    continue
                    
                pos_sim = curr_sim[j][pos_mask_j]  # (P,)
                neg_sim = curr_sim[j][neg_mask_j]  # (N,)
                pos_coef = curr_span_coef[j][pos_mask_j]  # (P,)
                
                # Numerically stable 계산
                max_val = torch.max(torch.cat([pos_sim, neg_sim]))
                
                # logsumexp 사용하여 numerical stability 향상
                pos_term = torch.logsumexp(pos_sim - max_val, dim=0) + torch.log(pos_coef.sum())
                neg_term = torch.logsumexp(neg_sim - max_val, dim=0)
                
                # Loss 계산 및 clipping
                loss_j = torch.clamp(-(pos_term - neg_term) / cp_sums[j], min=-100, max=100)
                total_loss = total_loss + loss_j
                valid_count += 1
        
        # 실제 계산된 valid한 bbox 수로 normalize
        return total_loss / max(valid_count, 1)

    def forward(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Batch inputs
        tag_targets: torch.Tensor = batch['tokens']                 # (B, L)
        box_indices: torch.Tensor = batch['box_indices']            # (B, N, M)
        data_tag_mask: torch.Tensor = batch['data_tag_mask']       # (B, T)
        
        # Model outputs
        tag_logits: torch.Tensor = outputs['tag_logits']            # (B, L, V)
        pointer_logits: torch.Tensor = outputs['pointer_logits']    # (B, N, T)
        empty_logits: torch.Tensor = outputs['empty_logits']        # (B, T)
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
            box_indices=box_indices,        # (B, N, M)
            data_tag_mask=data_tag_mask,    # (B, T)
        )
        
        # 3. Empty Pointer Loss
        empty_ptr_loss = self.compute_empty_pointer_loss(
            empty_logits=empty_logits,
            box_indices=box_indices,
            data_tag_mask=data_tag_mask,
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