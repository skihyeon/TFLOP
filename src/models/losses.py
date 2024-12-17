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
        """Layout pointer loss for non-empty cells (Equation 2)"""
        B, num_boxes, seq_len = pointer_logits.shape
        
        # Temperature scaling
        pointer_logits = pointer_logits / temperature
        
        # 1. Get target positions for each box
        valid_boxes = box_indices != -1  # (B, num_boxes, max_mappings)
        valid_box_mask = valid_boxes.any(dim=-1)  # (B, num_boxes)
        
        # 2. Get valid C tag positions (OTSL sequence 내에서의 상대적 위치 사용)
        valid_tag_positions = data_tag_mask[:, self.layout_prompt_length:]  # (B, 30)
        
        # 3. Compute loss for each valid box
        box_losses = torch.zeros_like(valid_box_mask, dtype=torch.float, device=pointer_logits.device)
        
        # print("\n[Pointer Loss Debug]")
        for b in range(B):
            for j in range(num_boxes):
                if valid_box_mask[b, j]:
                    # Get all target positions for this box (k*)
                    target_positions = box_indices[b, j][valid_boxes[b, j]]  # 여러 target positions
                    
                    # Compute log probabilities over valid positions only
                    log_probs = F.log_softmax(pointer_logits[b, j], dim=-1)
                    
                    # Average log probs across all target positions
                    box_losses[b, j] = -log_probs[target_positions].mean()
                    
                    # print(f"Batch {b}, Box {j}:")
                    # print(f"  Target positions: {target_positions.tolist()}")
                    # print(f"  Valid positions: {valid_tag_positions[b].nonzero().squeeze(-1).tolist()}")
                    # print(f"  Loss: {box_losses[b, j].item():.4f}")
        
        # 4. Average over valid boxes (excluding inf losses)
        valid_losses = box_losses[valid_box_mask & ~torch.isinf(box_losses)]
        final_loss = valid_losses.mean() if len(valid_losses) > 0 else torch.tensor(0.0, device=pointer_logits.device)
        
        # print(f"\nValid losses: {len(valid_losses)}/{valid_box_mask.sum().item()}")
        # print(f"Final pointer loss: {final_loss.item():.4f}")
        
        return final_loss

    def compute_empty_pointer_loss(self, empty_logits: torch.Tensor,
                             data_tag_mask: torch.Tensor,
                             empty_tag_mask: torch.Tensor) -> torch.Tensor:
        """Empty pointer loss (Equation 3)
        
        Args:
            empty_logits: (B, 1, 30) - raw logits
            data_tag_mask: (B, 1024) - non-empty C 태그 위치
            empty_tag_mask: (B, 1024) - empty C 태그 위치
        """
        # Ensure correct shape
        if empty_logits.dim() == 3:
            empty_logits = empty_logits.squeeze(1)  # (B, 30)
        
        # 1. OTSL sequence 부분만 선택
        otsl_mask = (data_tag_mask | empty_tag_mask)[:, self.layout_prompt_length:]  # (B, 30) - 모든 C 태그
        targets = empty_tag_mask[:, self.layout_prompt_length:]  # (B, 30) - empty C 태그
        
        # 2. Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            empty_logits,  # (B, 30)
            targets.float(),  # (B, 30)
            reduction='none'  # (B, 30)
        )
        
        # 3. Apply mask and compute mean
        masked_loss = bce_loss * otsl_mask  # only consider C tag positions
        num_c_tags = otsl_mask.sum(dim=1, keepdim=True) + 1e-8
        batch_losses = masked_loss.sum(dim=1) / num_c_tags.squeeze()
        
        # Debug information
        # print("\n[Empty Pointer Loss Debug]")
        # print(f"Total samples: {len(batch_losses)}")
        # for b in range(len(batch_losses)):
            # n_total = otsl_mask[b].sum().item()
            # n_empty = targets[b].sum().item()
            # print(f"\nBatch {b}:")
            # print(f"  Total C tags: {n_total}")
            # print(f"  Empty C tags: {n_empty}")
            # print(f"  Empty ratio: {n_empty/n_total:.2f}")
            # print(f"  Loss: {batch_losses[b].item():.4f}")
            
            # Show predictions vs targets
            # probs = torch.sigmoid(empty_logits[b])
            # mask = otsl_mask[b]
            # print("\n  C tag positions (prob -> target):")
            # for pos in mask.nonzero().squeeze(-1):
                # print(f"    Pos {pos}: {probs[pos]:.3f} -> {targets[b, pos].item()}")
        
        final_loss = batch_losses.mean()
        # print(f"\nFinal loss: {final_loss.item():.4f}")
        return final_loss
    
    def compute_span_aware_contrastive_loss(
            self,
            sim_matrix: torch.Tensor,  # (B, N, N)
            span_coef_matrix: torch.Tensor,  # (B, N, N)
        ) -> torch.Tensor:
            device = sim_matrix.device
            span_coef_matrix = span_coef_matrix.to(device)
            
            # Temperature scaling
            sim_matrix = sim_matrix / self.temperature  # (B, N, N)
            
            # 1. Compute exp(sim_matrix/τ) for all pairs
            exp_sim = torch.exp(sim_matrix)  # (B, N, N)
            
            # 2. Compute denominator (sum over all pairs)
            denominator = exp_sim.sum(dim=-1, keepdim=True)  # (B, N, 1)
            
            # 3. Compute log probabilities
            log_probs = torch.log(exp_sim / denominator + 1e-8)  # (B, N, N)
            
            # 4. Mask for valid pairs (coef >= 0)
            valid_mask = (span_coef_matrix >= 0)  # (B, N, N)
            
            # 5. Compute weighted sum of log probs for valid pairs
            weighted_log_probs = span_coef_matrix * log_probs * valid_mask
            
            # 6. Normalize by sum of coefficients and compute mean
            coef_sum = span_coef_matrix.sum(dim=-1) + 1e-8  # (B, N)
            box_losses = -(weighted_log_probs.sum(dim=-1) / coef_sum)  # (B, N)
            
            # 7. Mask out invalid boxes (no positive pairs)
            valid_boxes = valid_mask.any(dim=-1)  # (B, N)
            final_loss = (box_losses * valid_boxes).sum() / (valid_boxes.sum() + 1e-8)
            
            return final_loss
            
    def compute_tag_loss(self, tag_logits, tag_targets):
        """
        Args:
            tag_logits: (B, S, V)
            tag_targets: (B, V)
        """
        # print(f"tag_logits shape: {tag_logits.shape}")
        # print(f"tag_targets shape: {tag_targets.shape}")
        # print(f"tag_logits stats: min={tag_logits.min():.3f}, max={tag_logits.max():.3f}")
        # print(f"unique targets: {torch.unique(tag_targets).tolist()}")
    
        return F.cross_entropy(
            tag_logits.reshape(-1, self.tokenizer.vocab_size),  # (B*S, V)
            tag_targets.reshape(-1),  # (B*S)
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=0.1
        )

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
            temperature=self.temperature,
        )

        # 3. Empty Pointer Loss (Equation 3)
        empty_ptr_loss = self.compute_empty_pointer_loss(
            empty_logits=empty_pointer_logits,
            data_tag_mask=batch['data_tag_mask'],
            empty_tag_mask=batch['empty_tag_mask']
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
