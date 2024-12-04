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
        
    def ____compute_pointer_loss(self, box_features, tag_features, box_indices, data_tag_mask):
        B, max_N, D = box_features.shape
        L = tag_features.size(1)
        
        total_loss = 0
        valid_count = 0
        
        for b in range(B):
            # 현재 배치의 실제 box 수 계산 (non-padding)
            N = (box_indices[b, :, 0] != -1).sum()
            
            # 현재 배치의 data_tag_mask 자르기
            batch_data_tag_mask = data_tag_mask[b, N:N+L]  # 현재 배치의 실제 N 사용
            
            # 현재 배치의 relative indices 계산
            batch_indices = box_indices[b].clone()
            batch_indices[batch_indices != -1] -= N  # 현재 배치의 실제 N으로 상대 위치 계산
            
            for n in range(N):  # 실제 box 수만큼만 반복
                if not (box_indices[b, n] != -1).any():
                    continue
                    
                box_feat = box_features[b, n]
                logits = torch.matmul(box_feat, tag_features[b].t())
                logits = logits / self.temperature
                
                exp_logits = torch.exp(logits)
                exp_logits = exp_logits.masked_fill(~batch_data_tag_mask, 0.0)
                
                for m in range(batch_indices.size(1)):
                    target_idx = batch_indices[n, m]
                    if target_idx == -1:
                        break
                        
                    numerator = exp_logits[target_idx]
                    denominator = exp_logits.sum()
                    
                    loss = -torch.log(numerator / (denominator + 1e-10))
                    total_loss += loss
                    valid_count += 1
        
        return total_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=total_loss.device)
    
    def compute_pointer_loss(self, box_features, tag_features, box_indices, data_tag_mask):
        """Layout pointer loss 계산 (논문 Equation 2)
        
        Args:
            box_features: (B, N, D) - projected box features
            tag_features: (B, T, D) - projected tag features
            box_indices: (B, N, M) - 각 box가 가리키는 실제 sequence 내 위치
            data_tag_mask: (B, L) - text가 있는 tag 위치
        """
        B, N, D = box_features.shape
        L = tag_features.size(1)
        
        # print(f"\nInitial shapes:")
        # print(f"box_features: {box_features.shape}")
        # print(f"tag_features: {tag_features.shape}")
        # print(f"box_indices: {box_indices.shape}")
        # print(f"data_tag_mask: {data_tag_mask.shape}")
        # print(f"N: {N}, L: {L}")
        
        # data_tag_mask 변형 전후 비교
        # print(f"\ndata_tag_mask before:")
        # print(f"True positions: {torch.where(data_tag_mask[0])[0]}")
        data_tag_mask = data_tag_mask[:, N:N+L]
        # print(f"data_tag_mask after:")
        # print(f"True positions: {torch.where(data_tag_mask[0])[0]}")
        
        # relative_indices 변환 과정
        relative_indices = box_indices.clone()
        # print(f"\nbox_indices before:")
        # print(box_indices[0, :10])  # 첫 10개만
        relative_indices[relative_indices != -1] -= N
        # print(f"relative_indices after:")
        # print(relative_indices[0, :10])  # 첫 10개만
            
        total_loss = 0
        valid_count = 0
        
        for b in range(B):
            # print(f"\nBatch {b}:")
            true_positions = torch.where(data_tag_mask[b])[0]
            # print(f"True positions in data_tag_mask: {true_positions}")
            
            for n in range(N):
                if not (box_indices[b, n] != -1).any():
                    continue
                    
                # print(f"\nBox {n}:")
                # valid_targets = box_indices[b, n][box_indices[b, n] != -1]
                # print(f"Valid targets: {valid_targets}")
                
                box_feat = box_features[b, n]  # (D,)
                
                # 모든 tag와의 유사도 계산
                logits = torch.matmul(box_feat, tag_features[b].t())  # (L,)
                # print(f"logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                
                # temperature scaling 적용
                logits = logits / self.temperature
                # print(f"logits range after temperature scaling: [{logits.min():.4f}, {logits.max():.4f}]")
                
                # exp 계산 및 masking - text가 있는 cell(D)에 대해서만
                exp_logits = torch.exp(logits)  # (L,)
                # print(f"exp_logits shape: {exp_logits.shape}, data_tag_mask shape: {data_tag_mask.shape}")
                exp_logits = exp_logits.masked_fill(~data_tag_mask[b], 0.0)  # D에 속하지 않는 위치는 0
                # print(f"exp_logits range: [{exp_logits.min():.4f}, {exp_logits.max():.4f}]")
                
                # 각 매핑에 대해 loss 계산 (many-to-one 지원)
                # for m in range(box_indices.size(2)):
                    # target_idx = box_indices[b, n, m]
                for m in range(relative_indices.size(2)):
                    target_idx = relative_indices[b, n, m]  # 이미 상대적 위치로 변환됨
                    if target_idx == -1:  # 더 이상 매핑 없음
                        break
                    
                    # print(f"target_idx: {target_idx}")
                    # print(f"data_tag_mask at target: {data_tag_mask[b, target_idx]}")
                    # print(f"logits at target: {logits[target_idx]}")
                    
                    numerator = exp_logits[target_idx]
                    # print(f"numerator: {numerator}")
                    denominator = exp_logits.sum()
                    # print(f"denominator: {denominator}")
                    
                    # loss 계산: -log(numerator/denominator)
                    loss = -torch.log(numerator / (denominator + 1e-10))
                    # print(f"loss: {loss}")
                    
                    total_loss += loss
                    valid_count += 1
        
        # -1/B * Σ 계산
        return total_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=total_loss.device)

    def compute_empty_pointer_loss(
        self,
        empty_embedding: torch.Tensor,  # (1, 1, D)
        tag_features: torch.Tensor,     # (B, T, D)
        box_indices: torch.Tensor,      # (B, N, M)
        data_tag_mask: torch.Tensor     # (B, N+L)
    ):
        B, T, D = tag_features.shape
        N = box_indices.size(1)
        L = tag_features.size(1)
        data_tag_mask = data_tag_mask[:, N:N+L]  # (B, L)
        
        total_loss = 0
        total_tags = 0
        
        # 각 batch와 data tag에 대해 개별적으로 계산
        for b in range(B):
            for t in range(T):
                if not data_tag_mask[b, t]:  # data tag가 아닌 경우 skip
                    continue
                    
                # 현재 tag의 feature와 empty embedding의 내적
                logit = torch.matmul(
                    tag_features[b, t].unsqueeze(0),  # (1, D)
                    empty_embedding.squeeze(0).t()     # (D, 1)
                ).squeeze()  # scalar
                
                # target: box_indices에서 참조되지 않은 data tag는 empty
                is_empty = not any((box_indices[b, :, 0] == t).any() for n in range(N))
                target = torch.tensor([is_empty], dtype=torch.float, device=logit.device)
                
                # BCE loss 계산
                loss = F.binary_cross_entropy_with_logits(
                    logit.unsqueeze(0),
                    target
                )
                
                total_loss += loss
                total_tags += 1
        
        return total_loss / total_tags if total_tags > 0 else torch.tensor(0.0, device=total_loss.device)

    # def compute_pointer_loss(
    #     self,
    #     pointer_logits: torch.Tensor,  # (B, N, L) - 이미 temperature로 나눠진 값
    #     box_indices: torch.Tensor,     # (B, N, M)
    #     data_tag_mask: torch.Tensor,   # (B, N+L)
    # ) -> torch.Tensor:
    #     # Lptr = -1/B * Σ((b̄j·t̄k*/τ) - log(Σk'∈D exp(b̄j·t̄k'/τ)))
    #     B, N, L = pointer_logits.shape
        
    #     # data tag 위치만 남기기 (set D)
    #     data_tag_mask = data_tag_mask[:, N:N+L]  # (B, L)
        
    #     # valid boxes에 대해서만 loss 계산
    #     valid_boxes = (box_indices != -1).any(dim=-1)  # (B, N)
    #     target_indices = box_indices[:, :, 0]  # k* indices
        
    #     # 첫 번째 항: b̄j·t̄k*/τ (이미 temperature로 나눠진 값)
    #     batch_idx = torch.arange(B, device=pointer_logits.device)[:, None]
    #     box_idx = torch.arange(N, device=pointer_logits.device)[None, :]
    #     first_term = pointer_logits[batch_idx, box_idx, target_indices]  # (B, N)
    #     # print(f"first_term range: [{first_term.min():.4f}, {first_term.max():.4f}]")
        
    #     # 두 번째 항: log(Σk'∈D exp(b̄j·t̄k'/τ))
    #     # LogSumExp trick 사용
    #     max_logits = pointer_logits.max(dim=-1, keepdim=True)[0]  # (B, N, 1)
    #     # print(f"max_logits range: [{max_logits.min():.4f}, {max_logits.max():.4f}]")
    #     stable_logits = pointer_logits - max_logits
    #     # print(f"stable_logits range: [{stable_logits.min():.4f}, {stable_logits.max():.4f}]")
        
    #     # data tag가 아닌 위치는 masking
    #     exp_logits = torch.exp(stable_logits)
    #     exp_logits = exp_logits.masked_fill(~data_tag_mask.unsqueeze(1), 0.0)
    #     # print(f"exp_logits range: [{exp_logits.min():.4f}, {exp_logits.max():.4f}]")    
        
    #     # sum over k'∈D
    #     second_term = torch.log(exp_logits.sum(dim=-1) + 1e-10) + max_logits.squeeze(-1)  # (B, N)
    #     # print(f"second_term range: [{second_term.min():.4f}, {second_term.max():.4f}]")
        
    #     # loss 계산: -1/B * Σ(first_term - second_term)
    #     loss = -(first_term - second_term)  # negative log-likelihood
    #     # print(f"loss range: [{loss.min():.4f}, {loss.max():.4f}]")
        
    #     # valid boxes에 대해서만 평균
    #     loss = loss[valid_boxes]
    #     # print(f"valid_loss range: [{loss.min():.4f}, {loss.max():.4f}]")
    #     loss = loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=loss.device)
    #     # print(f"final_loss: {loss.mean()}")
        
    #     return loss

    # def compute_empty_pointer_loss(
    #     self,
    #     empty_logits: torch.Tensor,  # (B, L)
    #     box_indices: torch.Tensor,   # (B, N, M)
    #     data_tag_mask: torch.Tensor  # (B, N+L)
    # ) -> torch.Tensor:
    #     """
    #     Empty pointer loss 계산 (논문 Equation 3)
        
    #     Lempty_ptr = -1/|D| * Σk'∈D(BCE(σ(b̄0·t̄k'), I(k')))
        
    #     Args:
    #         empty_logits: Empty cell detection logits (σ(b̄0·t̄k'))
    #         box_indices: Box-tag mapping indices
    #         data_tag_mask: Data tag positions (set D)
    #     """
    #     B, L = empty_logits.shape
    #     N = box_indices.size(1)
        
    #     # # print(f"\nEmpty pointer loss:")
    #     # # print(f"  empty_logits shape: {empty_logits.shape}")
    #     # # print(f"  box_indices shape: {box_indices.shape}")
    #     # # print(f"  data_tag_mask shape: {data_tag_mask.shape}")
    #     # # print(f"  N: {N}, L: {L}")
    
       
        
    #     # 1. Get data tag mask (set D)
    #     data_tag_mask = data_tag_mask[:, N:N+L]  # (B, L) layout prompt 제외
    #     # # print(f"  sliced data_tag_mask shape: {data_tag_mask.shape}")
    #     # 2. Find referenced positions
    #     referenced_positions = torch.zeros((B, L), dtype=torch.bool, device=empty_logits.device)
    #     valid_indices = (box_indices != -1)
    #     flat_indices = box_indices[valid_indices].long()
    #     batch_indices = torch.arange(B, device=box_indices.device).view(-1, 1, 1).expand_as(box_indices)[valid_indices]
    #     referenced_positions[batch_indices, flat_indices] = True
        
    #     # # print(f"  referenced_positions shape: {referenced_positions.shape}")
    #     # 3. Target: True for empty positions among data tags (I(k'))
    #     target = ~referenced_positions & data_tag_mask
        
    #     # 4. Compute loss only for data tags (-1/|D| * Σk'∈D)
    #     loss = F.binary_cross_entropy_with_logits(
    #         empty_logits[data_tag_mask],
    #         target[data_tag_mask].float(),
    #         reduction='mean'  # -1/|D| 구현
    #     )
        
    #     return loss

    def compute_span_aware_contrastive_loss(
        self,
        sim_matrix: torch.Tensor,  # (B, N, N)
        span_coef: torch.Tensor,   # (B, N, N)
    ) -> torch.Tensor:
        """
        Span-aware contrastive loss 계산 (논문 Equation 5)
        
        Lcontr,j = -1/cp(j) * Σ(cp(j) * log(exp(b̂j·b̂p/τ)/Σ(exp(b̂j·b̂a/τ))))
        """
        B, N, _ = sim_matrix.shape
        
        # Normalize similarity matrix with temperature
        sim_matrix = sim_matrix / self.temperature
        
        # Mask for valid positive samples (B, N)
        valid_pos_mask = (span_coef > 0).any(dim=-1)  # (B, N)
        
        # Calculate numerator with span coefficients
        pos_sim = sim_matrix * span_coef  # (B, N, N)
        numerator = pos_sim.sum(dim=-1)  # (B, N)
        
        # Calculate denominator with numerical stability
        max_sim = sim_matrix.max(dim=-1, keepdim=True)[0]  # (B, N, 1)
        stable_sim = sim_matrix - max_sim
        exp_sim = stable_sim.exp()  # (B, N, N)
        denominator = exp_sim.sum(dim=-1)  # (B, N)
        
        # Compute loss with span coefficients
        box_loss = -numerator + torch.log(denominator.clamp(min=1e-10)) + max_sim.squeeze(-1)
        
        # Average over valid positions only
        masked_loss = box_loss * valid_pos_mask.float()
        total_loss = masked_loss.sum()
        valid_count = valid_pos_mask.sum().clamp(min=1)
        
        return total_loss / valid_count
    
    def compute_cls_loss(
        self,
        logits: torch.Tensor,      # (B*L, V)
        targets: torch.Tensor,     # (B*L)
        valid_mask: torch.Tensor,  # (B*L)
    ) -> torch.Tensor:
        """Tag classification loss (논문 Section 3.4)
        
        Cross-entropy loss를 사용하여 decoder의 tag classification을 학습
        """
        loss = F.cross_entropy(
            logits[valid_mask],
            targets[valid_mask],
            reduction='mean'
        )
        return loss
    
    def compute_focal_loss(
        self,
        logits: torch.Tensor,      # (B*L, V)
        targets: torch.Tensor,     # (B*L)
        valid_mask: torch.Tensor,  # (B*L)
    ) -> torch.Tensor:
        """
        Compute Focal Loss with class weights
        
        FL(pt) = -α(1-pt)^γ * log(pt)
        where pt is the predicted probability of the target class
        """
        # Get predicted probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (B*L, V)
        probs = torch.exp(log_probs)  # (B*L, V)
        
        # Get target probabilities
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B*L)
        
        # Compute focal weights
        focal_weights = (1 - target_probs) ** self.focal_gamma
        
        # Apply class weights if available
        if self.tag_weights is not None:
            class_weights = self.tag_weights[targets]  # (B*L)
            focal_weights = focal_weights * class_weights
        
        # Compute focal loss
        focal_loss = -self.focal_alpha * focal_weights * log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply valid mask and average
        focal_loss = (focal_loss * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)
        
        return focal_loss

    def forward(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Batch inputs
        tag_targets: torch.Tensor = batch['tokens']                 # (B, L)
        box_indices: torch.Tensor = batch['box_indices']            # (B, N, M)
        data_tag_mask: torch.Tensor = batch['data_tag_mask']        # (B, N+L)
        
        # Model outputs
        tag_logits: torch.Tensor = outputs['tag_logits']            # (B, L, V)
        box_proj: torch.Tensor = outputs['box_proj']                  # (B, N, D)
        tag_proj: torch.Tensor = outputs['tag_proj']                  # (B, T, D)
        empty_proj: torch.Tensor = outputs['empty_proj']              # (B, D)
        # pointer_logits: torch.Tensor = outputs['pointer_logits']    # (B, N, T)
        # empty_logits: torch.Tensor = outputs['empty_logits']        # (B, T)
        row_sim_matrix: torch.Tensor = outputs['row_sim_matrix']    # (B, N, N)
        col_sim_matrix: torch.Tensor = outputs['col_sim_matrix']    # (B, N, N)
        row_span_coef: torch.Tensor = outputs['row_span_coef']      # (B, N, N)
        col_span_coef: torch.Tensor = outputs['col_span_coef']      # (B, N, N)
        
        # Layout prompt 길이
        N = box_indices.size(1)
        B, L, V = tag_logits.shape
        # 1. Classification Loss with Focal Loss
        # tag_targets를 tag_logits의 길이에 맞춤
        tag_targets = tag_targets[:, :L]  # 추가된 부분
        
        
        # # print(f"\nLoss forward:")
        # # print(f"  tag_logits shape: {tag_logits.shape}")
        # # print(f"  tag_targets shape: {tag_targets.shape}")
        # # print(f"  B: {B}, L: {L}, V: {V}")
        # # print(f"  Attempting to reshape to: {B*L}")
        
        tag_logits = tag_logits.reshape(B*L, V)
        tag_targets = tag_targets.reshape(B*L)
        
        # Mask out special tokens
        valid_token_mask = ~torch.isin(tag_targets, torch.tensor(self.special_token_ids, device=tag_targets.device))
        
        # Update tag weights
        if self.tag_weights is None or self.tag_weights.device != tag_targets.device:
            self.update_tag_weights(tag_targets)
        
        # Compute focal loss
        cls_loss = self.compute_cls_loss(
            logits=tag_logits,
            targets=tag_targets,
            valid_mask=valid_token_mask
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
            box_indices=box_indices,
            data_tag_mask=data_tag_mask,
        )
        
        # 4. Span-aware Contrastive Loss (Equation 5)
        row_contr_loss = torch.tensor(0.0, device=tag_logits.device)
        col_contr_loss = torch.tensor(0.0, device=tag_logits.device)
        
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