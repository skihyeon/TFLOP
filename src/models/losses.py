import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math

class TFLOPLoss(nn.Module):
    """
    TFLOP Loss Function (논문 Section 3.7)
    
    L = λ1*Lcls + λ2*Lptr + λ3*Lempty_ptr + λ4*Σ(Lrow_contr)/B + λ5*Σ(Lcol_contr)/B
    """
    def __init__(
        self,
        temperature: float = 0.1,
        lambda_cls: float = 1.0,
        lambda_ptr: float = 1.0,
        lambda_empty_ptr: float = 1.0,
        lambda_row_contr: float = 0.5,
        lambda_col_contr: float = 0.5,
        feature_dim: int = 768,
        pad_token_id: int = 1
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_cls = lambda_cls
        self.lambda_ptr = lambda_ptr
        self.lambda_empty_ptr = lambda_empty_ptr
        self.lambda_row_contr = lambda_row_contr
        self.lambda_col_contr = lambda_col_contr
        self.pad_token_id = pad_token_id
        
        # Feature projections (논문 Equation 1, 4)
        self.box_proj = nn.Linear(feature_dim, feature_dim)  # projb
        self.tag_proj = nn.Linear(feature_dim, feature_dim)  # projt
        self.span_proj = nn.Linear(feature_dim, feature_dim)  # projs
        
        # Empty box embedding (논문 Equation 3)
        self.empty_embedding = nn.Parameter(torch.zeros(1, feature_dim))
        
    def compute_pointer_loss(
        self,
        box_features: torch.Tensor,     # (B, N, D)
        tag_features: torch.Tensor,     # (B, T, D)
        box_indices: torch.Tensor,      # (B, N)
        data_tag_mask: torch.Tensor     # (B, T)
    ) -> torch.Tensor:
        """Layout Pointer Loss (논문 Equation 2)
        Lptr = -1/B * Σ(j=1 to B) log(exp(b̄j·t̄k*)/Σ(k'∈D)exp(b̄j·t̄k'/τ))
        """
        B, N, D = box_features.size()
        _, T, _ = tag_features.size()
        
        # Project features (Equation 1)
        box_proj = self.box_proj(box_features)  # b̄j (B, N, D)
        tag_proj = self.tag_proj(tag_features)  # t̄k (B, T, D)
        
        # Normalize projections for numerical stability
        box_proj = F.normalize(box_proj, dim=-1)
        tag_proj = F.normalize(tag_proj, dim=-1)
        
        # Compute similarity scores
        sim = torch.matmul(box_proj, tag_proj.transpose(-2, -1))  # (B, N, T)
        sim = sim / self.temperature
        
        # Mask for data tags only (k'∈D)
        mask = data_tag_mask.unsqueeze(1).expand(-1, N, -1)  # (B, N, T)
        sim = sim.masked_fill(~mask, -1e9)  # float('-inf') 대신 큰 음수 사용
        
        # Compute log probabilities with stable softmax
        log_probs = F.log_softmax(sim, dim=-1)  # (B, N, T)
        
        # box_indices를 N개의 box에 맞게 자르기
        box_indices = box_indices[:, :N]  # (B, N)
        
        # Compute loss for each box-tag pair
        valid_mask = box_indices != self.pad_token_id  # (B, N)
        if valid_mask.any():
            # Gather target probabilities
            target_probs = torch.gather(
                log_probs,  # (B, N, T)
                dim=-1,     # T 차원에서 gather
                index=box_indices.unsqueeze(-1).clamp(0, T-1)  # index가 범위를 벗어나지 않도록 clamp
            ).squeeze(-1)  # (B, N)
            
            # Compute mean loss over valid boxes
            loss = -target_probs[valid_mask].mean()
            
            # Loss가 너무 크지 않도록 제한
            loss = torch.clamp(loss, min=0.0, max=10.0)
        else:
            loss = torch.tensor(0.0, device=log_probs.device)
        
        return loss

    def compute_empty_pointer_loss(
        self,
        tag_features: torch.Tensor,    # (B, T, D)
        empty_mask: torch.Tensor       # (B, T)
    ) -> torch.Tensor:
        """Empty Pointer Loss (논문 Equation 3)
        Lempty_ptr = -1/|D| * Σ(k'∈D) BCE(σ(b̄0·t̄k'), I(k'))
        """
        B, T, D = tag_features.size()
        
        # Project features
        empty_proj = self.box_proj(self.empty_embedding)  # b̄0 (1, D)
        tag_proj = self.tag_proj(tag_features)  # t̄k' (B, T, D)
        
        # Compute similarity scores
        logits = torch.matmul(tag_proj, empty_proj.transpose(0, 1))  # (B, T, 1)
        logits = logits.squeeze(-1) / self.temperature  # (B, T)
        
        # Numerical stability를 위해 logits clamp
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        
        # Binary cross entropy loss
        num_data_tags = empty_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # |D|
        loss = F.binary_cross_entropy_with_logits(
            logits,
            empty_mask.float(),
            reduction='none'
        )
        
        # Average over data tags only
        loss = (loss * empty_mask.float()).sum(dim=-1) / num_data_tags
        loss = loss.mean()
        
        return loss

    def compute_span_aware_contrastive_loss(
        self,
        features: torch.Tensor,     # (B, N, D)
        spans: torch.Tensor,        # (B, N, N)
        direction: str = 'row'
    ) -> torch.Tensor:
        """Span-aware Contrastive Loss (논문 Equation 4, 5, 6)
        Lcontr,j = -1/cp(j) * Σ(p∈P(j)) cp(j)log(exp(b̂j·b̂p/τ)/Σ(a∈A(j))exp(b̂j·b̂a/τ))
        """
        B, N, D = features.size()
        
        # Project features (Equation 4)
        proj_features = self.span_proj(features)  # b̂j
        proj_features = F.normalize(proj_features, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(proj_features, proj_features.transpose(-2, -1)) / self.temperature  # (B, N, N)
        
        # Compute span coefficients (Equation 6)
        span_overlap = torch.matmul(spans, spans.transpose(-2, -1))  # (B, N, N)
        span_sizes = torch.sum(spans, dim=-1, keepdim=True)  # (B, N, 1)
        span_coef = span_overlap / (span_sizes * span_sizes.transpose(-2, -1) + 1e-8)  # cp(j)
        
        # Mask for valid pairs (exclude self)
        mask = ~torch.eye(N, dtype=torch.bool, device=features.device).unsqueeze(0)  # (1, N, N)
        mask = mask.expand(B, -1, -1)  # (B, N, N)
        
        # Compute positive and negative masks
        positive_mask = (span_coef > 0) & mask  # P(j)
        negative_mask = mask & ~positive_mask   # A(j)
        
        # Compute log probabilities
        exp_sim = torch.exp(sim_matrix)  # (B, N, N)
        
        # Denominator: Σ(a∈A(j))exp(b̂j·b̂a/τ)
        denominator = torch.sum(
            exp_sim * negative_mask.float(),
            dim=-1,
            keepdim=True
        ) + 1e-8
        
        # Compute loss for each positive pair
        log_prob = sim_matrix - torch.log(denominator)  # (B, N, N)
        
        # Apply span coefficients and positive mask
        loss = -(span_coef * log_prob * positive_mask.float())  # (B, N, N)
        
        # Normalize by number of positive pairs per box
        num_positives = positive_mask.float().sum(dim=-1)  # (B, N)
        loss = loss.sum(dim=-1) / (num_positives + 1e-8)  # (B, N)
        
        # Average over boxes and batch
        loss = loss.mean()
        
        return loss

    def forward(
        self,
        tag_logits: torch.Tensor,          # (B, L, V)
        tag_targets: torch.Tensor,         # (B, L)
        box_features: torch.Tensor,        # (B, N, D)
        tag_features: torch.Tensor,        # (B, T, D)
        box_indices: torch.Tensor,         # (B, N)
        data_tag_mask: torch.Tensor,       # (B, T)
        empty_mask: torch.Tensor,          # (B, T)
        row_spans: Optional[torch.Tensor] = None,  # (B, N, N)
        col_spans: Optional[torch.Tensor] = None   # (B, N, N)
    ) -> Dict[str, torch.Tensor]:
        """Total Loss (논문 Equation 7)"""
        # 1. Classification Loss
        cls_loss = F.cross_entropy(
            tag_logits.reshape(-1, tag_logits.size(-1)).to(torch.float32),
            tag_targets.reshape(-1),
            ignore_index=self.pad_token_id
        )
        
        # 2. Layout Pointer Loss
        ptr_loss = self.compute_pointer_loss(
            box_features=box_features,
            tag_features=tag_features,
            box_indices=box_indices,
            data_tag_mask=data_tag_mask
        )
        
        # 3. Empty Pointer Loss
        empty_ptr_loss = self.compute_empty_pointer_loss(
            tag_features=tag_features,
            empty_mask=empty_mask
        )
        
        # 4. Span-aware Contrastive Loss
        if row_spans is not None and col_spans is not None:
            row_contr_loss = self.compute_span_aware_contrastive_loss(
                features=box_features,
                spans=row_spans,
                direction='row'
            )
            col_contr_loss = self.compute_span_aware_contrastive_loss(
                features=box_features,
                spans=col_spans,
                direction='col'
            )
        else:
            row_contr_loss = torch.tensor(0.0, device=cls_loss.device)
            col_contr_loss = torch.tensor(0.0, device=cls_loss.device)
        
        # 5. Total Loss (Equation 7)
        total_loss = (
            self.lambda_cls * cls_loss +
            self.lambda_ptr * ptr_loss +
            self.lambda_empty_ptr * empty_ptr_loss +
            self.lambda_row_contr * row_contr_loss +
            self.lambda_col_contr * col_contr_loss
        )
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'ptr_loss': ptr_loss,
            'empty_ptr_loss': empty_ptr_loss,
            'row_contr_loss': row_contr_loss,
            'col_contr_loss': col_contr_loss
        }