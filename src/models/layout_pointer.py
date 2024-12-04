import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LayoutPointer(nn.Module):
    """
    Layout Pointer Module (논문 Section 3.5)
    
    decoder의 last hidden states를 이용해 box-tag associations 생성:
    1. Split hidden states into box/tag features
    2. Project features using linear transformations
    3. Compute attention scores with temperature scaling
    """
    def __init__(self, feature_dim: int, temperature: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Feature projections (Equation 1)
        self.box_proj = nn.Linear(feature_dim, feature_dim)  # projb
        self.tag_proj = nn.Linear(feature_dim, feature_dim)  # projt
        
        # Special empty embedding and its projection (b̄0)
        self.empty_embedding = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.empty_proj = nn.Linear(feature_dim, feature_dim)
    
    # def forward(
    #     self,
    #     decoder_hidden_states: torch.Tensor,  # (B, N+T, D)
    #     num_boxes: int,                       # B (number of boxes)
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Args:
    #         decoder_hidden_states: Decoder의 last hidden states {hi}
    #         num_boxes: 실제 box 수 (B)
            
    #     Returns:
    #         pointer_logits: Box-tag attention scores (b̄j·t̄k/τ) (B, B, T)
    #         empty_logits: Empty cell detection scores (B, T)
    #     """
    #     B = decoder_hidden_states.size(0)
        
    #     # print(f"decoder_hidden_states shape: {decoder_hidden_states.shape}")
    #     # print(f"decoder hidden states range: [{decoder_hidden_states.min():.4f}, {decoder_hidden_states.max():.4f}]")
    #     # 1. Split hidden states into box/tag features
    #     box_features = decoder_hidden_states[:, :num_boxes]  # {bj}
    #     tag_features = decoder_hidden_states[:, num_boxes:]  # {tk}
        
    #     # # print ranges before projection
    #     # print(f"box_features before proj: [{box_features.min():.4f}, {box_features.max():.4f}]")
    #     # print(f"tag_features before proj: [{tag_features.min():.4f}, {tag_features.max():.4f}]")
        
    #     # 2. Project features (Equation 1)
    #     box_features = self.box_proj(box_features)  # b̄j = projb(bj)
    #     tag_features = self.tag_proj(tag_features)  # t̄k = projt(tk)
        
    #     # # print ranges after projection
    #     # print(f"box_features after proj: [{box_features.min():.4f}, {box_features.max():.4f}]")
    #     # print(f"tag_features after proj: [{tag_features.min():.4f}, {tag_features.max():.4f}]")
        
    #     # 3. Compute attention scores (b̄j·t̄k/τ)
    #     pointer_logits = torch.matmul(box_features, tag_features.transpose(-2, -1))
    #     pointer_logits = pointer_logits / self.temperature
    #     # print(f"pointer_logits range: [{pointer_logits.min():.4f}, {pointer_logits.max():.4f}]")
        
    #     # 4. Empty cell detection (Equation 3)
    #     empty_embedding = self.empty_proj(self.empty_embedding)  # b̄0
    #     empty_logits = torch.matmul(tag_features, empty_embedding.transpose(-2, -1)).squeeze(-1)
    #     empty_logits = torch.sigmoid(empty_logits)
        
    #     return pointer_logits, empty_logits
    
    def forward(self, decoder_hidden_states, num_boxes):
        B = decoder_hidden_states.size(0)
        
        # 1. Split hidden states
        box_features = decoder_hidden_states[:, :num_boxes]  # (B, N, D)
        tag_features = decoder_hidden_states[:, num_boxes:]  # (B, T, D)
        
        # 2. Project features
        box_features = self.box_proj(box_features)  # (B, N, D)
        tag_features = self.tag_proj(tag_features)  # (B, T, D)
        
        # print(f"box_features range: [{box_features.min():.4f}, {box_features.max():.4f}]")
        # print(f"tag_features range: [{tag_features.min():.4f}, {tag_features.max():.4f}]")
        
        box_features = F.normalize(box_features, dim=-1)
        tag_features = F.normalize(tag_features, dim=-1)
        
        # print(f"box_features after normalize range: [{box_features.min():.4f}, {box_features.max():.4f}]")
        # print(f"tag_features after normalize range: [{tag_features.min():.4f}, {tag_features.max():.4f}]")
        
        # 3. Return projected features directly
        # matmul 제거하고 projected features 반환
        return box_features, tag_features, self.empty_proj(self.empty_embedding)